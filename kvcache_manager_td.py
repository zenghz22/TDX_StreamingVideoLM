"""
KV Cache Manager：encode 阶段内存受限的 KV Cache 调度器。

两种工作模式
------------
1. 全局注意力 + LRU 换入换出（window_size=None）
   - 每次 forward 需要所有历史 chunk KV，超出 max_in_memory 时 LRU evict 到磁盘。
   - 适合 chunk 数量少、内存稍宽裕的场景。
   - 注意：若 max_in_memory < 总 chunk 数，会有大量磁盘 IO（O(n²) 最坏情况）。

2. 滑动窗口注意力（window_size=W）
   - 每次 forward 只看最近 W 个 chunk 的 KV，窗口外的 chunk 永久驱逐（不再 reload）。
   - 内存严格限制为 O(W * chunk_kv_size)，适合长视频 + 极度受限的 TDX 场景。
   - 代价：chunk i 无法 attend 到超过 W 个 chunk 以前的内容。
   - RoPE position_ids 修正：通过 language_model 的 pre-hook 注入正确的绝对位置，
     保证新 token 与窗口内 KV 之间的相对位置计算正确。

超参数
------
max_in_memory : int
    内存中最多保留的 chunk delta KV 数量（LRU 驱逐）。
window_size : int | None
    滑动窗口大小（chunk 为单位）。None 表示全局注意力。
"""

import os
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any

import gc
import torch

from kvcache_retrieve_td import _load_single_safetensors_kv, _concat_kv_segments


# ---------------------------------------------------------------------------
# Position ID Hook（用于滑动窗口模式的 RoPE 修正）
# ---------------------------------------------------------------------------

class _PositionIdHook:
    """
    挂在 language_model.forward 上的 pre-hook。

    当 sliding window 模式下 past_key_values 仅包含窗口内的 KV 时，
    模型默认会把 position_ids 的起始位置计算为 window_seq_len，而非真实的
    full_merged_past_seq_len。这会导致 RoPE 位置编码偏差。

    此 hook 在 forward 前拦截并替换 kwargs["position_ids"]，确保新 token
    的绝对位置与其在完整序列中的真实位置一致。

    为什么这样有效（RoPE 数学保证）：
      Q 在位置 p_q，K 在位置 p_k，RoPE dot product = cos(p_q - p_k)。
      只要 Q 和 K 都用各自的真实绝对位置做 RoPE，相对距离就是正确的，
      无论中间是否有被驱逐的窗口外 token。
    """

    def __init__(self):
        self._start_pos: int = 0   # 当前 chunk 的全局起始 merged 位置
        self._active: bool = False
        self._handle = None

    def register(self, language_model):
        self._handle = language_model.register_forward_pre_hook(
            self._hook_fn, with_kwargs=True
        )

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def activate(self, full_merged_past_seq_len: int):
        """在每次 forward 前调用，设置本次 chunk 的正确起始位置。"""
        self._start_pos = full_merged_past_seq_len
        self._active = True

    def deactivate(self):
        self._active = False

    def _hook_fn(self, module, args, kwargs):
        if not self._active:
            return args, kwargs

        # LlavaOnevision 在 language_model.forward 时通过 inputs_embeds 传入
        # 已合并视觉特征的嵌入序列，其 shape[1] = merged_current_seq_len。
        embeds = kwargs.get("inputs_embeds")
        if embeds is None:
            return args, kwargs

        merged_len = embeds.shape[1]
        position_ids = torch.arange(
            self._start_pos,
            self._start_pos + merged_len,
            device=embeds.device,
        ).unsqueeze(0)  # [1, merged_len]
        kwargs["position_ids"] = position_ids
        return args, kwargs


# ---------------------------------------------------------------------------
# KVCacheManager
# ---------------------------------------------------------------------------

class KVCacheManager:
    """
    管理 encode 阶段各 chunk 的 delta KV cache，严格限制内存中 chunk 数量。

    Parameters
    ----------
    kv_cache_dir : str
        delta shard 文件的存放目录（safetensors 格式）。
    max_in_memory : int
        内存中最多同时保留的 chunk 数量（LRU 驱逐）。
    window_size : int | None
        滑动窗口大小。None = 全局注意力（所有历史 chunk）。
    device : str
        KV tensor 所在设备（encode 阶段通常为 "cpu"）。
    """

    def __init__(
        self,
        kv_cache_dir: str,
        max_in_memory: int = 2,
        window_size: Optional[int] = None,
        device: str = "cpu",
    ):
        if window_size is not None and max_in_memory < window_size:
            raise ValueError(
                f"max_in_memory ({max_in_memory}) 必须 >= window_size ({window_size})，"
                "否则每次 forward 都会触发不必要的磁盘 IO。"
            )

        self.kv_cache_dir = kv_cache_dir
        self.max_in_memory = max_in_memory
        self.window_size = window_size
        self.device = device

        # LRU cache: chunk_idx -> delta_kv (tuple of per-layer (k, v))
        self._memory: OrderedDict[int, tuple] = OrderedDict()

        # 已注册（已保存到磁盘）的 chunk 元数据
        # {chunk_idx: {"file": str, "seq_start": int, "seq_end": int}}
        self._registry: Dict[int, dict] = {}

        # 累积的完整 merged 序列长度（包含视觉 token 展开后的真实长度）
        # 用于 sliding window 模式的 position_ids 修正
        self._full_merged_seq_len: int = 0

        # position_ids hook（仅 sliding window 模式使用）
        self._pos_hook: Optional[_PositionIdHook] = None

        # 统计
        self.stats: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "disk_loads": 0,
        }

    # ------------------------------------------------------------------
    #  Hook 管理（sliding window 模式）
    # ------------------------------------------------------------------

    def install_position_hook(self, language_model):
        """
        注册 position_ids 修正 hook 到 language_model。
        仅在 window_size 不为 None 时需要调用。
        在 encode_video 开始时调用一次，结束后调用 remove_position_hook。
        """
        if self.window_size is None:
            return
        self._pos_hook = _PositionIdHook()
        self._pos_hook.register(language_model)

    def remove_position_hook(self):
        """移除 position_ids 修正 hook。"""
        if self._pos_hook is not None:
            self._pos_hook.remove()
            self._pos_hook = None

    # ------------------------------------------------------------------
    #  注册与内存管理
    # ------------------------------------------------------------------

    def register_chunk(
        self,
        chunk_idx: int,
        file_name: str,
        delta_kv: tuple,
        seq_start: int,
        seq_end: int,
    ):
        """
        注册一个刚编码好、已保存到磁盘的 chunk delta KV。

        Parameters
        ----------
        chunk_idx  : chunk 编号
        file_name  : 对应的 .safetensors 文件名（相对于 kv_cache_dir）
        delta_kv   : 仅当前 chunk 的增量 KV（tuple of per-layer (k, v)）
        seq_start  : 该 chunk 在完整 KV 序列中的起始 token 位置（文本侧）
        seq_end    : 该 chunk 在完整 KV 序列中的结束 token 位置（文本侧）
        """
        self._registry[chunk_idx] = {
            "file": file_name,
            "seq_start": seq_start,
            "seq_end": seq_end,
        }
        self._put_in_memory(chunk_idx, delta_kv)

        if self.window_size is not None:
            self._evict_outside_window()

    def update_full_merged_seq_len(self, delta_merged_len: int):
        """
        每次 forward 后调用，更新真实的 merged 序列长度。

        delta_merged_len = outputs.past_key_values[0][0].shape[-2] - window_kv_seq_len
        即本次 chunk forward 实际新增的 merged token 数（包含视觉展开）。
        """
        self._full_merged_seq_len += delta_merged_len

    def _put_in_memory(self, chunk_idx: int, delta_kv: tuple):
        """将 delta_kv 放入内存 LRU，超限时驱逐最旧的。"""
        if chunk_idx in self._memory:
            self._memory.move_to_end(chunk_idx)
            return

        while len(self._memory) >= self.max_in_memory:
            evicted_idx, evicted_kv = self._memory.popitem(last=False)
            self.stats["evictions"] += 1
            print(f"[KVCacheManager] LRU evict chunk {evicted_idx} from memory.")
            # 逐层显式 del tensor，断开所有引用后 gc 才能真正回收
            for layer_k, layer_v in evicted_kv:
                del layer_k, layer_v
            del evicted_kv
            gc.collect()

        self._memory[chunk_idx] = delta_kv

    def _evict_outside_window(self):
        """主动驱逐当前窗口以外的 chunk（已在磁盘，不再需要）。"""
        sorted_indices = sorted(self._registry.keys())
        if len(sorted_indices) <= self.window_size:
            return
        outside = sorted_indices[:-self.window_size]
        for idx in outside:
            if idx in self._memory:
                evicted_kv = self._memory.pop(idx)
                self.stats["evictions"] += 1
                print(f"[KVCacheManager] Window evict chunk {idx} from memory (outside window).")
                for layer_k, layer_v in evicted_kv:
                    del layer_k, layer_v
                del evicted_kv
        gc.collect()

    # ------------------------------------------------------------------
    #  磁盘 IO
    # ------------------------------------------------------------------

    def _load_from_disk(self, chunk_idx: int) -> tuple:
        if chunk_idx not in self._registry:
            raise KeyError(
                f"Chunk {chunk_idx} 未在 registry 中注册，无法从磁盘加载。"
            )
        file_name = self._registry[chunk_idx]["file"]
        path = os.path.join(self.kv_cache_dir, file_name)
        kv, _ = _load_single_safetensors_kv(path, map_location=self.device)
        self.stats["disk_loads"] += 1
        print(f"[KVCacheManager] Disk load chunk {chunk_idx} <- {path}")
        return kv

    def _get_chunk_kv(self, chunk_idx: int) -> tuple:
        """获取某 chunk 的 delta KV（优先内存，miss 则磁盘加载）。"""
        if chunk_idx in self._memory:
            self._memory.move_to_end(chunk_idx)
            self.stats["hits"] += 1
            return self._memory[chunk_idx]

        self.stats["misses"] += 1
        kv = self._load_from_disk(chunk_idx)
        self._put_in_memory(chunk_idx, kv)
        return kv

    # ------------------------------------------------------------------
    #  核心接口：供 encode_video 调用
    # ------------------------------------------------------------------

    def get_past_kv_for_forward(
        self, current_chunk_idx: int
    ) -> Tuple[Optional[tuple], int]:
        """
        返回 (past_key_values, window_kv_seq_len)，供 chunk current_chunk_idx 的 forward 使用。

        - past_key_values : 拼接好的历史 KV（窗口或全局），None 表示第一个 chunk。
        - window_kv_seq_len : past_key_values 的实际 seq 维度长度（用于 attention_mask 构建）。

        如果 window_size 不为 None，同时激活 position_ids hook（若已安装）。
        """
        if current_chunk_idx == 0 or not self._registry:
            return None, 0

        sorted_indices = sorted(self._registry.keys())

        if self.window_size is not None:
            needed_indices = sorted_indices[-self.window_size:]
        else:
            needed_indices = sorted_indices

        delta_kvs = [self._get_chunk_kv(idx) for idx in needed_indices]
        past_kv = _concat_kv_segments(delta_kvs)
        window_kv_seq_len = int(past_kv[0][0].shape[-2]) if past_kv else 0

        # 激活 position_ids hook，注入正确的起始位置
        if self._pos_hook is not None and self.window_size is not None:
            self._pos_hook.activate(self._full_merged_seq_len)

        return past_kv, window_kv_seq_len

    def deactivate_position_hook(self):
        """每次 forward 结束后调用，关闭 position hook。"""
        if self._pos_hook is not None:
            self._pos_hook.deactivate()

    # ------------------------------------------------------------------
    #  诊断与调试
    # ------------------------------------------------------------------

    def memory_status(self) -> Dict[str, Any]:
        return {
            "in_memory": sorted(self._memory.keys()),
            "on_disk": sorted(self._registry.keys()),
            "full_merged_seq_len": self._full_merged_seq_len,
            "max_in_memory": self.max_in_memory,
            "window_size": self.window_size,
            **self.stats,
        }

    def print_stats(self):
        info = self.memory_status()
        total_access = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_access * 100 if total_access > 0 else 0.0
        print("\n===== KVCacheManager Stats =====")
        print(f"  max_in_memory={self.max_in_memory}, window_size={self.window_size}")
        print(f"  In memory : {info['in_memory']}")
        print(f"  On disk   : {info['on_disk']}")
        print(f"  Hits      : {self.stats['hits']}  ({hit_rate:.1f}%)")
        print(f"  Misses    : {self.stats['misses']}")
        print(f"  Disk loads: {self.stats['disk_loads']}")
        print(f"  Evictions : {self.stats['evictions']}")
        print(f"  Full merged seq len (true): {info['full_merged_seq_len']}")

    def stats(self) -> Dict[str, Any]:
        """memory_status() 的别名，供外部直接调用 manager.stats()。"""
        return self.memory_status()