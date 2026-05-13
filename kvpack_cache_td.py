"""kvpack_cache_td.py — TD 侧明文 KV block 缓存。

命名说明:`kvpack_cache` 容易和"kvpack 这种 KV cache 存储格式"混淆,
但这里的 cache 指的是**运行时驻留 TD 内存的 (K, V) 缓存层**,服务于 decode
阶段 read_layer_frame 的快速命中。底层 kvpack.bin 完整密文存盘不受影响。

设计要点
--------
- **键值**: key=(layer_index, frame_index), value=(K, V, header) 明文 tensors
- **6 个策略**(decode 侧):
    a · NoCache           : 不缓存,每次 FETCH 走 bridge/磁盘
    b · LRU(N)            : N-block LRU,所有 admit,evict 按访问顺序
    c · PerLayerPrefix(K) : 每层前 K 帧,N = K × num_layers
    d · MidLayerPrefix(K) : 中段层 [lo, hi] 的前 K 帧,N = K × (hi-lo+1)
    e · ShallowFirst(N)   : 前 N 个 block 按 (L 主, F 次) 顺序 pin,浅层优先
    f · DeepestFirst(N)   : 前 N 个 block 按 (L 主反向, F 次) 顺序 pin,深层优先
- **驱逐规则**: LRU 末位淘汰,但**被 is_pinned 标记的永不驱逐**;
                若所有 cache 槽位都被 pinned 占满,新 admission 直接 reject
- **encode 端 pre-warm**: encode 阶段每个 (L, F) block 产出后调用
                          maybe_admit,由策略决定是否进 cache
- **decode 端透明拦截**: CachingReader 包装现有 reader,hit 直接返回,
                         miss 落到 underlying.read_layer_frame
- **加密不变**: encode 阶段的 encrypt → offload 完全走原路径,
                cache 只是把已经在 TD 内存里的明文 tensor 多保留一份引用
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

BlockKey = Tuple[int, int]   # (layer_index, frame_index)


# ═══════════════════════════════════════════════════════════════════════════
# Policy 整数常量(供 CLI / 配置文件直接使用)
# 旧字母编码 a..f 已废弃;若仍想以字母引用,用下方的 _STR_TO_CODE 映射。
# ═══════════════════════════════════════════════════════════════════════════

DECODE_POLICY_NONE              = 0   # 旧 a · 不缓存
DECODE_POLICY_LRU               = 1   # 旧 b · 标准 LRU
DECODE_POLICY_PER_LAYER_PREFIX  = 2   # 旧 c · 每层前 K 帧
DECODE_POLICY_MID_LAYER_PREFIX  = 3   # 旧 d · 中段层前 K 帧
DECODE_POLICY_SHALLOW_FIRST     = 4   # 旧 e · 浅层优先光栅
DECODE_POLICY_DEEPEST_FIRST     = 5   # 旧 f · 深层优先光栅

VALID_POLICY_CODES = (
    DECODE_POLICY_NONE,
    DECODE_POLICY_LRU,
    DECODE_POLICY_PER_LAYER_PREFIX,
    DECODE_POLICY_MID_LAYER_PREFIX,
    DECODE_POLICY_SHALLOW_FIRST,
    DECODE_POLICY_DEEPEST_FIRST,
)

# 旧字母 → 新整数,backward compat
_STR_TO_CODE = {
    "a": DECODE_POLICY_NONE,
    "b": DECODE_POLICY_LRU,
    "c": DECODE_POLICY_PER_LAYER_PREFIX,
    "d": DECODE_POLICY_MID_LAYER_PREFIX,
    "e": DECODE_POLICY_SHALLOW_FIRST,
    "f": DECODE_POLICY_DEEPEST_FIRST,
}


def _normalize_policy(policy: Union[int, str]) -> int:
    """允许传 'a'..'f' 或 0..5,统一返回整数 code。"""
    if isinstance(policy, str):
        key = policy.strip().lower()
        if key in _STR_TO_CODE:
            return _STR_TO_CODE[key]
        # 也允许 "0".."5" 字符串
        try:
            n = int(key)
            if n in VALID_POLICY_CODES:
                return n
        except ValueError:
            pass
        raise ValueError(
            f"unknown policy {policy!r}; expected 0..5 or one of a..f"
        )
    if isinstance(policy, int):
        if policy in VALID_POLICY_CODES:
            return policy
    raise ValueError(
        f"unknown policy {policy!r}; expected 0..5 or one of a..f"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 策略基类
# ═══════════════════════════════════════════════════════════════════════════

class CachePolicy(ABC):
    """缓存策略接口。

    should_admit : 该 (L, F) 是否允许进 cache(放行性)
    is_pinned    : 该 (L, F) 是否禁止被驱逐(粘性)

    对于 c/d/e/f 这种静态策略,should_admit == is_pinned;
    对于 LRU,所有都 should_admit 但无任何 is_pinned。
    """
    name: str = "abstract"

    @abstractmethod
    def should_admit(self, layer: int, frame: int) -> bool: ...

    @abstractmethod
    def is_pinned(self, layer: int, frame: int) -> bool: ...

    def pin_count(self) -> int:
        """静态策略返回 pin 集合大小;动态策略返回 0。"""
        return 0

    def describe(self) -> str:
        return self.name


# ═══════════════════════════════════════════════════════════════════════════
# 6 个策略
# ═══════════════════════════════════════════════════════════════════════════

class NoCachePolicy(CachePolicy):
    """0 · 不缓存,接口仍然存在以便代码路径统一。"""
    name = "0/no_cache"

    def should_admit(self, layer, frame): return False
    def is_pinned(self, layer, frame): return False


class LRUPolicy(CachePolicy):
    """1 · 标准 LRU,容量 N。无 pin,evict 按访问时间末位。"""
    name = "1/lru"

    def __init__(self, capacity: int):
        self.capacity = int(capacity)

    def should_admit(self, layer, frame): return True
    def is_pinned(self, layer, frame): return False
    def describe(self): return f"{self.name}(capacity={self.capacity})"


class PerLayerPrefixPolicy(CachePolicy):
    """2 · 每层前 K 帧固定保留,总 N = K × num_layers。"""
    name = "2/per_layer_prefix"

    def __init__(self, k_per_layer: int, num_layers: int):
        self.k = int(k_per_layer)
        self.num_layers = int(num_layers)

    def should_admit(self, layer, frame):
        return 0 <= layer < self.num_layers and 0 <= frame < self.k

    def is_pinned(self, layer, frame):
        return self.should_admit(layer, frame)

    def pin_count(self): return self.k * self.num_layers
    def describe(self): return f"{self.name}(K={self.k}, layers={self.num_layers})"


class MidLayerPrefixPolicy(CachePolicy):
    """3 · 中段层 [lo, hi] 的前 K 帧固定保留,N = K × (hi - lo + 1)。"""
    name = "3/mid_layer_prefix"

    def __init__(self, k_per_layer: int, layer_lo: int, layer_hi: int):
        self.k = int(k_per_layer)
        self.lo = int(layer_lo)
        self.hi = int(layer_hi)

    def should_admit(self, layer, frame):
        return self.lo <= layer <= self.hi and 0 <= frame < self.k

    def is_pinned(self, layer, frame):
        return self.should_admit(layer, frame)

    def pin_count(self): return self.k * (self.hi - self.lo + 1)
    def describe(self): return f"{self.name}(K={self.k}, L=[{self.lo},{self.hi}])"


class _RasterPinPolicy(CachePolicy):
    """e/f 的共享父类:把 pin 集合预先算成 set,查询 O(1)。"""
    name = "raster_abstract"

    def __init__(self):
        self.pin_set: Set[BlockKey] = set()

    def should_admit(self, layer, frame):
        return (int(layer), int(frame)) in self.pin_set

    def is_pinned(self, layer, frame):
        return self.should_admit(layer, frame)

    def pin_count(self): return len(self.pin_set)


class ShallowFirstPolicy(_RasterPinPolicy):
    """4 · 浅层优先,(L 升序, F 升序) 光栅顺序填充前 N 个 block。"""
    name = "4/shallow_first"

    def __init__(self, total_budget: int, num_layers: int, num_chunks: int):
        super().__init__()
        self.total_budget = int(total_budget)
        self.num_layers = int(num_layers)
        self.num_chunks = int(num_chunks)
        filled = 0
        for L in range(num_layers):
            if filled >= total_budget: break
            for F in range(num_chunks):
                if filled >= total_budget: break
                self.pin_set.add((L, F))
                filled += 1

    def describe(self):
        return f"{self.name}(N={len(self.pin_set)}/{self.total_budget}, L-major)"


class DeepestFirstPolicy(_RasterPinPolicy):
    """5 · 深层优先,(L 降序, F 升序) 光栅顺序填充前 N 个 block。"""
    name = "5/deepest_first"

    def __init__(self, total_budget: int, num_layers: int, num_chunks: int):
        super().__init__()
        self.total_budget = int(total_budget)
        self.num_layers = int(num_layers)
        self.num_chunks = int(num_chunks)
        filled = 0
        for L in range(num_layers - 1, -1, -1):
            if filled >= total_budget: break
            for F in range(num_chunks):
                if filled >= total_budget: break
                self.pin_set.add((L, F))
                filled += 1

    def describe(self):
        return f"{self.name}(N={len(self.pin_set)}/{self.total_budget}, L-major rev)"


# ═══════════════════════════════════════════════════════════════════════════
# 监控器
# ═══════════════════════════════════════════════════════════════════════════

class CacheMonitor:
    """累积统计:hit / miss / admit / reject / evict + per-layer 分布。"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or globals()["logger"]
        self.hits = 0
        self.misses = 0
        self.admits = 0
        self.rejects = 0
        self.evictions = 0
        self.bytes_resident = 0
        self.bytes_peak = 0
        self.layer_hits: Dict[int, int] = defaultdict(int)
        self.layer_misses: Dict[int, int] = defaultdict(int)

    def record_hit(self, key: BlockKey):
        self.hits += 1
        self.layer_hits[key[0]] += 1

    def record_miss(self, key: BlockKey):
        self.misses += 1
        self.layer_misses[key[0]] += 1

    def record_admit(self, key: BlockKey, nbytes: int):
        self.admits += 1
        self.bytes_resident += nbytes
        if self.bytes_resident > self.bytes_peak:
            self.bytes_peak = self.bytes_resident

    def record_reject(self, key: BlockKey):
        self.rejects += 1

    def record_evict(self, key: BlockKey, nbytes: int):
        self.evictions += 1
        self.bytes_resident -= nbytes

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def log_summary(self, cache: "TDBlockCache") -> None:
        sep = "=" * 70
        log = self.logger.info
        total = self.hits + self.misses
        hr = self.hits / total * 100 if total > 0 else 0.0
        log(sep)
        log(f"  TD Block Cache Summary  policy={cache.policy.describe()}")
        log(sep)
        log(
            f"  capacity={cache.capacity}  resident={len(cache)} blocks  "
            f"bytes={self.bytes_resident/1024/1024:.2f}MB "
            f"(peak {self.bytes_peak/1024/1024:.2f}MB)"
        )
        log(
            f"  requests={total}  hits={self.hits}  misses={self.misses}  "
            f"hit_rate={hr:.1f}%"
        )
        log(
            f"  admits={self.admits}  rejects={self.rejects}  "
            f"evictions={self.evictions}"
        )
        if self.layer_hits or self.layer_misses:
            log("  per-layer hit rate (layers with traffic only):")
            all_layers = sorted(set(self.layer_hits.keys()) | set(self.layer_misses.keys()))
            for L in all_layers:
                h = self.layer_hits.get(L, 0)
                m = self.layer_misses.get(L, 0)
                t = h + m
                rate = h / t * 100 if t > 0 else 0.0
                log(
                    f"    L={L:>3}  hits={h:>4}  misses={m:>4}  rate={rate:>5.1f}%"
                )
        log(sep)


# ═══════════════════════════════════════════════════════════════════════════
# 核心 Cache
# ═══════════════════════════════════════════════════════════════════════════

class TDBlockCache:
    """
    LRU + pinning 缓存。

    存储:dict[(L, F)] -> (k_tensor, v_tensor, header_dict)
    LRU 顺序通过 OrderedDict 维护,值是该条目字节数(便于驱逐统计)。
    """

    def __init__(self, policy: CachePolicy, capacity: int,
                 monitor: Optional[CacheMonitor] = None):
        self.policy = policy
        self.capacity = max(0, int(capacity))
        self.monitor = monitor or CacheMonitor()

        self._store: Dict[BlockKey, Tuple[Any, Any, dict]] = {}
        self._lru: "OrderedDict[BlockKey, int]" = OrderedDict()   # key -> nbytes

    def __len__(self) -> int:
        return len(self._store)

    def get(self, layer: int, frame: int):
        """返回 (k, v, header) 或 None。命中则把条目移到 MRU。"""
        key = (int(layer), int(frame))
        if key in self._store:
            self._lru.move_to_end(key)
            self.monitor.record_hit(key)
            return self._store[key]
        self.monitor.record_miss(key)
        return None

    def maybe_admit(self, layer: int, frame: int,
                    k_tensor, v_tensor, header: dict) -> bool:
        """按策略尝试接纳。返回 True 表示已在或新进 cache。"""
        key = (int(layer), int(frame))
        if not self.policy.should_admit(layer, frame):
            self.monitor.record_reject(key)
            return False

        if key in self._store:
            self._lru.move_to_end(key)
            return True

        # CPU + contiguous 化(若已是,等价 no-op)
        k = k_tensor.detach().to("cpu").contiguous()
        v = v_tensor.detach().to("cpu").contiguous()
        nbytes = (k.element_size() * k.numel()) + (v.element_size() * v.numel())

        # 腾位
        while len(self._store) >= self.capacity:
            evict_key = self._find_evict_candidate()
            if evict_key is None:
                # 所有条目都是 pinned,无法腾位
                self.monitor.record_reject(key)
                return False
            freed = self._lru.pop(evict_key)
            del self._store[evict_key]
            self.monitor.record_evict(evict_key, freed)

        self._store[key] = (k, v, header)
        self._lru[key] = nbytes
        self.monitor.record_admit(key, nbytes)
        return True

    def _find_evict_candidate(self) -> Optional[BlockKey]:
        """返回 LRU 端最早非 pinned 的 key;若全 pinned 则返回 None。"""
        for key in self._lru:
            if not self.policy.is_pinned(*key):
                return key
        return None

    def clear(self) -> None:
        self._store.clear()
        self._lru.clear()

    def stats(self) -> dict:
        return {
            "policy": self.policy.describe(),
            "capacity": self.capacity,
            "resident": len(self._store),
            "bytes_resident": self.monitor.bytes_resident,
            "bytes_peak": self.monitor.bytes_peak,
            "hits": self.monitor.hits,
            "misses": self.monitor.misses,
            "hit_rate": self.monitor.hit_rate,
            "admits": self.monitor.admits,
            "rejects": self.monitor.rejects,
            "evictions": self.monitor.evictions,
        }

    def log_summary(self) -> None:
        self.monitor.log_summary(self)


# ═══════════════════════════════════════════════════════════════════════════
# CachingReader:透明包装 KVPackReader / KVPackBridgeClient
# ═══════════════════════════════════════════════════════════════════════════

class CachingReader:
    """
    包装现有 reader,把 read_layer_frame 改为先查 cache。
    其他属性(common_metadata / by_layer_frame / frames / _mmap 等)
    通过 __getattr__ 透传。

    注意:`_assemble_per_layer_kv` 的 parallel decrypt 路径直接访问
    `reader._mmap`,会绕过本 cache。Phase 1 不优化该路径;若启用
    crypto.parallel_workers>1,需要手工把 cache 检查加进去。
    """

    def __init__(self, underlying, cache: TDBlockCache):
        object.__setattr__(self, "_u", underlying)
        object.__setattr__(self, "_c", cache)

    def read_layer_frame(self, layer_index: int, frame_index: int, **kw):
        cached = self._c.get(layer_index, frame_index)
        if cached is not None:
            return cached
        k, v, header = self._u.read_layer_frame(layer_index, frame_index, **kw)
        self._c.maybe_admit(layer_index, frame_index, k, v, header)
        return k, v, header

    def close(self):
        return self._u.close()

    def __getattr__(self, name):
        return getattr(self._u, name)


# ═══════════════════════════════════════════════════════════════════════════
# 工厂
# ═══════════════════════════════════════════════════════════════════════════

def build_policy(
    policy: Union[int, str],
    decode_memory: int,
    *,
    num_layers: int,
    num_chunks: int,
    mid_layer_lo: int = 10,
    mid_layer_hi: int = 19,
) -> Tuple[CachePolicy, int]:
    """返回 (policy, capacity)。

    policy 可传整数 0..5 或旧字母 'a'..'f'(自动归一化)。
    """
    code = _normalize_policy(policy)

    if code == DECODE_POLICY_NONE:
        return NoCachePolicy(), 0

    if code == DECODE_POLICY_LRU:
        if decode_memory <= 0:
            raise ValueError("policy 1 (LRU) requires decode_memory > 0")
        return LRUPolicy(decode_memory), decode_memory

    if code == DECODE_POLICY_PER_LAYER_PREFIX:
        if decode_memory < num_layers:
            logger.warning(
                f"decode_memory={decode_memory} < num_layers={num_layers}; "
                f"K = decode_memory // num_layers = 0, nothing will be pinned."
            )
        k = decode_memory // num_layers
        p = PerLayerPrefixPolicy(k, num_layers)
        return p, max(p.pin_count(), 1)

    if code == DECODE_POLICY_MID_LAYER_PREFIX:
        num_mid = mid_layer_hi - mid_layer_lo + 1
        if num_mid <= 0:
            raise ValueError(
                f"invalid mid-layer range [{mid_layer_lo}, {mid_layer_hi}]"
            )
        if decode_memory < num_mid:
            logger.warning(
                f"decode_memory={decode_memory} < num_mid={num_mid}; K=0."
            )
        k = decode_memory // num_mid
        p = MidLayerPrefixPolicy(k, mid_layer_lo, mid_layer_hi)
        return p, max(p.pin_count(), 1)

    if code == DECODE_POLICY_SHALLOW_FIRST:
        p = ShallowFirstPolicy(decode_memory, num_layers, num_chunks)
        return p, max(p.pin_count(), 1)

    if code == DECODE_POLICY_DEEPEST_FIRST:
        p = DeepestFirstPolicy(decode_memory, num_layers, num_chunks)
        return p, max(p.pin_count(), 1)

    raise ValueError(f"unhandled policy code {code}")


def make_td_cache(
    policy: Union[int, str],
    decode_memory: int,
    *,
    num_layers: int,
    num_chunks: int,
    mid_layer_lo: int = 10,
    mid_layer_hi: int = 19,
    logger_obj: Optional[logging.Logger] = None,
) -> Optional[TDBlockCache]:
    """policy = 0 (NONE) 或 decode_memory <= 0 时返回 None。"""
    log = logger_obj or logger
    code = _normalize_policy(policy)
    if code == DECODE_POLICY_NONE or decode_memory <= 0:
        log.info(
            f"[td_cache] disabled (policy={code}, decode_memory={decode_memory})"
        )
        return None

    pol, capacity = build_policy(
        code, decode_memory,
        num_layers=num_layers, num_chunks=num_chunks,
        mid_layer_lo=mid_layer_lo, mid_layer_hi=mid_layer_hi,
    )
    monitor = CacheMonitor(logger=log)
    cache = TDBlockCache(pol, capacity=capacity, monitor=monitor)
    log.info(
        f"[td_cache] policy={pol.describe()}  capacity={capacity}  "
        f"pin_count={pol.pin_count()}  decode_memory={decode_memory}"
    )
    return cache


# ═══════════════════════════════════════════════════════════════════════════
# 自检
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    class FakeTensor:
        def __init__(self, tag, nbytes=8):
            self.tag = tag
            self._nbytes = nbytes
        def detach(self): return self
        def to(self, *_): return self
        def contiguous(self): return self
        def element_size(self): return 4
        def numel(self): return self._nbytes // 4

    def kv(L, F): return FakeTensor((L, F, "k")), FakeTensor((L, F, "v"))

    # ── 0/NONE ──
    cache = make_td_cache(DECODE_POLICY_NONE, 0, num_layers=28, num_chunks=50)
    assert cache is None
    print("policy 0/NONE: returns None ✓")

    # ── 1/LRU(4) ──
    cb = make_td_cache(DECODE_POLICY_LRU, 4, num_layers=28, num_chunks=50)
    assert cb is not None
    for L in range(6):
        cb.maybe_admit(L, 0, *kv(L, 0), {})
    assert len(cb) == 4
    assert cb.get(0, 0) is None        # evicted
    assert cb.get(5, 0) is not None
    print(f"policy 1/LRU: resident={len(cb)} evictions={cb.monitor.evictions} ✓")

    # ── 2/PerLayerPrefix(K=2 from decode_memory=56) ──
    cc = make_td_cache(DECODE_POLICY_PER_LAYER_PREFIX, 56,
                        num_layers=28, num_chunks=10)
    for L in range(28):
        for F in range(5):
            cc.maybe_admit(L, F, *kv(L, F), {})
    assert len(cc) == 56
    assert cc.get(0, 0) is not None
    assert cc.get(0, 1) is not None
    assert cc.get(0, 2) is None
    assert cc.get(27, 1) is not None
    print(f"policy 2/PER_LAYER_PREFIX: resident={len(cc)} ✓")

    # ── 3/MidLayerPrefix(L=[10,19], K=2 from decode_memory=20) ──
    cd = make_td_cache(DECODE_POLICY_MID_LAYER_PREFIX, 20,
                        num_layers=28, num_chunks=10)
    for L in range(28):
        for F in range(5):
            cd.maybe_admit(L, F, *kv(L, F), {})
    assert len(cd) == 20
    assert cd.get(5, 0) is None
    assert cd.get(10, 0) is not None
    assert cd.get(19, 1) is not None
    assert cd.get(20, 0) is None
    assert cd.get(10, 2) is None
    print(f"policy 3/MID_LAYER_PREFIX: resident={len(cd)} ✓")

    # ── 4/ShallowFirst(N=20, 28×10) ──
    ce = make_td_cache(DECODE_POLICY_SHALLOW_FIRST, 20,
                        num_layers=28, num_chunks=10)
    for L in range(5):
        for F in range(10):
            ce.maybe_admit(L, F, *kv(L, F), {})
    assert len(ce) == 20
    assert ce.get(0, 0) is not None
    assert ce.get(1, 9) is not None
    assert ce.get(2, 0) is None
    print(f"policy 4/SHALLOW_FIRST: resident={len(ce)} ✓")

    # ── 5/DeepestFirst(N=20, 28×10) ──
    cf = make_td_cache(DECODE_POLICY_DEEPEST_FIRST, 20,
                        num_layers=28, num_chunks=10)
    for L in range(22, 28):
        for F in range(10):
            cf.maybe_admit(L, F, *kv(L, F), {})
    assert len(cf) == 20
    assert cf.get(27, 0) is not None
    assert cf.get(26, 9) is not None
    assert cf.get(25, 0) is None
    print(f"policy 5/DEEPEST_FIRST: resident={len(cf)} ✓")

    # ── 旧字母 backward-compat ──
    legacy = make_td_cache("c", 56, num_layers=28, num_chunks=10)
    assert legacy is not None and legacy.policy.name.startswith("2/")
    print(f"legacy 'c' → policy {legacy.policy.name} ✓")

    # ── Verify hit accounting for code=2 (relative count) ──
    hits_before = cc.monitor.hits
    for L in range(28):
        for F in range(2):
            assert cc.get(L, F) is not None
    assert cc.monitor.hits - hits_before == 56
    print(f"policy 2 full-hit replay: +{cc.monitor.hits - hits_before} hits ✓")

    print("\nall 6 policies validated. printing one example summary:")
    cc.log_summary()