# TDX_StreamingVideoLM 仓库结构说明

本仓库围绕 **LLaVA-OneVision** 构建了一个“视频先编码为 KV Cache，再按需解码问答”的实验流程，核心由 5 个 Python 脚本组成：

1. `main_td.py`：主流程入口，串联编码、保存、重载模型与解码。
2. `kvcache_generate_td.py`：视频采样、分块编码并生成/保存 KV Cache。
3. `kvcache_retrieve_td.py`：从磁盘加载 KV Cache，并在文本后缀条件下执行增量解码。
4. `zhz_model_eval_utils.py`：向模型/视觉塔/语言模型注入 hook，采集 prefill/decode 时间统计。
5. `zhz_hardware_eval_utils.py`：采集 CPU/内存曲线、事件打点与资源图表输出。

---

## 1. 端到端执行路径（`main_td.py`）

`main_td.py` 负责把所有模块粘起来：

- 先记录系统信息并设置 HF 镜像地址。
- 在 `measure_resources` 上下文中：
  - 加载模型（编码阶段）。
  - 加载并采样视频。
  - 调用 `encode_video` 逐块编码，得到 `past_key_values` 形式的 KV Cache。
  - 调用 `save_kv_cache` 将 cache + metadata 存盘。
  - 释放模型后再次加载模型（解码阶段）。
  - 调用 `decode_kvcache` 仅基于 KV Cache + 文本问题生成答案。

这个流程体现了项目目标：**把昂贵的视频编码前置并缓存，后续问答只做文本续写以减少重复计算**。

---

## 2. 编码侧模块（`kvcache_generate_td.py`）

### 2.1 关键常量与上下文约束

- `ENCODE_PREFIX`：编码前缀，强调“先理解视频并记住关键事件”。
- `VIDEO_PLACEHOLDER = "<video>"`：强制文本中带视频占位符，避免 token 与视觉特征错位。

`_build_encode_text()` 保证首块编码含 `<video>` 语境；后续块仍保留视频占位符，维持上下文一致性。

### 2.2 视频采样

`load_video()` 使用 `decord.VideoReader` 按 `sample_fps` 进行抽帧，返回 numpy 帧序列。

### 2.3 KV Cache 构建

`encode_video()` 是核心：

- 将视频按 `chunk_size` 切块。
- 每个 chunk 通过 processor 构造模型输入。
- 首块使用完整编码前缀，后续块用 `"<video>"`。
- 若已有历史 cache，则显式扩展 `attention_mask` 到 `past_seq_len + current_seq_len`。
- 模型前向后拿 `outputs.past_key_values` 回写为新 cache，逐块累积。

### 2.4 持久化

`save_kv_cache()` 会把 cache 递归 `detach+to(cpu)` 后与 metadata 一起存为 `torch.save` 文件，metadata 包含层数、dtype、past 序列长度、模型名等字段，用于解码阶段校验。

---

## 3. 解码侧模块（`kvcache_retrieve_td.py`）

### 3.1 加载与兼容

`load_kv_cache()` 兼容不同 `torch.load` 方式（包括 `weights_only=True` 的回退逻辑），并返回 `(kv_cache, metadata)`。

### 3.2 上下文拼接

`decode_kvcache()` 会：

- 读取 cache 并校验 `model_name_or_path` 与当前模型一致。
- 将 cache 迁移到当前 device。
- 构造文本后缀：`问题：... 回答：...`。
- 依据 `past_seq_len` + 当前输入长度重建完整 `attention_mask`。

### 3.3 生成策略

不是直接 `model.generate`，而是手写循环：

- 首 token 从 prefill logits 采样。
- 后续 token 逐步 decode（每步传入上一 token + past）。
- 支持 `temperature`、`top_p` nucleus sampling、`repetition_penalty`。
- 满足 EOS 且达到 `min_new_tokens` 才提前停止。

这让实验可控，便于观察 prefill/decode 各阶段开销。

---

## 4. 模型计时模块（`zhz_model_eval_utils.py`）

该模块通过 hook 将计时拆解到不同子路径：

- 顶层 model：区分 prefill 与 decode 的耗时和调用次数。
- `vision_tower`：记录视觉编码耗时与输入/输出形状。
- `language_model`：记录 LLM prefill/decode 耗时。

`inject_timing_hook_to_model()` 负责安装 hooks，`remove_timing_hooks_from_model()` 负责清理，`get_timing_stats()/print_timing_stats()` 输出吞吐指标（如 FTPS、TPS）。

---

## 5. 硬件资源监控模块（`zhz_hardware_eval_utils.py`）

`measure_resources()` 提供上下文管理器：

- 后台线程周期采样进程 CPU/内存。
- 支持 `mark_event()` 记录阶段事件（例如 load/visual/prefill）。
- 结束时输出统计，并可调用 `plot_resource_usage()` 生成图。

`plot_resource_usage()` 当前主图是 **内存曲线**，并叠加事件竖线与系统信息。

---

## 6. 目录层次（当前仓库）

当前项目是扁平结构（无子包目录）：

- 入口层：`main_td.py`
- 编解码核心：`kvcache_generate_td.py`、`kvcache_retrieve_td.py`
- 评测/观测工具：`zhz_model_eval_utils.py`、`zhz_hardware_eval_utils.py`

如果后续继续扩展，建议按 `pipeline/`、`metrics/`、`scripts/` 分层并补齐 `README` 与 `requirements`。
