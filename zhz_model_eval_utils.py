import time


HOOK_HANDLES = []


def _init_timing_state(model):
    model._timing = {
        "model_prefill_calls": 0,
        "model_decode_calls": 0,
        "model_prefill_time_us": 0.0,
        "model_decode_total_us": 0.0,
        "vision_time_us": 0.0,
        "llm_prefill_time_us": 0.0,
        "llm_decode_total_us": 0.0,
        "llm_decode_calls": 0,
        "first_step_time_us": 0.0,
    }

    model._timing_event_cb = None

    model._start_time_ns = 0
    model._current_stage = "prefill"

    model.vision_tower._start_time_ns = 0
    model.language_model._start_time_ns = 0

    # 子模块 hook 里需要回写到顶层 model 的统计字典。
    model.vision_tower._timing_parent = model
    model.language_model._timing_parent = model

    model.vision_tower._encoder_inputs_shape = None
    model.vision_tower._encoder_output_shape = None


def _emit_event(model, label, payload=None):
    cb = getattr(model, "_timing_event_cb", None)
    if cb is None:
        return
    if payload is None:
        payload = {}
    cb(label, payload)


def _forward_pre_hook_model(model, args, kwargs):
    model._start_time_ns = time.perf_counter_ns()

    input_ids = kwargs.get("input_ids")
    pixel_values_videos = kwargs.get("pixel_values_videos")

    # 兼容某些版本 hooks kwargs 为空时，从 args 兜底推断。
    if input_ids is None and args:
        first_arg = args[0]
        if getattr(first_arg, "shape", None) is not None:
            input_ids = first_arg

    is_prefill = False
    if pixel_values_videos is not None:
        is_prefill = True
    elif input_ids is not None and getattr(input_ids, "shape", None) is not None:
        is_prefill = input_ids.shape[1] > 1

    model._current_stage = "prefill" if is_prefill else "decode"
    _emit_event(model, f"model_{model._current_stage}_start")


def _forward_hook_model(model, args, kwargs, outputs):
    end_ns = time.perf_counter_ns()
    elapsed_us = (end_ns - model._start_time_ns) / 1000.0

    if model._current_stage == "prefill":
        model._timing["model_prefill_time_us"] = elapsed_us
        model._timing["first_step_time_us"] = elapsed_us
        model._timing["model_prefill_calls"] += 1
        _emit_event(model, "model_prefill_end", {"elapsed_us": elapsed_us})
    else:
        model._timing["model_decode_total_us"] += elapsed_us
        model._timing["model_decode_calls"] += 1
        _emit_event(model, "model_decode_end", {"elapsed_us": elapsed_us})


def _forward_pre_hook_visual(model, inputs):
    model._start_time_ns = time.perf_counter_ns()
    if inputs:
        model._encoder_inputs_shape = tuple(inputs[0].shape)


def _forward_hook_visual(model, inputs, outputs):
    end_ns = time.perf_counter_ns()
    elapsed_us = (end_ns - model._start_time_ns) / 1000.0
    model._time_us = elapsed_us

    if isinstance(outputs, tuple) and outputs:
        try:
            model._encoder_output_shape = tuple(outputs[0].shape)
        except Exception:
            model._encoder_output_shape = None
    else:
        try:
            model._encoder_output_shape = tuple(outputs.shape)
        except Exception:
            model._encoder_output_shape = None


def _forward_pre_hook_llm(model, inputs):
    model._start_time_ns = time.perf_counter_ns()


def _forward_hook_llm(model, inputs, outputs):
    end_ns = time.perf_counter_ns()
    elapsed_us = (end_ns - model._start_time_ns) / 1000.0

    parent = getattr(model, "_timing_parent", None)
    if parent is None:
        return

    if getattr(parent, "_current_stage", "prefill") == "prefill":
        parent._timing["llm_prefill_time_us"] = elapsed_us
    else:
        parent._timing["llm_decode_total_us"] += elapsed_us
        parent._timing["llm_decode_calls"] += 1


def inject_timing_hook_to_model(model, event_callback=None):
    """给 model、vision_tower、language_model 注入计时 hook。

    event_callback: 可选回调，签名为 callback(label: str, payload: dict)
    用于将阶段事件打到外部 monitor。
    """
    _init_timing_state(model)
    model._timing_event_cb = event_callback

    handles = [
        model.register_forward_pre_hook(_forward_pre_hook_model, with_kwargs=True),
        model.register_forward_hook(_forward_hook_model, with_kwargs=True),
        model.vision_tower.register_forward_pre_hook(_forward_pre_hook_visual),
        model.vision_tower.register_forward_hook(_forward_hook_visual),
        model.language_model.register_forward_pre_hook(_forward_pre_hook_llm),
        model.language_model.register_forward_hook(_forward_hook_llm),
    ]
    HOOK_HANDLES.extend(handles)
    return handles


def remove_timing_hooks_from_model():
    """移除所有计时 hooks。"""
    for handle in HOOK_HANDLES:
        handle.remove()
    HOOK_HANDLES.clear()


def reset_timing_stats(model):
    """重置计时统计。"""
    _init_timing_state(model)


def get_timing_stats(model):
    """返回结构化计时统计（单位秒）。"""
    t = model._timing

    avg_decode_model_us = 0.0
    if t["model_decode_calls"] > 0:
        avg_decode_model_us = t["model_decode_total_us"] / t["model_decode_calls"]

    avg_decode_llm_us = 0.0
    if t["llm_decode_calls"] > 0:
        avg_decode_llm_us = t["llm_decode_total_us"] / t["llm_decode_calls"]

    stats = {
        "visual_input_shape": model.vision_tower._encoder_inputs_shape,
        "visual_output_shape": model.vision_tower._encoder_output_shape,
        "visual_time_s": round(getattr(model.vision_tower, "_time_us", 0.0) / 1e6, 6),
        "model_prefill_time_s": round(t["model_prefill_time_us"] / 1e6, 6),
        "llm_prefill_time_s": round(t["llm_prefill_time_us"] / 1e6, 6),
        "first_step_time_s": round(t["first_step_time_us"] / 1e6, 6),
        "model_decode_total_s": round(t["model_decode_total_us"] / 1e6, 6),
        "model_decode_avg_s": round(avg_decode_model_us / 1e6, 6),
        "llm_decode_total_s": round(t["llm_decode_total_us"] / 1e6, 6),
        "llm_decode_avg_s": round(avg_decode_llm_us / 1e6, 6),
        "model_decode_calls": t["model_decode_calls"],
        "llm_decode_calls": t["llm_decode_calls"],
        "first_token_per_second": round(1e6 / t["first_step_time_us"], 2) if t["first_step_time_us"] > 0 else 0.0,
        "decode_token_per_second": round(1e6 / avg_decode_model_us, 2) if avg_decode_model_us > 0 else 0.0,
    }
    return stats


def print_timing_stats(model):
    """打印可读的计时统计。"""
    stats = get_timing_stats(model)
    print("\n===== Model Timing Stats =====")
    print(f"visual input shape: {stats['visual_input_shape']}")
    print(f"visual output shape: {stats['visual_output_shape']}")
    print(f"visual time: {stats['visual_time_s']:.6f} s")
    print(f"model prefill time: {stats['model_prefill_time_s']:.6f} s")
    print(f"llm prefill time: {stats['llm_prefill_time_s']:.6f} s")
    print(f"first step time: {stats['first_step_time_s']:.6f} s")
    print(f"model decode total: {stats['model_decode_total_s']:.6f} s")
    print(f"model decode avg: {stats['model_decode_avg_s']:.6f} s")
    print(f"llm decode total: {stats['llm_decode_total_s']:.6f} s")
    print(f"llm decode avg: {stats['llm_decode_avg_s']:.6f} s")
    print(f"model decode calls: {stats['model_decode_calls']}")
    print(f"llm decode calls: {stats['llm_decode_calls']}")
    print(f"FTPS: {stats['first_token_per_second']:.2f}")
    print(f"TPS(decode): {stats['decode_token_per_second']:.2f}")
