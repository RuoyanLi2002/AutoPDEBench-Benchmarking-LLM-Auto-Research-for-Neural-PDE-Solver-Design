from typing import Dict, List, Optional


def _full_curve(name: str, values: List[float], fmt: str = ".6f",
                per_line: int = 10) -> str:
    """Render every epoch value, wrapped at `per_line` entries per line."""
    if not values:
        return f"{name}: (no data)"

    n = len(values)
    lo_i = min(range(n), key=lambda i: values[i])
    hi_i = max(range(n), key=lambda i: values[i])

    # Spike/jump detection: epochs where the value rises notably vs the previous.
    spikes = []
    for i in range(1, n):
        prev = values[i - 1]
        cur = values[i]
        if prev > 0 and cur > prev * 1.5:  # >50% jump upward
            spikes.append(i)

    lines = [
        f"{name}:",
        f"  first={values[0]:{fmt}}  last={values[-1]:{fmt}}  "
        f"min={values[lo_i]:{fmt}}@{lo_i}  max={values[hi_i]:{fmt}}@{hi_i}",
    ]
    if spikes:
        spike_str = ", ".join(
            f"e{i}({values[i-1]:{fmt}}->{values[i]:{fmt}})" for i in spikes[:20]
        )
        more = "" if len(spikes) <= 20 else f" ... (+{len(spikes)-20} more)"
        lines.append(f"  upward jumps >50%: {spike_str}{more}")

    lines.append("  full per-epoch values:")
    for start in range(0, n, per_line):
        chunk = values[start:start + per_line]
        rendered = "  ".join(f"e{start+j}:{v:{fmt}}" for j, v in enumerate(chunk))
        lines.append("    " + rendered)

    return "\n".join(lines)


def summarize_telemetry(results: Dict, gpu_memory_gb: Optional[float] = None) -> str:
    """Build a full-curve textual telemetry summary from a results dict."""
    parts = []

    loss = results.get("loss", [])
    parts.append(_full_curve("Training loss", loss))

    grad = results.get("grad_norm", [])
    if grad:
        parts.append(_full_curve("Gradient norm", grad, fmt=".4f"))

    lr = results.get("lr", [])
    if lr:
        parts.append(_full_curve("Learning rate", lr, fmt=".3e"))

    components = results.get("loss_components", {}) or {}
    for cname, cvals in components.items():
        if cvals:
            parts.append(_full_curve(f"Loss component '{cname}'", cvals))

    # Resource usage
    num_params = results.get("num_params")
    peak_mem = results.get("peak_gpu_memory_gb")
    res_lines = ["Resources:"]
    if num_params is not None:
        res_lines.append(f"  parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    if peak_mem is not None:
        budget = f" of {gpu_memory_gb:.1f} GB budget" if gpu_memory_gb else ""
        res_lines.append(f"  peak GPU memory: {peak_mem:.3f} GB{budget}")
    runtime = results.get("runtime", [])
    if runtime:
        total_min = sum(runtime) / 60.0
        res_lines.append(f"  total train time: {total_min:.1f} min "
                         f"({len(runtime)} epochs, {runtime[-1]:.1f}s/epoch last)")
    parts.append("\n".join(res_lines))

    return "\n\n".join(parts)