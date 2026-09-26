import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------------------------------------------------------
# Small text helpers
# -----------------------------------------------------------------------------

def _first_line(text: str, max_len: int = 240) -> str:
    """First non-empty line of `text`, trimmed to `max_len` chars."""
    if not text:
        return ""
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s if len(s) <= max_len else s[: max_len - 1] + "\u2026"
    return ""


def _truncate(text: str, max_len: int) -> str:
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "\u2026"


def _fmt_metrics(metrics: Dict[str, float]) -> str:
    """One-line rendering of a metrics dict in a fixed, readable order.

    Keys mirror eval.py's results dict: fluid/solid position RMSE (fluid is
    the primary), KE RMSE and Sinkhorn divergence per phase, plus the fluid
    Eulerian velocity RMSE / fRMSE / R^2. Legacy "mse" is kept for old
    result files.
    """
    if not metrics:
        return "no metrics"
    order = [("fluid_rmse_pos", "fPosRMSE", "{:.4e}"),
             ("mse", "MSE", "{:.6f}"),
             ("solid_rmse_pos", "sPosRMSE", "{:.4e}"),
             ("fluid_ke_rmse", "fKE", "{:.4e}"),
             ("solid_ke_rmse", "sKE", "{:.4e}"),
             ("fluid_sinkhorn", "fSink", "{:.4e}"),
             ("solid_sinkhorn", "sSink", "{:.4e}"),
             ("fluid_eul_vel_rmse", "eulRMSE", "{:.4e}"),
             ("fluid_eul_vel_frmse", "eulfRMSE", "{:.4e}"),
             ("fluid_eul_vel_r2", "eulR2", "{:.6f}")]
    parts = []
    for key, label, fmt in order:
        if key in metrics and metrics[key] is not None:
            parts.append(f"{label}={fmt.format(metrics[key])}")
    return ", ".join(parts) if parts else "no metrics"


# -----------------------------------------------------------------------------
# Programmatic telemetry summary — replaces the old LLM-based analyzer
# -----------------------------------------------------------------------------

def summarize_telemetry(results: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact, factual summary from train.py's results dict.

    The orchestrator gets *the same* numerical view of every prior run, so it
    can compare iterations on equal footing without needing an LLM in the loop
    to produce prose that then has to be re-truncated for memory.

    Output keys (any may be absent if the corresponding series is missing):
      loss_start, loss_final, loss_best, loss_best_epoch, loss_plateau_epoch,
      loss_diverged, grad_norm_max, grad_norm_max_epoch, grad_norm_final,
      lr_start, lr_final, num_epochs, loss_components (name -> final value).
    """
    out: Dict[str, Any] = {}

    losses = list(results.get("loss") or [])
    if losses:
        out["loss_start"] = float(losses[0])
        out["loss_final"] = float(losses[-1])
        best_i = min(range(len(losses)), key=lambda i: losses[i])
        out["loss_best"] = float(losses[best_i])
        out["loss_best_epoch"] = best_i
        out["num_epochs"] = len(losses)

        # Divergence: any NaN/inf, or final loss substantially worse than start.
        import math
        bad = any((not math.isfinite(v)) for v in losses)
        if not bad and len(losses) >= 5 and losses[-1] > 2.0 * losses[0]:
            bad = True
        out["loss_diverged"] = bool(bad)

        # Plateau: the earliest epoch at which the loss is within 1% of the
        # eventual best (i.e. when meaningful improvement effectively stopped).
        # None if the curve was still improving at the end (best epoch lives in
        # the last 10% of training).
        if best_i >= max(1, int(0.9 * len(losses))):
            out["loss_plateau_epoch"] = None
        else:
            ceiling = out["loss_best"] * 1.01
            plateau = next(
                (i for i, v in enumerate(losses) if v <= ceiling),
                None,
            )
            out["loss_plateau_epoch"] = plateau

    grads = list(results.get("grad_norm") or [])
    if grads:
        gi = max(range(len(grads)), key=lambda i: grads[i])
        out["grad_norm_max"] = float(grads[gi])
        out["grad_norm_max_epoch"] = gi
        out["grad_norm_final"] = float(grads[-1])

    lrs = list(results.get("lr") or [])
    if lrs:
        out["lr_start"] = float(lrs[0])
        out["lr_final"] = float(lrs[-1])

    comps = results.get("loss_components") or {}
    # `loss_components` is name -> list-per-epoch; we report only final values.
    if isinstance(comps, dict) and comps:
        finals: Dict[str, float] = {}
        for name, series in comps.items():
            if isinstance(series, (list, tuple)) and series:
                try:
                    finals[str(name)] = float(series[-1])
                except (TypeError, ValueError):
                    pass
        if finals:
            out["loss_components"] = finals

    return out


def _fmt_summary(s: Dict[str, Any]) -> str:
    """One- or two-line rendering of a telemetry summary."""
    if not s:
        return "no telemetry"
    bits: List[str] = []
    if "loss_start" in s and "loss_final" in s:
        bit = f"loss {s['loss_start']:.4g} -> {s['loss_final']:.4g}"
        if "loss_best" in s and "loss_best_epoch" in s:
            bit += f" (best {s['loss_best']:.4g} @ ep {s['loss_best_epoch']})"
        bits.append(bit)
    if s.get("loss_diverged"):
        bits.append("DIVERGED")
    if s.get("loss_plateau_epoch") is not None:
        bits.append(f"plateau @ ep {s['loss_plateau_epoch']}")
    elif "num_epochs" in s and "loss_best_epoch" in s \
            and s["loss_best_epoch"] >= max(1, int(0.9 * s["num_epochs"])):
        bits.append("still improving at end")
    if "grad_norm_max" in s:
        bits.append(
            f"grad max {s['grad_norm_max']:.3g} @ ep {s['grad_norm_max_epoch']}, "
            f"final {s['grad_norm_final']:.3g}"
        )
    if "lr_start" in s and "lr_final" in s:
        bits.append(f"lr {s['lr_start']:.2g} -> {s['lr_final']:.2g}")
    comps = s.get("loss_components") or {}
    if comps:
        bits.append("components: "
                    + ", ".join(f"{k}={v:.4g}" for k, v in comps.items()))
    return "; ".join(bits) if bits else "no telemetry"


# -----------------------------------------------------------------------------
# Memory entry + container
# -----------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A compact, persisted record of one iteration."""
    iteration: int
    strategy: str = ""
    files_to_change: str = ""
    plan: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    primary_metric: Optional[float] = None   # value we minimize (RMSE, else MSE)
    primary_metric_name: str = ""
    num_params: Optional[int] = None
    peak_gpu_memory_gb: Optional[float] = None
    debug_attempts: int = 0
    failed: bool = False
    failure_reason: str = ""
    # Deterministic numerical summary of this iteration's training telemetry,
    # produced by summarize_telemetry(). Replaces the old LLM-written
    # analyzer_summary / analyzer_recommendations fields.
    telemetry_summary: Dict[str, Any] = field(default_factory=dict)


class Memory:
    """Accumulates :class:`MemoryEntry` objects and renders prompt context."""

    def __init__(self) -> None:
        self.entries: List[MemoryEntry] = []

    # ------------------------------------------------------------------ record
    def record(self, *, iteration: int, strategy: str = "",
               files_to_change: str = "", plan: str = "",
               metrics: Optional[Dict[str, float]] = None,
               primary_metric: Optional[float] = None,
               primary_metric_name: str = "",
               num_params: Optional[int] = None,
               peak_gpu_memory_gb: Optional[float] = None,
               debug_attempts: int = 0, failed: bool = False,
               failure_reason: str = "",
               telemetry_summary: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        entry = MemoryEntry(
            iteration=iteration,
            strategy=_truncate(strategy, 600),
            files_to_change=_truncate(files_to_change, 200),
            plan=_truncate(plan, 800),
            metrics=dict(metrics or {}),
            primary_metric=primary_metric,
            primary_metric_name=primary_metric_name,
            num_params=num_params,
            peak_gpu_memory_gb=peak_gpu_memory_gb,
            debug_attempts=debug_attempts,
            failed=failed,
            failure_reason=_truncate(failure_reason, 300),
            telemetry_summary=dict(telemetry_summary or {}),
        )
        self.entries.append(entry)
        return entry

    # -------------------------------------------------------------- best so far
    def best_entry(self) -> Optional[MemoryEntry]:
        """The non-failed entry with the smallest primary metric (lower=better)."""
        scored = [e for e in self.entries
                  if not e.failed and e.primary_metric is not None]
        if not scored:
            return None
        return min(scored, key=lambda e: e.primary_metric)

    # ------------------------------------------------------ orchestrator context
    def as_orchestrator_context(self) -> str:
        """Render memory as a compact context block for the orchestrator prompt.

        Every prior iteration is rendered with the same level of detail — there
        is no "recent vs old" decay, because the per-iteration block is small
        enough (~7 short lines) that many iterations still fit comfortably.
        """
        if not self.entries:
            return "(no prior iterations yet)"

        lines: List[str] = []

        # (a) Trajectory table -------------------------------------------------
        lines.append("### Trajectory (one row per prior iteration)")
        lines.append("iter | status | primary | other metrics | params | peak mem")
        for e in self.entries:
            status = "FAILED" if e.failed else "ok"
            if e.primary_metric is not None:
                pm = f"{e.primary_metric_name}={e.primary_metric:.6e}"
            else:
                pm = "N/A"
            others = {k: v for k, v in e.metrics.items()
                      if k.lower() != (e.primary_metric_name or "").lower()}
            other_str = _fmt_metrics(others) if others else "-"
            params = f"{e.num_params/1e6:.2f}M" if e.num_params else "N/A"
            mem = f"{e.peak_gpu_memory_gb:.2f}GB" if e.peak_gpu_memory_gb else "N/A"
            lines.append(f"{e.iteration} | {status} | {pm} | {other_str} | "
                         f"{params} | {mem}")
        lines.append("")

        # (b) Best so far + regression note -----------------------------------
        best = self.best_entry()
        last = self.entries[-1]
        if best is not None:
            lines.append(
                f"### Best so far: iteration {best.iteration} "
                f"({best.primary_metric_name}={best.primary_metric:.6e}; "
                f"{_fmt_metrics(best.metrics)})"
            )
            if (not last.failed and last.primary_metric is not None
                    and best.iteration != last.iteration
                    and last.primary_metric > best.primary_metric):
                delta = last.primary_metric - best.primary_metric
                lines.append(
                    f"- REGRESSION: the latest iteration {last.iteration} "
                    f"({last.primary_metric_name}={last.primary_metric:.6e}) is "
                    f"WORSE than the best by {delta:.6e}. Consider reverting to or "
                    f"building on iteration {best.iteration} rather than continuing "
                    f"the current direction."
                )
            elif last.failed:
                lines.append(
                    f"- The latest iteration {last.iteration} FAILED "
                    f"({last.failure_reason}); the best known-good configuration is "
                    f"iteration {best.iteration}."
                )
        else:
            lines.append("### Best so far: none yet (no iteration produced a "
                         "validation metric).")
        lines.append("")

        # (c) Per-iteration detail blocks — SAME for every iteration ----------
        lines.append("### Per-iteration detail")
        for e in self.entries:
            status = "FAILED" if e.failed else "ok"
            header_metric = (f"{e.primary_metric_name}={e.primary_metric:.6e}"
                             if e.primary_metric is not None else "N/A")
            lines.append(f"#### iter {e.iteration} ({status}, {header_metric})")
            if e.strategy:
                lines.append(f"- Strategy: {_truncate(e.strategy, 280)}")
            if e.files_to_change:
                lines.append(f"- Files changed: {e.files_to_change}")
            if e.telemetry_summary:
                lines.append(f"- Training: {_fmt_summary(e.telemetry_summary)}")
            if e.metrics:
                lines.append(f"- Validation: {_fmt_metrics(e.metrics)}")
            if e.num_params or e.peak_gpu_memory_gb:
                bits = []
                if e.num_params:
                    bits.append(f"{e.num_params/1e6:.2f}M params")
                if e.peak_gpu_memory_gb:
                    bits.append(f"peak {e.peak_gpu_memory_gb:.2f} GB")
                lines.append(f"- Resources: {', '.join(bits)}")
            if e.failed and e.failure_reason:
                lines.append(f"- Failure: {e.failure_reason}")

        return "\n".join(lines).strip()

    # ----------------------------------------------------------------- persist
    def to_dict(self) -> dict:
        best = self.best_entry()
        return {
            "num_iterations": len(self.entries),
            "best_iteration": best.iteration if best else None,
            "best_primary_metric": best.primary_metric if best else None,
            "entries": [asdict(e) for e in self.entries],
        }

    def save(self, run_dir: Path) -> None:
        """Write ``memory.json`` (structured) and ``memory.md`` (human-readable)."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "memory.json").write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        )
        (run_dir / "memory.md").write_text(self._render_markdown())

    def _render_markdown(self) -> str:
        lines = ["# Orchestrator memory", "",
                 "Chronological record of every iteration: the decision taken, "
                 "the deterministic telemetry summary, and the validation "
                 "metrics. Lower RMSE/MSE is better.", ""]
        best = self.best_entry()
        if best is not None:
            lines.append(f"**Best iteration:** {best.iteration} "
                         f"({best.primary_metric_name}={best.primary_metric:.6e})")
            lines.append("")
        for e in self.entries:
            status = "FAILED" if e.failed else "ok"
            lines.append(f"## Iteration {e.iteration} ({status})")
            lines.append(f"- Validation metrics: {_fmt_metrics(e.metrics)}")
            if e.num_params:
                lines.append(f"- Parameters: {e.num_params/1e6:.2f}M")
            if e.peak_gpu_memory_gb:
                lines.append(f"- Peak GPU memory: {e.peak_gpu_memory_gb:.3f} GB")
            lines.append(f"- Debug attempts: {e.debug_attempts}")
            if e.telemetry_summary:
                lines.append(f"- Training: {_fmt_summary(e.telemetry_summary)}")
            if e.failed and e.failure_reason:
                lines.append(f"- Failure reason: {e.failure_reason}")
            if e.files_to_change:
                lines.append(f"- Files changed: {e.files_to_change}")
            if e.strategy:
                lines.append(f"\n**Strategy:**\n{e.strategy}")
            if e.plan:
                lines.append(f"\n**Plan:**\n{e.plan}")
            lines.append("")
        return "\n".join(lines)