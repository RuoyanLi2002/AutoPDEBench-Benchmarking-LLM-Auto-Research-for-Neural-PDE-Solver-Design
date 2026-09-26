import json
import os
import pickle
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .llm_client import LLMClient
from .agents import Orchestrator, Coder, Debugger
from .parser import CodeFiles
from .runner import run_training, run_eval
from .telemetry import summarize_telemetry as summarize_telemetry_text
from .memory import Memory, _fmt_metrics, summarize_telemetry


# -----------------------------------------------------------------------------
# Records
# -----------------------------------------------------------------------------

@dataclass
class IterationRecord:
    iteration: int
    plan: str = ""
    strategy: str = ""
    coder_notes: str = ""
    debug_attempts: int = 0
    debug_diagnoses: List[str] = field(default_factory=list)
    
    
    valid_mse: Optional[float] = None
    primary_metric_name: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    num_params: Optional[int] = None
    peak_gpu_memory_gb: Optional[float] = None


    telemetry_summary: Dict[str, object] = field(default_factory=dict)
    failed: bool = False
    failure_reason: str = ""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


_METRIC_PATTERNS = {
    "fluid_rmse_pos":      re.compile(r"Fluid RMSE \(pos\)\s*:\s*([0-9.eE+-]+)"),
    "fluid_ke_rmse":       re.compile(r"Fluid RMSE \(KE\)\s*:\s*([0-9.eE+-]+)"),
    "fluid_sinkhorn":      re.compile(r"Fluid Sinkhorn div\s*:\s*([0-9.eE+-]+)"),
    "solid_rmse_pos":      re.compile(r"Solid RMSE \(pos\)\s*:\s*([0-9.eE+-]+)"),
    "solid_ke_rmse":       re.compile(r"Solid RMSE \(KE\)\s*:\s*([0-9.eE+-]+)"),
    "solid_sinkhorn":      re.compile(r"Solid Sinkhorn div\s*:\s*([0-9.eE+-]+)"),
    "fluid_eul_vel_rmse":  re.compile(r"Fluid Eul\. vel RMSE\s*:\s*([0-9.eE+-]+)"),
    "fluid_eul_vel_frmse": re.compile(r"Fluid Eul\. vel fRMSE\s*:\s*([0-9.eE+-]+)"),
    "fluid_eul_vel_r2":    re.compile(r"Fluid Eul\. vel R\^2\s*:\s*([0-9.eE+-]+)"),
}



_PRIMARY_PREFERENCE = ("fluid_rmse_pos", "mse")


def read_valid_metrics(eval_result_path: Path) -> Dict[str, float]:
    if not eval_result_path.exists():
        return {}
    text = eval_result_path.read_text()
    out: Dict[str, float] = {}
    for name, pat in _METRIC_PATTERNS.items():
        m = pat.search(text)
        if m:
            try:
                out[name] = float(m.group(1))
            except ValueError:
                pass
    return out


def primary_metric(metrics: Dict[str, float]) -> (Optional[float], str):
    """Pick the scalar to minimize (lower=better) and its name.

    Prefers the fluid position RMSE; falls back to legacy MSE. Returns
    (value, name) or (None, "") if neither is present. The returned name is
    the metrics-dict key upper-cased (e.g. "FLUID_RMSE_POS"), so memory.py's
    case-insensitive dedup of the primary metric keeps working.
    """
    for name in _PRIMARY_PREFERENCE:
        if name in metrics and metrics[name] is not None:
            return metrics[name], name.upper()
    return None, ""


def patch_config(config_yaml: str, exp_name_abs: str,
                 base_config_path: str = "configs/base_config.yaml") -> str:
    cfg = yaml.safe_load(config_yaml) or {}
    cfg.pop("base_config", None)

    training = cfg.setdefault("training", {})
    training["exp_name"] = exp_name_abs

    
    training["model_path"] = os.path.join(exp_name_abs, "model.pth")

    
    training["num_epochs"] = 100

    
    sched = training.get("scheduler")
    if isinstance(sched, dict):
        params = sched.get("params")
        if isinstance(params, dict) and "T_max" in params:
            params["T_max"] = 100

    ordered = {"base_config": base_config_path}
    ordered.update(cfg)
    return yaml.safe_dump(ordered, sort_keys=False, default_flow_style=False)


def install_files(files: CodeFiles, project_dir: Path):
    """Write the editable files into the project so the next run picks them up."""
    if files.model_code is not None:
        (project_dir / "model.py").write_text(files.model_code)
    if files.losses_code is not None:
        (project_dir / "losses.py").write_text(files.losses_code)

        


def format_agent_io(role: str, system: str, user: str, output: str) -> str:
    """Format an agent call's full input and output as one combined block."""
    sep = "=" * 70
    blocks = [
        f"{sep}\n  {role}\n{sep}",
        "----- INPUT: SYSTEM PROMPT -----\n" + system,
        "----- INPUT: USER PROMPT -----\n" + user,
        "----- OUTPUT: RESPONSE -----\n" + output,
    ]
    return "\n\n".join(blocks)


def log_agent_io(path: Path, role: str, system: str, user: str, output: str):
    """Write ONE combined file containing an agent call's full input and output."""
    path.write_text(format_agent_io(role, system, user, output))


class IOCollector:
    """Collects every raw LLM input/output in an iteration into ONE file.

    Task requirement: for each iteration, persist exactly what was passed into
    the LLM (system + user prompts) and exactly what came back (raw response),
    for every agent call, in a single file. We emit:
      - ``llm_io.json`` : machine-readable, exact raw strings (canonical).
      - ``llm_io.md``   : the same content rendered for easy reading.
    """

    def __init__(self, iteration: int):
        self.iteration = iteration
        self.calls: List[dict] = []

    def add(self, role: str, system: str, user: str, output: str):
        self.calls.append({
            "call_index": len(self.calls),
            "role": role,
            "input": {"system_prompt": system, "user_prompt": user},
            "output": output,
        })

    def _render_markdown(self) -> str:
        sep = "=" * 70
        blocks = [f"# LLM I/O — iteration {self.iteration}",
                  "",
                  f"{len(self.calls)} LLM call(s) this iteration. Each block shows "
                  "the exact system prompt and user prompt passed in, and the exact "
                  "raw model output.", ""]
        for c in self.calls:
            blocks.append(f"{sep}\n## [{c['call_index']}] {c['role']}\n{sep}")
            blocks.append("### INPUT — system prompt\n```\n"
                          + c["input"]["system_prompt"] + "\n```")
            blocks.append("### INPUT — user prompt\n```\n"
                          + c["input"]["user_prompt"] + "\n```")
            blocks.append("### OUTPUT — raw response\n```\n" + c["output"] + "\n```")
            blocks.append("")
        return "\n\n".join(blocks)

    def write(self, iter_dir: Path):
        payload = {"iteration": self.iteration, "num_calls": len(self.calls),
                   "calls": self.calls}
        (iter_dir / "llm_io.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
        (iter_dir / "llm_io.md").write_text(self._render_markdown())


# -----------------------------------------------------------------------------
# Single iteration
# -----------------------------------------------------------------------------

def run_iteration(iteration: int, project_dir: Path, run_dir: Path,
                  orchestrator: Orchestrator, coder: Coder,
                  debugger: Debugger,
                  current: CodeFiles, gpu_memory_gb: float,
                  gpu_id: Optional[int],
                  max_debug_attempts: int,
                  memory: Memory) -> (IterationRecord, CodeFiles):
    iter_dir = run_dir / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = (iter_dir / "exp").resolve()

    rec = IterationRecord(iteration=iteration)
    io = IOCollector(iteration)

    # Captured during the iteration and consumed by `finalize` when this
    # iteration is recorded into the persistent orchestrator memory.
    extra = {"files_to_change": ""}

    def finalize(record: IterationRecord, new_current: CodeFiles):
        """Persist the single-file LLM I/O log and record this iteration into
        the orchestrator memory, then return. Called on every exit path so
        failures are remembered too."""
        io.write(iter_dir)
        memory.record(
            iteration=record.iteration,
            strategy=record.strategy,
            files_to_change=extra["files_to_change"],
            plan=record.plan,
            metrics=record.metrics,
            primary_metric=record.valid_mse,
            primary_metric_name=record.primary_metric_name,
            num_params=record.num_params,
            peak_gpu_memory_gb=record.peak_gpu_memory_gb,
            debug_attempts=record.debug_attempts,
            failed=record.failed,
            failure_reason=record.failure_reason,
            telemetry_summary=record.telemetry_summary,
        )
        memory.save(run_dir)
        return record, new_current

    # --- 1. Orchestrator plans ---------------------------------------------
    print(f"\n{'='*70}\n  Iteration {iteration}: ORCHESTRATOR planning\n{'='*70}")
    memory_context = memory.as_orchestrator_context()
    orch_sys, orch_usr, orch_raw, orch = orchestrator.plan(
        iteration=iteration, gpu_memory_gb=gpu_memory_gb,
        memory_context=memory_context,
    )
    log_agent_io(iter_dir / "agent_orchestrator.log", "ORCHESTRATOR",
                 orch_sys, orch_usr, orch_raw)
    io.add("ORCHESTRATOR", orch_sys, orch_usr, orch_raw)
    rec.plan = orch.plan
    rec.strategy = orch.strategy
    extra["files_to_change"] = orch.files_to_change

    # --- 2. Coder writes ----------------------------------------------------
    print(f"  Iteration {iteration}: CODER writing")
    coder_sys, coder_usr, coder_raw, coder_out = coder.write(
        iteration=iteration, gpu_memory_gb=gpu_memory_gb, plan=orch.plan,
        current_model=current.model_code or "",
        current_losses=current.losses_code or "",
        current_config=current.config_yaml or "",
    )
    log_agent_io(iter_dir / "agent_coder.log", "CODER",
                 coder_sys, coder_usr, coder_raw)
    io.add("CODER", coder_sys, coder_usr, coder_raw)
    rec.coder_notes = coder_out.notes

    if coder_out.files.is_empty():
        rec.failed = True
        rec.failure_reason = "Coder produced no parseable code files."
        return finalize(rec, current)

    # Merge new files onto current state (carry forward unchanged files)
    candidate = coder_out.files.merged_onto(current)
    if candidate.config_yaml is None:
        rec.failed = True
        rec.failure_reason = "No config.yaml available (neither new nor carried)."
        return finalize(rec, current)

    # --- 3. Install + train, with debugger retry loop ----------------------
    patched_yaml = patch_config(candidate.config_yaml, exp_name_abs=str(exp_dir))
    candidate = CodeFiles(candidate.model_code, candidate.losses_code, patched_yaml)

    project_config_path = project_dir / "configs" / f"iter_{iteration}_config.yaml"
    config_rel = os.path.relpath(project_config_path, project_dir)

    train_rc = None
    train_out = ""
    for attempt in range(max_debug_attempts + 1):
        install_files(candidate, project_dir)
        project_config_path.write_text(candidate.config_yaml)

        # snapshot what we are about to run
        tag = "" if attempt == 0 else f"_debug{attempt}"
        (iter_dir / f"model{tag}.py").write_text(candidate.model_code or "")
        (iter_dir / f"losses{tag}.py").write_text(candidate.losses_code or "")
        (iter_dir / f"config{tag}.yaml").write_text(candidate.config_yaml)

        print(f"  Iteration {iteration}: TRAINING (attempt {attempt})")
        train_rc, train_out = run_training(
            project_dir=str(project_dir), config_path=config_rel,
            log_path=str(iter_dir / f"train_stdout{tag}.txt"), gpu_id=gpu_id,
        )
        if train_rc == 0:
            break

        if attempt >= max_debug_attempts:
            rec.failed = True
            rec.failure_reason = (
                f"Training failed after {max_debug_attempts} debug attempt(s). "
                f"Exit {train_rc}."
            )
            rec.debug_attempts = attempt
            return finalize(rec, current)

        # --- Debugger fixes ---
        print(f"  Iteration {iteration}: DEBUGGER (attempt {attempt+1})")
        dbg_sys, dbg_usr, dbg_raw, dbg = debugger.fix(
            error_log=train_out[-4000:],
            model_code=candidate.model_code or "",
            losses_code=candidate.losses_code or "",
            config_yaml=candidate.config_yaml,
        )
        # All debugger attempts for this iteration go into ONE combined file.
        dbg_log = iter_dir / "agent_debugger.log"
        block = format_agent_io(f"DEBUGGER (attempt {attempt+1})",
                                dbg_sys, dbg_usr, dbg_raw)
        if dbg_log.exists():
            dbg_log.write_text(dbg_log.read_text() + "\n\n\n" + block)
        else:
            dbg_log.write_text(block)
        io.add(f"DEBUGGER (attempt {attempt+1})", dbg_sys, dbg_usr, dbg_raw)

        rec.debug_diagnoses.append(dbg.diagnosis)
        rec.debug_attempts = attempt + 1

        if dbg.files.is_empty():
            rec.failed = True
            rec.failure_reason = "Debugger produced no fixed files."
            return finalize(rec, current)

        fixed = dbg.files.merged_onto(candidate)
        # Re-patch config in case the debugger touched it
        fixed_yaml = patch_config(fixed.config_yaml, exp_name_abs=str(exp_dir))
        candidate = CodeFiles(fixed.model_code, fixed.losses_code, fixed_yaml)

    # --- 4. Eval (valid) ----------------------------------------------------
    print(f"  Iteration {iteration}: EVAL (valid)")
    eval_rc, eval_out = run_eval(
        project_dir=str(project_dir), config_path=config_rel, split="valid",
        log_path=str(iter_dir / "eval_stdout.txt"), gpu_id=gpu_id,
    )

    results_pkl = exp_dir / "results.pkl"
    telemetry_text = "results.pkl not found"
    results = {}
    if results_pkl.exists():
        with open(results_pkl, "rb") as f:
            results = pickle.load(f)
        telemetry_text = summarize_telemetry_text(results, gpu_memory_gb=gpu_memory_gb)
    # Keep the human-readable epoch table on disk for inspection, even though
    # no LLM agent reads it anymore.
    (iter_dir / "telemetry.txt").write_text(telemetry_text)

    rec.num_params = results.get("num_params")
    rec.peak_gpu_memory_gb = results.get("peak_gpu_memory_gb")
    rec.metrics = read_valid_metrics(exp_dir / "eval_valid_results.txt")
    rec.valid_mse, rec.primary_metric_name = primary_metric(rec.metrics)
    # Deterministic structured summary that will be stored in memory and shown
    # to the orchestrator next round (replaces the LLM-based analyzer).
    rec.telemetry_summary = summarize_telemetry(results) if results else {}

    primary_str = (f"{rec.primary_metric_name}={rec.valid_mse:.7e}"
                   if rec.valid_mse is not None else "N/A")
    valid_metrics_str = _fmt_metrics(rec.metrics) if rec.metrics else "N/A"

    np_str = f"{rec.num_params/1e6:.2f}M" if rec.num_params else "N/A"
    mem_str = f"{rec.peak_gpu_memory_gb:.3f} GB" if rec.peak_gpu_memory_gb else "N/A"
    print(f"\n  Iteration {iteration} done. "
          f"Valid {primary_str} | metrics: {valid_metrics_str} | "
          f"Params: {np_str} | Peak mem: {mem_str} | "
          f"debug attempts: {rec.debug_attempts}")

    return finalize(rec, candidate)


# -----------------------------------------------------------------------------
# Top-level loop
# -----------------------------------------------------------------------------

def _read_initial_files(project_dir: Path) -> CodeFiles:
    """Seed 'current' state from the project's existing files."""
    model = (project_dir / "model.py").read_text() if (project_dir / "model.py").exists() else ""
    losses = (project_dir / "losses.py").read_text() if (project_dir / "losses.py").exists() else ""
    cfg_path = project_dir / "configs" / "config.yaml"
    config = cfg_path.read_text() if cfg_path.exists() else ""
    return CodeFiles(model_code=model or None,
                     losses_code=losses or None,
                     config_yaml=config or None)


def run_agent(num_iterations: int, run_dir: str, project_dir: str,
              model_name: str = "gpt-4o-mini",
              temperature: float = 0.7,
              gpu_memory_gb: float = 24.0,
              gpu_id: Optional[int] = None,
              max_debug_attempts: int = 2) -> List[IterationRecord]:
    project_dir_p = Path(project_dir).resolve()
    run_dir_p = Path(run_dir).resolve()
    run_dir_p.mkdir(parents=True, exist_ok=True)

    if not (project_dir_p / "main.py").exists():
        raise FileNotFoundError(
            f"main.py not found in project_dir={project_dir_p}. "
            "Pass --project_dir pointing at the project root."
        )

    llm = LLMClient(model_name=model_name, temperature=temperature)
    orchestrator = Orchestrator(llm)
    coder = Coder(llm)
    debugger = Debugger(llm)

    current = _read_initial_files(project_dir_p)
    history: List[IterationRecord] = []
    memory = Memory()

    for i in range(num_iterations):
        rec, current = run_iteration(
            iteration=i, project_dir=project_dir_p, run_dir=run_dir_p,
            orchestrator=orchestrator, coder=coder, debugger=debugger,
            current=current, gpu_memory_gb=gpu_memory_gb, gpu_id=gpu_id,
            max_debug_attempts=max_debug_attempts, memory=memory,
        )
        history.append(rec)
        write_summary(run_dir_p, history)

    return history


def write_summary(run_dir: Path, history: List[IterationRecord]) -> None:
    lines = ["# Agent run summary", ""]

    # Leaderboard
    ranked = sorted(
        [h for h in history if h.valid_mse is not None],
        key=lambda h: h.valid_mse,
    )
    if ranked:
        lines.append("## Best iterations by validation metric (lower is better)")
        for h in ranked:
            name = h.primary_metric_name or "metric"
            lines.append(f"- iter {h.iteration}: {name}={h.valid_mse:.7e}")
        lines.append("")

    for h in history:
        name = h.primary_metric_name or "metric"
        vl = f"{name}={h.valid_mse:.7e}" if h.valid_mse is not None else "N/A"
        np_str = f"{h.num_params/1e6:.2f}M" if h.num_params else "N/A"
        mem_str = f"{h.peak_gpu_memory_gb:.3f} GB" if h.peak_gpu_memory_gb else "N/A"
        status = "FAILED" if h.failed else "ok"

        lines.append(f"## Iteration {h.iteration} ({status})")
        lines.append(f"- Validation (primary): {vl}")
        lines.append(f"- All validation metrics: {_fmt_metrics(h.metrics)}")
        lines.append(f"- Parameters: {np_str}")
        lines.append(f"- Peak GPU memory: {mem_str}")
        lines.append(f"- Debug attempts: {h.debug_attempts}")
        if h.failed:
            lines.append(f"- Failure reason: {h.failure_reason}")
        if h.strategy:
            lines.append(f"\n**Strategy:**\n{h.strategy}")
        if h.plan:
            lines.append(f"\n**Plan:**\n{h.plan}")
        if h.coder_notes:
            lines.append(f"\n**Coder notes:**\n{h.coder_notes}")
        if h.debug_diagnoses:
            lines.append("\n**Debugger diagnoses:**")
            for j, d in enumerate(h.debug_diagnoses, 1):
                lines.append(f"  {j}. {d}")
        if h.telemetry_summary:
            # Import here to avoid a circular import at module load time.
            from .memory import _fmt_summary
            lines.append(f"\n**Training summary:** {_fmt_summary(h.telemetry_summary)}")
        lines.append("")

    (run_dir / "summary.md").write_text("\n".join(lines))