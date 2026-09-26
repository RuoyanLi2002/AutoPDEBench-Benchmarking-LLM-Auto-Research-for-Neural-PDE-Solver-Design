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
    
    
    per_field_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    num_params: Optional[int] = None
    peak_gpu_memory_gb: Optional[float] = None

    
    telemetry_summary: Dict[str, object] = field(default_factory=dict)
    failed: bool = False
    failure_reason: str = ""




_METRIC_PATTERNS = {
    "rmse":  re.compile(r"\bRMSE\b\s*:?\s*([0-9.eE+-]+)"),
    "frmse": re.compile(r"\bfRMSE\b\s*:?\s*([0-9.eE+-]+)"),
}



_PRIMARY_PREFERENCE = ("rmse", "mse")


def read_valid_metrics(eval_result_path: Path) -> Dict[str, float]:
    """Parse every recognized metric from the eval results file.

    Returns a dict like {"rmse": .., "frmse": ..} (or {"mse": ..} for the legacy
    format). Empty dict if the file is missing or no metric line is found.
    """
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

    Prefers validation RMSE; falls back to legacy MSE. Returns (value, name)
    or (None, "") if neither is present.
    """
    for name in _PRIMARY_PREFERENCE:
        if name in metrics and metrics[name] is not None:
            return metrics[name], name.upper()
    return None, ""


def average_field_metrics(per_field: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Average each metric across the per-field validation results.

    `per_field` maps field name -> that field's metrics dict (rmse, frmse).
    Returns a single metrics dict whose values are the mean over the fields that
    reported each metric. The three fields are separately-normalized regression
    problems on the same [-1, 1] scale, so a plain mean of validation RMSE is a
    sensible single objective for ranking one shared design across all three.
    Empty input -> empty dict.
    """
    if not per_field:
        return {}
    agg: Dict[str, float] = {}
    keys = set()
    for m in per_field.values():
        keys.update(m.keys())
    for key in keys:
        vals = [m[key] for m in per_field.values()
                if key in m and m[key] is not None]
        if vals:
            agg[key] = sum(vals) / len(vals)
    return agg




FIELD_SPECS = {
    "neutron": {"input_dim": 2, "output_dim": 1},
    "solid":   {"input_dim": 2, "output_dim": 1},
    "fluid":   {"input_dim": 1, "output_dim": 4},
}


FIELD_COUPLE_EXP = {
    "neutron": "NTCouple/ntc_neutron",
    "solid":   "NTCouple/ntc_solid",
    "fluid":   "NTCouple/ntc_fluid",
}


def patch_config(config_yaml: str, exp_name_abs: str,
                 base_config_path: str = "configs/base_config.yaml",
                 field: Optional[str] = None) -> str:
    """Harness-controlled config overrides.

    The agent may propose anything, but these fields are owned by the harness
    and always overwritten so a bad config can't break the run:
      - base_config : pinned to the canonical data config
      - exp_name    : pinned to this iteration's (per-field) experiment dir
      - model_path  : pinned to <exp_name>/model.pth (must match where train.py
                      saves and where eval.py loads, so eval never fails on a
                      renamed/missing checkpoint)
      - num_epochs  : pinned to the fixed 100-epoch budget
      - T_max       : only if the chosen scheduler already defines a `T_max`
                      param, it is aligned to the 100-epoch budget. The harness
                      does not require or assume any particular scheduler.

    When `field` is given (the three-model nuclear setup), the harness also pins
    the per-field data.field and the model.input_dim / model.output_dim channel
    counts to that field's fixed spec, so the agent's single shared config is
    reused to train each of the three surrogates. Any input_dim/output_dim/field
    the agent proposed is overwritten.
    """
    cfg = yaml.safe_load(config_yaml) or {}
    cfg.pop("base_config", None)

    if field is not None:
        spec = FIELD_SPECS[field]
        data = cfg.setdefault("data", {})
        data["field"] = field
        model = cfg.setdefault("model", {})
        model["input_dim"] = spec["input_dim"]
        model["output_dim"] = spec["output_dim"]

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

    # --- 3. Install + train ALL THREE fields, with debugger retry loop -----
    # The coupled fixed-point test needs one trained surrogate per field, so
    # each iteration trains all of {neutron, solid, fluid} with the SAME agent
    # code (one shared architecture + loss). Per field, the harness reuses the
    # agent's config and overrides only data.field + the channel counts, then
    # trains to a per-field experiment dir under this iteration.
    fields = list(FIELD_SPECS.keys())
    # Per-field experiment dirs for THIS iteration (where train.py saves and
    # eval.py's valid split loads from).
    field_exp = {f: (iter_dir / f"exp_{f}").resolve() for f in fields}

    project_config_paths = {
        f: project_dir / "configs" / f"iter_{iteration}_{f}_config.yaml"
        for f in fields
    }
    config_rels = {f: os.path.relpath(project_config_paths[f], project_dir)
                   for f in fields}

    def build_field_configs(cand: CodeFiles) -> Dict[str, str]:
        """Patch the agent's config once per field (field + channels + harness pins)."""
        return {
            f: patch_config(cand.config_yaml, exp_name_abs=str(field_exp[f]),
                            field=f)
            for f in fields
        }

    for attempt in range(max_debug_attempts + 1):
        install_files(candidate, project_dir)
        field_yaml = build_field_configs(candidate)
        for f in fields:
            project_config_paths[f].write_text(field_yaml[f])

        # snapshot what we are about to run
        tag = "" if attempt == 0 else f"_debug{attempt}"
        (iter_dir / f"model{tag}.py").write_text(candidate.model_code or "")
        (iter_dir / f"losses{tag}.py").write_text(candidate.losses_code or "")
        for f in fields:
            (iter_dir / f"config_{f}{tag}.yaml").write_text(field_yaml[f])

        # Train each field in turn. The first field that fails to train stops the
        # attempt and its error is handed to the debugger (a code/shape/OOM issue
        # in the shared model or loss will surface on whichever field runs first).
        failed_field = None
        train_rc = 0
        train_out = ""
        for f in fields:
            print(f"  Iteration {iteration}: TRAINING field '{f}' (attempt {attempt})")
            rc, out = run_training(
                project_dir=str(project_dir), config_path=config_rels[f],
                log_path=str(iter_dir / f"train_stdout_{f}{tag}.txt"), gpu_id=gpu_id,
            )
            if rc != 0:
                failed_field, train_rc, train_out = f, rc, out
                break

        if failed_field is None:
            break  # all three fields trained

        if attempt >= max_debug_attempts:
            rec.failed = True
            rec.failure_reason = (
                f"Training of field '{failed_field}' failed after "
                f"{max_debug_attempts} debug attempt(s). Exit {train_rc}."
            )
            rec.debug_attempts = attempt
            return finalize(rec, current)

        # --- Debugger fixes (on the failing field's error) ---
        print(f"  Iteration {iteration}: DEBUGGER (attempt {attempt+1}, "
              f"field '{failed_field}')")
        dbg_sys, dbg_usr, dbg_raw, dbg = debugger.fix(
            error_log=train_out[-4000:],
            model_code=candidate.model_code or "",
            losses_code=candidate.losses_code or "",
            config_yaml=field_yaml[failed_field],
        )
        # All debugger attempts for this iteration go into ONE combined file.
        dbg_log = iter_dir / "agent_debugger.log"
        block = format_agent_io(f"DEBUGGER (attempt {attempt+1}, field {failed_field})",
                                dbg_sys, dbg_usr, dbg_raw)
        if dbg_log.exists():
            dbg_log.write_text(dbg_log.read_text() + "\n\n\n" + block)
        else:
            dbg_log.write_text(block)
        io.add(f"DEBUGGER (attempt {attempt+1}, field {failed_field})",
               dbg_sys, dbg_usr, dbg_raw)

        rec.debug_diagnoses.append(dbg.diagnosis)
        rec.debug_attempts = attempt + 1

        if dbg.files.is_empty():
            rec.failed = True
            rec.failure_reason = "Debugger produced no fixed files."
            return finalize(rec, current)

        # Carry the fix forward. Keep the agent's config intact (per-field
        # patching re-applies channels/field on the next attempt); only adopt a
        # debugger config edit if one was actually provided.
        fixed = dbg.files.merged_onto(candidate)
        candidate = CodeFiles(fixed.model_code, fixed.losses_code, candidate.config_yaml)
        if dbg.files.config_yaml is not None:
            candidate = CodeFiles(candidate.model_code, candidate.losses_code,
                                  dbg.files.config_yaml)

    # --- 4. Promote checkpoints + eval (valid) per field -------------------
    # Copy each freshly trained checkpoint into the canonical coupling dir that
    # configs/ntc_{field}.yaml (and eval.py's coupled test) reads from, so that
    # `--eval_split test` after the run loads THIS iteration's three surrogates.
    for f in fields:
        src = field_exp[f] / "model.pth"
        dst_dir = project_dir / FIELD_COUPLE_EXP[f]
        if src.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst_dir / "model.pth")

    per_field_metrics: Dict[str, Dict[str, float]] = {}
    per_field_telemetry: Dict[str, Dict[str, object]] = {}
    num_params = None
    peak_mem = None
    telemetry_blocks = []

    for f in fields:
        print(f"  Iteration {iteration}: EVAL (valid) field '{f}'")
        run_eval(
            project_dir=str(project_dir), config_path=config_rels[f], split="valid",
            log_path=str(iter_dir / f"eval_stdout_{f}.txt"), gpu_id=gpu_id,
        )

        results_pkl = field_exp[f] / "results.pkl"
        results = {}
        if results_pkl.exists():
            with open(results_pkl, "rb") as fh:
                results = pickle.load(fh)
            telemetry_blocks.append(
                f"===== field '{f}' =====\n"
                + summarize_telemetry_text(results, gpu_memory_gb=gpu_memory_gb)
            )
        # Model size / peak memory: same architecture across fields; report the
        # max peak memory seen and the (identical up to channels) param count.
        if results.get("num_params") is not None:
            num_params = max(num_params or 0, results["num_params"])
        if results.get("peak_gpu_memory_gb") is not None:
            peak_mem = max(peak_mem or 0.0, results["peak_gpu_memory_gb"])

        per_field_metrics[f] = read_valid_metrics(field_exp[f] / "eval_valid_results.txt")
        per_field_telemetry[f] = summarize_telemetry(results) if results else {}

    (iter_dir / "telemetry.txt").write_text("\n\n".join(telemetry_blocks)
                                            or "no telemetry")

    # Aggregate: rank on the MEAN validation RMSE across the three fields.
    rec.num_params = num_params
    rec.peak_gpu_memory_gb = peak_mem
    rec.metrics = average_field_metrics(per_field_metrics)
    rec.per_field_metrics = per_field_metrics
    rec.valid_mse, base_name = primary_metric(rec.metrics)
    rec.primary_metric_name = f"avg{base_name}" if base_name else ""
    # Store the per-field telemetry keyed by field, plus a compact per-field
    # RMSE map, so the orchestrator sees which field is the weak link.
    rec.telemetry_summary = {
        "per_field": per_field_telemetry,
        "per_field_rmse": {f: per_field_metrics[f].get("rmse")
                           for f in fields if per_field_metrics.get(f)},
    }

    primary_str = (f"{rec.primary_metric_name}={rec.valid_mse:.7f}"
                   if rec.valid_mse is not None else "N/A")
    per_field_str = ", ".join(
        f"{f}:RMSE={per_field_metrics[f].get('rmse'):.5f}"
        for f in fields if per_field_metrics.get(f) and per_field_metrics[f].get("rmse") is not None
    ) or "N/A"

    np_str = f"{rec.num_params/1e6:.2f}M" if rec.num_params else "N/A"
    mem_str = f"{rec.peak_gpu_memory_gb:.3f} GB" if rec.peak_gpu_memory_gb else "N/A"
    print(f"\n  Iteration {iteration} done. "
          f"Valid {primary_str} | per-field [{per_field_str}] | "
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
            lines.append(f"- iter {h.iteration}: {name}={h.valid_mse:.7f}")
        lines.append("")

    for h in history:
        name = h.primary_metric_name or "metric"
        vl = f"{name}={h.valid_mse:.7f}" if h.valid_mse is not None else "N/A"
        np_str = f"{h.num_params/1e6:.2f}M" if h.num_params else "N/A"
        mem_str = f"{h.peak_gpu_memory_gb:.3f} GB" if h.peak_gpu_memory_gb else "N/A"
        status = "FAILED" if h.failed else "ok"

        lines.append(f"## Iteration {h.iteration} ({status})")
        lines.append(f"- Validation (primary, mean over fields): {vl}")
        lines.append(f"- Mean validation metrics: {_fmt_metrics(h.metrics)}")
        if h.per_field_metrics:
            pf = "; ".join(f"{fld}: {_fmt_metrics(m)}"
                           for fld, m in h.per_field_metrics.items())
            lines.append(f"- Per-field validation metrics: {pf}")
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