DATASET_DESCRIPTION = """\
## Dataset Description

This is a 2D circular blast simulation. A central high-pressure zone creates an
outward shock wave and an inward expansion wave. The shock wave repeatedly
reflects between the boundaries and the origin, losing strength with each
bounce. As reflecting shocks interact with the post-shock gas, they generate
turbulent instabilities. 3,500 training samples, 500 validation, 500 test.

## Modeling Objective

Let X^{t1} in R^{4 x 128 x 128} denote the system state at time t1. The four
feature channels are: temperature, density, x-velocity, y-velocity, each on a
128 x 128 spatial grid. The initial time t1 in R and the target time t2 in R
are explicitly provided as inputs. The goal is to learn a parameterized neural
simulator Psi such that

    X_hat^{t2} = Psi(X^{t1}, t1, t2)

where X_hat^{t2} has the same shape (4, 128, 128) as the input.

## Dataset-Specific Challenges

1. Standard neural solvers assume smooth temporal evolution. This dataset
   contains compressible flows at or above the speed of sound, producing
   abrupt, highly non-linear discontinuities (shock waves).
2. The numerical solver adapts its time resolution to capture sudden changes,
   so the step delta_t = t2 - t1 is non-uniform across samples. The model must
   condition on t1 and t2 (or delta_t).
"""


TASK_AND_GOAL = """\
## Task

Design a neural surrogate that maps an initial state and the source/target times
to the future state, handling the dataset challenges above that standard solvers
do not. The team works iteratively: design, train for a fixed budget, evaluate on
validation, then refine. The aim is to maximize validation accuracy and surpass
the strongest existing neural solvers for this problem; a minimal baseline is
not the target.
"""

CONSTRAINTS_BODY = """\
- GPU memory: {gpu_memory_gb} GB. Size the model and batch to use this budget
  with headroom for activations and optimizer state; there is no parameter cap.
  You do not need to use all available GPU memory. Select model size and batch
  size that produce the best performance.
- Training is fixed at 100 epochs. Set num_epochs: 100.
"""

CODE_CONTRACT = """\
## Editable Files

Exactly three files may change; everything else is fixed.

model.py — defines `class Model(nn.Module)`:
    def __init__(self, cfg): reads fields from cfg.model.*
    def forward(self, x, tx, ty): x is [B,C,H,W], tx/ty are [B,1];
        returns [B,C,H,W], the predicted state at t2.

losses.py — defines the TRAINING objective:
    def compute_loss(out, y, x, tx, ty):
        returns (scalar_loss_tensor, components_dict)
    where components_dict maps names to float sub-losses for logging (may be
    empty). compute_loss is used for training only. Validation and test are
    scored separately with a fixed set of reported metrics. Changing
    compute_loss changes what the model optimizes but not how it is reported;
    the leaderboard ranks on validation RMSE.

config.yaml — has model: and training: sections. training: contains optimizer:
    and scheduler: blocks, each with name: (a class in torch.optim /
    torch.optim.lr_scheduler) and params: (its kwargs), and may include an
    optional grad_clip: float. Required training fields: seed, num_epochs,
    batch_size, save_freq, optimizer, scheduler. Do not add a base_config: line.
    Do not set exp_name or model_path; the harness manages experiment paths and
    the checkpoint location.

Available: torch, torch.nn, torch.nn.functional, numpy.
"""


# =============================================================================
# ROLE: ORCHESTRATOR
# =============================================================================

ORCHESTRATOR_SYSTEM = """\
You are the lead orchestrator of a team building a neural PDE surrogate. You set
strategy and do not write code.

Each round, read the analyst's report and the Run Memory, identify the single
most important bottleneck, and give the coder a plan that changes one thing at a
time — usually one file — so its effect is attributable. Architecture and
optimization are the primary levers; leave the loss at its default unless the
evidence shows the objective is the bottleneck. Defer secondary ideas to later
rounds.

The Run Memory lists every prior iteration's decision and resulting validation
metrics and flags the best so far. Treat that best iteration as the baseline to
beat: build on it rather than on a configuration that regressed, never re-propose
a change already shown to hurt or to fail, and state how this round is expected
to improve on the best.
"""

ORCHESTRATOR_FORMAT = """\
## Response Format

## Situation
Where the project stands and the current bottleneck.

## Strategy
The single direction for this round and why it targets the bottleneck or a
dataset challenge.

## Files To Change
Which of model.py, losses.py, config.yaml to edit this round.

## Plan
Numbered instructions for the coder, each precise enough to implement directly.
Use only as many steps as the change needs.

Do not write code.
"""


# =============================================================================
# ROLE: CODER
# =============================================================================

CODER_SYSTEM = """\
You are the coder. You implement the orchestrator's plan as correct, complete
PyTorch: sound tensor shapes, proper device handling, no unused imports, no
placeholders. Implement architectures at a competitive scale rather than a
reduced version. Change only what the plan specifies and leave everything else
identical, so the effect of the change can be isolated.
"""

CODER_FORMAT = """\
## Response Format

## Notes
One paragraph on how the code realizes the plan.

Then emit only the files you changed, each complete, in a fenced block whose
first line names the file:

```python
# model.py
...
```
```python
# losses.py
...
```
```yaml
# config.yaml
...
```

Any file you do not emit is carried forward unchanged. Emit at least one file.
"""


# =============================================================================
# ROLE: DEBUGGER
# =============================================================================

DEBUGGER_SYSTEM = """\
You are the debugger. Given the current files and a runtime error, make the
smallest change that lets the code run. Do not redesign the model, change the
architecture's intent, alter the loss strategy, or adjust hyperparameters
except where one is the direct cause of the failure (such as a shape mismatch
or an out-of-memory error). Preserve the original design.
"""

DEBUGGER_FORMAT = """\
## Response Format

## Diagnosis
The root cause, citing the specific symptom.

## Fix
The minimal change being made.

Then emit only the file(s) you changed, complete, in fenced blocks whose first
line names the file (# model.py, # losses.py, or # config.yaml).
"""


# =============================================================================
# Builders
# =============================================================================

def _constraints_section(gpu_memory_gb: float) -> str:
    """Standalone `## Constraints` section (used by the coder)."""
    return "## Constraints\n\n" + CONSTRAINTS_BODY.format(gpu_memory_gb=gpu_memory_gb)


def build_task_section(gpu_memory_gb: float) -> str:
    """Task, goal, and constraints combined into one `## Task` section, with the
    constraints rendered as a `### Constraints` subsection."""
    return (TASK_AND_GOAL.rstrip()
            + "\n\n### Constraints\n\n"
            + CONSTRAINTS_BODY.format(gpu_memory_gb=gpu_memory_gb))


def build_orchestrator_prompt(iteration: int, gpu_memory_gb: float,
                              memory_context: str = "") -> str:
    if iteration == 0:
        context = (
            "This is iteration 0; there is no prior run. Propose the initial "
            "architecture, loss strategy, and training recipe."
        )
    else:
        context = (
            f"This is iteration {iteration}.\n\n"
            f"## Run Memory (all prior iterations; lower validation RMSE is better)\n"
            f"{memory_context or '(none)'}"
        )

    return "\n\n".join([
        DATASET_DESCRIPTION,
        build_task_section(gpu_memory_gb),
        CODE_CONTRACT,
        context,
        ORCHESTRATOR_FORMAT,
    ])


def build_coder_prompt(iteration: int, gpu_memory_gb: float, plan: str,
                       current_model: str, current_losses: str,
                       current_config: str) -> str:
    current = (
        "## Current Files\n\n"
        f"### model.py\n```python\n{current_model}\n```\n\n"
        f"### losses.py\n```python\n{current_losses}\n```\n\n"
        f"### config.yaml\n```yaml\n{current_config}\n```"
    )
    plan_block = f"## Plan To Implement\n{plan}"

    return "\n\n".join([
        _constraints_section(gpu_memory_gb),
        CODE_CONTRACT,
        current,
        plan_block,
        CODER_FORMAT,
    ])


def build_debugger_prompt(error_log: str, model_code: str, losses_code: str,
                          config_yaml: str) -> str:
    files = (
        "## Failing Files\n\n"
        f"### model.py\n```python\n{model_code}\n```\n\n"
        f"### losses.py\n```python\n{losses_code}\n```\n\n"
        f"### config.yaml\n```yaml\n{config_yaml}\n```"
    )
    err = "## Runtime Error\n```\n" + error_log + "\n```"
    return "\n\n".join([
        CODE_CONTRACT,
        files,
        err,
        DEBUGGER_FORMAT,
    ])