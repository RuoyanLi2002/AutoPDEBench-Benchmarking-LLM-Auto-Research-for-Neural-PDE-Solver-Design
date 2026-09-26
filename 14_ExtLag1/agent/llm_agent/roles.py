DATASET_DESCRIPTION = """\
## Dataset Description

This dataset contains piston-driven fluid dynamics in Lagrangian particle
format. A piston at the left boundary of the tank undergoes prescribed
horizontal motion, pushing the fluid and driving waves through the domain.
Each particle is labeled by type (0 = fluid, 1 = piston): fluid particles obey
the free dynamics, while piston particles follow a known rigid trajectory that
acts as the external forcing of the system.

## Modeling Objective

Let X^t in R^{N x 15} denote the system state at time t, where N is the number
of particles. The 15 features per particle are: the last step position (2),
the historical velocity over 5 frames (5 x 2 = 10), the next-step velocity
(2), and a piston indicator column (1). The next-step velocity slot is a
known-driver feature: for piston particles it contains the true (normalized)
t+1 velocity of the prescribed motion, while for fluid particles it is zero
since t+1 is exactly the quantity to be predicted; the indicator column lets
the model distinguish a masked zero from a genuinely zero driver. The goal is
to learn a parameterized neural simulator Psi such that
    V_hat^{t+1} = Psi(X^t)
where V_hat^{t+1} in R^{N x 2} is the (normalized, per-type) particle velocity
at the next time step. The loss is evaluated on fluid particles only, since
piston motion is prescribed rather than predicted.

## Dataset-Specific Challenges

1. The fluid is driven entirely by the piston boundary: momentum enters the
   system through a thin layer of fluid particles in contact with the piston
   and must propagate correctly through the bulk. The model must accurately
   handle this external force, resolving the sharp velocity discontinuity at
   the moving wall without smearing the impulse or letting fluid penetrate
   the piston.
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
    def forward(self, x): x is [B, N, 15];
        returns [B, N, 2], the predicted velocity.

losses.py — defines the TRAINING objective:
    def compute_loss(out, y, x, attr):
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