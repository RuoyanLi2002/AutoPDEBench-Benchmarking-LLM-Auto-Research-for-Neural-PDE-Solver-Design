DATASET_DESCRIPTION = """\
## Dataset Description

This dataset comes from a multiphysics simulation with tight
coupling. A system is described by three physical
subsystems that continuously feed back on one another:

  - source field (source): the source flux / power distribution. 
    It depends on the fuel temperature and on the coolant state, and on 
    the system boundary conditions (control parameters).
  - Solid field (fuel heat conduction): the temperature field inside the solid
    fuel. It is driven by the heat source from the source field and
    cooled at its interface with the surrounding fluid.
  - Fluid field (coolant thermal-hydraulics): the coolant state carrying heat
    away from the fuel, described by several channels (temperature, a velocity /
    momentum-like quantity, and further hydraulic variables). It is driven by the
    heat flux leaving the fuel surface.

The three subsystems are solved by separate physics codes and coupled
through a fixed-point (Picard) iteration: each solver is run with the current
estimate of the other fields as its boundary/source condition, and the process
repeats until all three fields stop changing. This iteration is the expensive
part; the goal is to replace each physics solver with a fast neural surrogate.

The data is provided in two forms:

  1. Decoupled per-field data (used for training and validation). For each
     field, inputs are the conditioning fields that drive it and the target is
     that field's converged solution, computed by the reference solver in
     isolation. These are the (input, target) pairs your surrogate is trained
     and validated on.
  2. Coupled ground truth (used only for the final coupled test). Boundary
     conditions plus the fully converged source / solid / fluid fields from the
     reference coupled multiphysics run.

Every field is stored as a 5-D tensor with shape [B, C, D, H, W]: a batch of B
samples, C physical channels, and a 3-axis spatial grid (D, H, W). The grids
differ per field (in particular the last axis W differs: the source, solid and
fluid fields do not share the same W), so a surrogate must not hard-code one
spatial size. All fields are mapped to a normalized [-1, 1] range before the
model sees them and mapped back afterwards; this normalization is fixed by the
harness and is identical for training, validation, and the coupled test.

Per-field channel counts and roles:

  - source : input_dim = 2, output_dim = 1
              (conditioned on boundary conditions + the other fields; predicts
              the source field, 1 channel)
  - solid   : input_dim = 2, output_dim = 1
              (conditioned on the source field + the fluid interface state;
              predicts the fuel temperature, 1 channel)
  - fluid   : input_dim = 1, output_dim = 4
              (conditioned on the heat flux from the fuel; predicts the 4-channel
              coolant state)

## Modeling Objective

Learn one neural surrogate per field. Each surrogate Psi_field maps that field's
input conditioning tensor X in R^{C_in x D x H x W} to the field's solution
    Y_hat = Psi_field(X),   Y_hat in R^{C_out x D x H x W}
replacing one call to the corresponding legacy physics solver.

A single agent run trains and validates the surrogate for ONE field (selected in
config.yaml via data.field, with model.input_dim / model.output_dim set to that
field's channel counts). The reported, ranked score for the run is that field's
validation accuracy on held-out decoupled data. The ultimate purpose is that the
three trained surrogates are dropped into the Picard fixed-point loop in place of
the physics solvers and iterated to convergence (the coupled "test"); a surrogate
is only useful if it is both accurate in isolation and stable when its (imperfect)
outputs are fed back in as another field's inputs over many coupling iterations.
"""


TASK_AND_GOAL = """\
## Task

Design a neural surrogate for the current field that maps its conditioning
tensor to that field's converged solution, handling the dataset challenges below
that a naive regressor does not. The team works iteratively: design, train for a
fixed budget, evaluate on the decoupled validation split, then refine. The aim
is to maximize validation accuracy (minimize validation RMSE) and surpass the
strongest existing neural surrogates for this problem; a minimal baseline is not
the target. Because the surrogate is ultimately used inside the coupled
fixed-point iteration, accuracy that also degrades gracefully under small input
perturbations (rather than accuracy that relies on perfectly clean inputs) is
preferred.
"""

CONSTRAINTS_BODY = """\
- GPU memory: {gpu_memory_gb} GB. Size the model and batch to use this budget
  with headroom for activations and optimizer state; there is no parameter cap.
  You do not need to use all available GPU memory. Select model size and batch
  size that produce the best performance.
- Training is fixed at 100 epochs. Set num_epochs: 100.
- Data is 5-D: tensors are [B, C, D, H, W] with a 3-axis spatial grid. The
  spatial sizes (and the last axis W) differ per field, so the model must handle
  an arbitrary (D, H, W) rather than assume a fixed grid.
- Channel counts are fixed by the field and are set in config.yaml as
  model.input_dim / model.output_dim; the model must read them from cfg and not
  hard-code them. The output must have shape [B, output_dim, D, H, W], matching
  the target.
"""

CODE_CONTRACT = """\
## Editable Files

Exactly three files may change; everything else (data loading, the training
loop, the metrics, and the coupled fixed-point evaluation) is fixed and must not
be relied upon to change.

model.py — defines `class Model(nn.Module)`:
    def __init__(self, cfg): reads fields from cfg.model.* (e.g. input_dim,
        output_dim, and any hyperparameters you add under model: in config.yaml).
    def forward(self, x): x is [B, C_in, D, H, W];
        returns [B, C_out, D, H, W], the predicted field in normalized space,
        where C_in == cfg.model.input_dim and C_out == cfg.model.output_dim.
    The same Model class is instantiated for whichever field the run targets, so
    it must work for any (D, H, W) and for the different per-field channel
    counts. It is also the class loaded for all three fields during the coupled
    test, so keep it general across fields.

losses.py — defines the TRAINING objective:
    def compute_loss(out, y, x):
        out : model prediction [B, C_out, D, H, W] (normalized field space)
        y   : ground-truth target, same shape as out
        x   : the input conditioning tensor [B, C_in, D, H, W]
        returns (scalar_loss_tensor, components_dict)
    where components_dict maps names to float sub-losses for logging (may be
    empty). compute_loss is used for training only. Validation and test are
    scored separately with a fixed set of reported metrics. Changing
    compute_loss changes what the model optimizes but not how it is reported;
    the leaderboard ranks on validation RMSE.

config.yaml — has model: and training: sections. model: must include input_dim
    and output_dim (the field's channel counts) plus any hyperparameters your
    Model reads. training: contains optimizer: and scheduler: blocks, each with
    name: (a class in torch.optim / torch.optim.lr_scheduler) and params: (its
    kwargs), and may include an optional grad_clip: float. Required training
    fields: seed, num_epochs, batch_size, save_freq, optimizer, scheduler. Do
    not add a base_config: line. Do not set exp_name or model_path, and do not
    change data.field or the per-field input_dim/output_dim away from the field
    the run targets; the harness manages experiment paths, the checkpoint
    location, and which field is being trained.

Available: torch, torch.nn, torch.nn.functional, numpy, einops.
"""


# =============================================================================
# ROLE: ORCHESTRATOR
# =============================================================================

ORCHESTRATOR_SYSTEM = """\
You are the lead orchestrator of a team building a neural surrogate for one
field of a coupled neutronics-thermal-fluid nuclear reactor simulation. You set
strategy and do not write code.

Each round, read the analyst's report and the Run Memory, identify the single
most important bottleneck, and give the coder a plan that changes one thing at a
time — usually one file — so its effect is attributable. Architecture and
optimization are the primary levers; leave the loss at its default unless the
evidence shows the objective is the bottleneck. Keep in mind that this surrogate
must later run inside a fixed-point coupling iteration, so prefer designs that
are accurate AND robust to slightly imperfect inputs over ones that merely fit
the clean validation pairs. Defer secondary ideas to later rounds.

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
placeholders. The data is 5-D ([B, C, D, H, W]) with per-field spatial sizes, so
write shape-agnostic code that reads channel counts from cfg and never hard-codes
a grid size. Implement architectures at a competitive scale rather than a reduced
version. Change only what the plan specifies and leave everything else identical,
so the effect of the change can be isolated.
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
on the 5-D [B, C, D, H, W] tensors or an out-of-memory error). Preserve the
original design.
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

    # The coder does not need the full dataset description: tensor shapes and the
    # file/API surface live in CODE_CONTRACT, and the orchestrator's plan
    # encodes any dataset-specific modeling decisions already.
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
