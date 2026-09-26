import re
from dataclasses import dataclass, field
from typing import Dict, Optional




def extract_section(text: str, header: str) -> str:
    """Return text under ``## {header}`` up to the next ``## `` header, fence, or EOF."""
    pattern = re.compile(
        r"##\s*" + re.escape(header) + r"\s*\n([\s\S]*?)(?=\n##\s|\n```|\Z)",
        re.IGNORECASE,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else ""




_FENCE = re.compile(
    r"```(python|py|yaml|yml)\s*\n([\s\S]*?)```",
    re.IGNORECASE,
)

_FILENAME_COMMENT = re.compile(
    r"^\s*#\s*(model\.py|losses\.py|config\.yaml)\b",
    re.IGNORECASE,
)


@dataclass
class CodeFiles:
    """Holds whichever of the three editable files a response contained."""
    model_code: Optional[str] = None
    losses_code: Optional[str] = None
    config_yaml: Optional[str] = None

    def is_empty(self) -> bool:
        return (self.model_code is None
                and self.losses_code is None
                and self.config_yaml is None)

    def merged_onto(self, base: "CodeFiles") -> "CodeFiles":
        """Return a copy of `base` with any non-None fields from `self` applied."""
        return CodeFiles(
            model_code=self.model_code if self.model_code is not None else base.model_code,
            losses_code=self.losses_code if self.losses_code is not None else base.losses_code,
            config_yaml=self.config_yaml if self.config_yaml is not None else base.config_yaml,
        )


def parse_code_files(text: str) -> CodeFiles:
    """Pull any of model.py / losses.py / config.yaml out of fenced blocks.

    Routing priority for each block:
      1. Leading `# <filename>` comment inside the block.
      2. Fence language: yaml/yml -> config.yaml; python/py -> model.py, unless
         a python block clearly defines compute_loss (then -> losses.py).
    """
    files = CodeFiles()

    for lang, body in _FENCE.findall(text):
        body_stripped = body.strip()
        first_line = body_stripped.splitlines()[0] if body_stripped else ""
        name_match = _FILENAME_COMMENT.match(first_line)

        target = None
        if name_match:
            target = name_match.group(1).lower()
        else:
            lang_l = lang.lower()
            if lang_l in ("yaml", "yml"):
                target = "config.yaml"
            else:  # python
                # Heuristic: a python block defining compute_loss is losses.py
                if re.search(r"def\s+compute_loss\s*\(", body_stripped):
                    target = "losses.py"
                else:
                    target = "model.py"

        if target == "model.py" and files.model_code is None:
            files.model_code = body_stripped
        elif target == "losses.py" and files.losses_code is None:
            files.losses_code = body_stripped
        elif target == "config.yaml" and files.config_yaml is None:
            files.config_yaml = body_stripped

    return files


# --- role-specific parsed structures ----------------------------------------

@dataclass
class OrchestratorOutput:
    situation: str
    strategy: str
    files_to_change: str
    plan: str
    raw: str


def parse_orchestrator(text: str) -> OrchestratorOutput:
    return OrchestratorOutput(
        situation=extract_section(text, "Situation"),
        strategy=extract_section(text, "Strategy"),
        files_to_change=extract_section(text, "Files To Change"),
        plan=extract_section(text, "Plan"),
        raw=text,
    )


@dataclass
class CoderOutput:
    notes: str
    files: CodeFiles
    raw: str


def parse_coder(text: str) -> CoderOutput:
    return CoderOutput(
        notes=extract_section(text, "Notes"),
        files=parse_code_files(text),
        raw=text,
    )


@dataclass
class DebuggerOutput:
    diagnosis: str
    fix: str
    files: CodeFiles
    raw: str


def parse_debugger(text: str) -> DebuggerOutput:
    return DebuggerOutput(
        diagnosis=extract_section(text, "Diagnosis"),
        fix=extract_section(text, "Fix"),
        files=parse_code_files(text),
        raw=text,
    )