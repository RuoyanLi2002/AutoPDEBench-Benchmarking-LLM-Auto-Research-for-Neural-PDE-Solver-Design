import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


def _run(cmd: List[str], cwd: str, log_path: Optional[str] = None,
         gpu_id: Optional[int] = None) -> Tuple[int, str]:
    if gpu_id is not None:
        cmd_str = (
            f"export CUDA_VISIBLE_DEVICES={gpu_id} && "
            + " ".join(shlex.quote(c) for c in cmd)
        )
        print(f"\n>>> {cmd_str}  (cwd={cwd})")
        process = subprocess.Popen(
            cmd_str,
            cwd=cwd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    else:
        print(f"\n>>> {' '.join(cmd)}  (cwd={cwd})")
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        lines.append(line)

    return_code = process.wait()
    output = "".join(lines)

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            f.write(output)

    return return_code, output


def run_training(project_dir: str, config_path: str,
                 log_path: Optional[str] = None,
                 gpu_id: Optional[int] = None) -> Tuple[int, str]:
    """Invoke `python main.py --config <path> --to_train`."""
    return _run(
        ["python", "main.py", "--config", config_path, "--to_train"],
        cwd=project_dir,
        log_path=log_path,
        gpu_id=gpu_id,
    )


def run_eval(project_dir: str, config_path: str, split: str = "valid",
             log_path: Optional[str] = None,
             gpu_id: Optional[int] = None) -> Tuple[int, str]:
    """Invoke `python main.py --config <path> --eval_split <split>`."""
    return _run(
        ["python", "main.py", "--config", config_path, "--eval_split", split],
        cwd=project_dir,
        log_path=log_path,
        gpu_id=gpu_id,
    )