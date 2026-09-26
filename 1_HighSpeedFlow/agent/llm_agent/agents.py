from .llm_client import LLMClient
from . import roles
from . import parser


class Orchestrator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, iteration, gpu_memory_gb, memory_context=""):
        system = roles.ORCHESTRATOR_SYSTEM
        user = roles.build_orchestrator_prompt(
            iteration=iteration,
            gpu_memory_gb=gpu_memory_gb,
            memory_context=memory_context,
        )
        raw = self.llm.chat(system, user)
        return system, user, raw, parser.parse_orchestrator(raw)


class Coder:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def write(self, iteration, gpu_memory_gb, plan,
              current_model, current_losses, current_config):
        system = roles.CODER_SYSTEM
        user = roles.build_coder_prompt(
            iteration=iteration,
            gpu_memory_gb=gpu_memory_gb,
            plan=plan,
            current_model=current_model,
            current_losses=current_losses,
            current_config=current_config,
        )
        raw = self.llm.chat(system, user)
        return system, user, raw, parser.parse_coder(raw)


class Debugger:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def fix(self, error_log, model_code, losses_code, config_yaml):
        system = roles.DEBUGGER_SYSTEM
        user = roles.build_debugger_prompt(
            error_log=error_log,
            model_code=model_code,
            losses_code=losses_code,
            config_yaml=config_yaml,
        )
        raw = self.llm.chat(system, user)
        return system, user, raw, parser.parse_debugger(raw)