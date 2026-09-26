import argparse

from llm_agent.agent import run_agent


def main():
    parser = argparse.ArgumentParser(
        description="Multi-agent LLM team that iteratively designs, trains, and "
                    "refines a neural surrogate (architecture + loss) for one "
                    "field of the coupled neutronics-thermal-fluid nuclear "
                    "dataset."
    )
    parser.add_argument("--num_iterations", type=int, default=5,
                        help="Number of design-train-eval cycles to run.")
    parser.add_argument("--run_dir", type=str, default="agent_runs/run_0",
                        help="Directory where all iteration artifacts are saved.")
    parser.add_argument("--project_dir", type=str, default=".",
                        help="Project root containing main.py (default: current dir).")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                        help="OpenAI model name. 'gpt-4o-mini' for cheap testing; "
                             "a stronger model (e.g. gpt-5.5) for deployment.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for the LLM.")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="CUDA device index. If set, each subprocess runs with "
                             "`export CUDA_VISIBLE_DEVICES=<gpu_id>`.")
    parser.add_argument("--gpu_memory_gb", type=float, default=24.0,
                        help="Available GPU memory in GB, passed to the agents so "
                             "they size the model to the budget (no param cap).")
    parser.add_argument("--max_debug_attempts", type=int, default=2,
                        help="Max debugger fix attempts per iteration on a "
                             "training crash before giving up.")
    args = parser.parse_args()

    history = run_agent(
        num_iterations=args.num_iterations,
        run_dir=args.run_dir,
        project_dir=args.project_dir,
        model_name=args.model_name,
        temperature=args.temperature,
        gpu_memory_gb=args.gpu_memory_gb,
        gpu_id=args.gpu_id,
        max_debug_attempts=args.max_debug_attempts,
    )

    print(f"\n{'=' * 70}\n  Agent run complete: {len(history)} iterations\n{'=' * 70}")
    for h in history:
        name = h.primary_metric_name or "metric"
        vl = f"{name}={h.valid_mse:.7f}" if h.valid_mse is not None else "N/A"
        np_str = f"{h.num_params/1e6:.2f}M" if h.num_params else "N/A"
        status = "FAILED" if h.failed else "ok"
        print(f"  iter {h.iteration} ({status}): valid {vl} | "
              f"params = {np_str} | debug attempts = {h.debug_attempts}")


if __name__ == "__main__":
    main()
