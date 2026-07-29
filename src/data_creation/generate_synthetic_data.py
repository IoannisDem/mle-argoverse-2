import torch
from data_creation import data_utils
from utils import read_yaml
import json
from pathlib import Path


def main():
    config = data_utils.parse_meta_env_config(read_yaml(data_utils.CONFIG_PATH))
    config_env = config.env_config
    env = data_utils.define_env(config_env)
    output_path = Path(config.output_path)

    summary = []

    try:
        for episode_idx in range(config_env.num_scenarios):
            seed = config_env.start_seed + episode_idx

            episode_result = data_utils.collect_episode(env, seed)
            meta = data_utils.save_episode(
                episode_idx,
                seed,
                episode_result,
                output_path,
            )
            summary.append(meta)

    finally:
        env.close()

    n_success = sum(m["arrive_dest"] for m in summary)
    total_steps = sum(m["num_steps"] for m in summary)

    with open(data_utils.OUTPUT_DIR / "collection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Collection complete ---")
    print(
        f"Episodes: {len(summary)} | Successes: {n_success} | Total steps: {total_steps}"
    )
    print(f"Saved to: {data_utils.OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
