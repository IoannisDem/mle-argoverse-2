import torch
from data_creation import data_utils
from utils import read_yaml
import json
from pathlib import Path


def main():
    config = data_utils.parse_meta_env_config(read_yaml(data_utils.CONFIG_PATH))
    config_env = config.env_config
    env = data_utils.define_env(config_env)
    spec = f"traffic_{config_env.traffic_density}_accident_{config_env.accident_prob}_steps_{config.max_steps}"
    output_path = Path(config.output_path) / spec
    max_steps = config.max_steps

    summary = []

    try:
        for episode_idx in range(config_env.num_scenarios):
            seed = config_env.start_seed + episode_idx

            episode_result = data_utils.collect_episode(env, seed, max_steps)
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
