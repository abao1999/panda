import json
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np

import wandb


def reconstruct_noise_schedule(
    epochs: list[float] | np.ndarray,
    epoch_stop: float,
    start: float,
    end: float,
    eps: float,
) -> np.ndarray:
    epochs = np.array(epochs)
    # Vectorized computation for the noise schedule
    condition = epochs >= epoch_stop  # or noise_schedule < end
    noise_schedule = np.where(
        condition,
        end,
        end
        + (start - end)
        * np.cos(((epochs / epoch_stop) + eps) / (1 + eps) * np.pi / 2) ** 2,
    )
    return noise_schedule


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    # Authenticate with W&B
    wandb.login()
    # Initialize the W&B API
    api = wandb.Api()
    # Fetch the run using the run ID
    run = api.run(
        f"{cfg.wandb.entity}/{cfg.wandb.project_name}/{cfg.run_metrics.wandb_run_id}"
    )

    metrics_dict = {}
    metric_names = [
        "train/epoch",
        "train/loss",
        "train/learning_rate",
        "train/grad_norm",
    ]

    plot_save_dir = os.path.join(cfg.run_metrics.plot_dir, cfg.run_metrics.wandb_run_id)
    os.makedirs(plot_save_dir, exist_ok=True)

    metrics_save_path = os.path.join(
        cfg.run_metrics.save_dir, cfg.run_metrics.save_fname
    )
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)

    for metric_name in metric_names:
        metrics = run.history(keys=[metric_name], pandas=False)  # list of metric dicts
        mname = metric_name.split("train/")[-1]  # abbreviated metric name
        metrics_dict[mname] = [m[metric_name] for m in metrics]
        # steps = [m["_step"] for m in metrics]

        if "epoch" in metrics_dict and mname != "epoch":
            print(f"shape of {mname}: {len(metrics_dict[mname])}")
            plt.plot(metrics_dict["epoch"][1:], metrics_dict[mname])
            plot_save_path = os.path.join(plot_save_dir, f"{mname}.png")
            plt.title(f"{mname.replace('_', ' ').title()}")
            plt.xlabel("Epoch")
            plt.tight_layout()
            plt.savefig(plot_save_path)
            plt.close()
            print(f"plot saved to {plot_save_path}")

    # epochs = np.linspace(0, 1, 201)
    epochs = metrics_dict["epoch"]
    noise_schedule = reconstruct_noise_schedule(
        epochs, cfg.noiser.epoch_stop, cfg.noiser.start, cfg.noiser.end, cfg.noiser.eps
    )
    metrics_dict["noise_schedule"] = noise_schedule.tolist()
    plt.plot(epochs, noise_schedule)
    plt.title("Noise Schedule")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_dir, "noise_schedule.png"))
    plt.close()

    print(f"metrics_dict: {metrics_dict}")

    with open(metrics_save_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"metrics saved to {metrics_save_path}")


if __name__ == "__main__":
    main()
