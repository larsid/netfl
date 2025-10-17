import sys
import json
from pathlib import Path
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
    }
)


def load_experiment_data(file_path: str) -> dict:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {e}") from e


def compute_averages(data):
    train_data = data["train"]

    name = data["name"]

    avg_exchange_time = mean(
        d["update_exchange_time"] for d in train_data if "update_exchange_time" in d
    )

    avg_train_time = mean(d["train_time"] for d in train_data)

    return {
        "Name": name,
        "Avg Update Exchange Time (s)": avg_exchange_time,
        "Avg Training Time (s)": avg_train_time,
    }


def plot_horizontal_bar_chart(data_device_1: dict, data_device_2: dict):
    metrics = ["Avg Update Exchange Time (s)", "Avg Training Time (s)"]

    bar_height = 0.15
    offset = bar_height * 1.5
    group_spacing = bar_height * 3.5

    y_pos = np.arange(len(metrics)) * group_spacing

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.barh(
        y_pos + offset / 2,
        [data_device_2[m] for m in metrics],
        height=bar_height,
        label=data_device_2["Name"],
        color="#2980B9",
    )

    ax.barh(
        y_pos - offset / 2,
        [data_device_1[m] for m in metrics],
        height=bar_height,
        label=data_device_1["Name"],
        color="#27AE60",
    )

    for i, metric in enumerate(metrics):
        v1 = data_device_1[metric]
        v2 = data_device_2[metric]
        ax.text(
            v1 + max(v1, v2) * 0.01,
            y_pos[i] - offset / 2,
            f"{v1:.2f}",
            va="center",
            fontsize=9,
        )
        ax.text(
            v2 + max(v1, v2) * 0.01,
            y_pos[i] + offset / 2,
            f"{v2:.2f}",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_title("Impact of the Device Heterogeneity", pad=15)
    ax.set_xlabel("Seconds")
    ax.legend(loc="upper right")

    ax.set_xlim(
        0,
        max(
            max([data_device_1[m] for m in metrics]),
            max([data_device_2[m] for m in metrics]),
        )
        * 1.2,
    )

    plt.tight_layout()
    output_path = "time_vs_device_heterogeneity.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved as '{output_path}'")


def validate_args(args: list[str]) -> str:
    if len(args) != 2:
        raise ValueError(
            "Usage: python time_vs_device_heterogeneity.py <path_to_experiment_result.json>"
        )
    return args[1]


if __name__ == "__main__":
    try:
        file_path = validate_args(sys.argv)
        experiment_result = load_experiment_data(file_path)

        data_device_1 = compute_averages(experiment_result["device_1"])
        data_device_2 = compute_averages(experiment_result["device_2"])

        plot_horizontal_bar_chart(data_device_1, data_device_2)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
