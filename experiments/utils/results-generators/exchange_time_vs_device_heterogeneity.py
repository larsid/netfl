import sys
import json
from pathlib import Path
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
	"text.usetex": False,
	"font.family": "serif",
	"axes.labelsize": 12,
	"axes.titlesize": 14,
	"legend.fontsize": 10,
	"xtick.labelsize": 10,
	"ytick.labelsize": 10,
	"figure.dpi": 300
})


def load_experiment_data(file_path: str):
	path = Path(file_path)
	if not path.is_file():
		print(f"Error: File not found - {file_path}")
		sys.exit(1)
	with open(file_path, 'r') as f:
		return json.load(f)


def compute_averages(data):
	name = data["name"]

	device_1_name = data["device_1"]["name"]
	device_1_avg_exchange_time = mean(d["exchange_time"] for d in data["device_1"]["train"] if "exchange_time" in d)

	device_2_name = data["device_2"]["name"]
	device_2_avg_exchange_time = mean(d["exchange_time"] for d in data["device_2"]["train"] if "exchange_time" in d)

	return {
		"name": name,
		"device_1_name": device_1_name,
		"device_1_avg_exchange_time": device_1_avg_exchange_time,
		"device_2_name": device_2_name,
		"device_2_avg_exchange_time": device_2_avg_exchange_time,
	}


def plot_horizontal_bar_chart(data_experiment_1: dict, data_experiment_2: dict, data_experiment_3: dict):
	experiments = [data_experiment_1, data_experiment_2, data_experiment_3]
	device_labels = [data_experiment_1["device_1_name"], data_experiment_1["device_2_name"]]

	bar_height = 0.15
	offset = bar_height * 1.5
	group_spacing = bar_height * 4

	y_pos = np.arange(len(experiments)) * group_spacing

	fig, ax = plt.subplots(figsize=(8, 6))

	device_1_values = [exp["device_1_avg_exchange_time"] for exp in experiments]
	device_2_values = [exp["device_2_avg_exchange_time"] for exp in experiments]

	experiment_names = [exp["name"] for exp in experiments]

	ax.barh(
		y_pos + offset / 2,
		device_2_values, 
		height=bar_height, 
		label=device_labels[1], 
		color="#2980B9"
	)

	ax.barh(
		y_pos - offset / 2, 
		device_1_values, 
		height=bar_height, 
		label=device_labels[0], 
		color="#27AE60"
	)

	for i in range(len(experiments)):
		v1 = device_1_values[i]
		v2 = device_2_values[i]
		ax.text(v1 + max(v1, v2) * 0.01, y_pos[i] - offset / 2, f"{v1:.2f}", va="center", fontsize=9)
		ax.text(v2 + max(v1, v2) * 0.01, y_pos[i] + offset / 2, f"{v2:.2f}", va="center", fontsize=9)

	ax.set_yticks(y_pos)
	ax.set_yticklabels(experiment_names)
	ax.set_title("Combined Impact of the Number of Clients and Device Heterogeneity", pad=15)
	ax.set_xlabel("Avg Update Exchange Time (s)")
	ax.legend(loc="lower right")

	max_value = max(device_1_values + device_2_values)
	ax.set_xlim(0, max_value * 1.2)

	plt.tight_layout()
	output_path = "exchange_time_vs_device_heterogeneity.png"
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	print(f"Figure saved as '{output_path}'")


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python exchange_time_vs_device_heterogeneity.py <path_to_experiment_result.json>")
		sys.exit(1)

	file_path = sys.argv[1]
	experiment_result = load_experiment_data(file_path)

	data_experiment_1 = compute_averages(experiment_result["experiment_1"])
	data_experiment_2 = compute_averages(experiment_result["experiment_2"])
	data_experiment_3 = compute_averages(experiment_result["experiment_3"])

	plot_horizontal_bar_chart(data_experiment_1, data_experiment_2, data_experiment_3)
