import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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


device_colors = {
	"08": "#F1C40F",
	"16": "#2980B9",
	"32": "#27AE60",
	"64": "#C0392B"
}


def load_experiment_data(file_path: str) -> dict:
	path = Path(file_path)
	if not path.is_file():
		raise FileNotFoundError(f"File not found: {file_path}")
	
	try:
		with open(path, 'r') as f:
			return json.load(f)
	except json.JSONDecodeError as e:
		raise ValueError(f"Invalid JSON in file {file_path}: {e}") from e


def compute_avg_exchange_time(data):
	averages = {}
	for device, entries in data.items():
		exchange_times = [entry["update_exchange_time"] for entry in entries if "update_exchange_time" in entry]
		if exchange_times:
			avg = sum(exchange_times) / len(exchange_times)
			averages[device] = avg
	return averages


def plot_horizontal_bar_chart(averages):
	devices = sorted(averages.keys(), key=lambda d: int(d))
	times = [averages[device] for device in devices]
	colors = [device_colors.get(device, "#000000") for device in devices]

	fig, ax = plt.subplots(figsize=(7.5, 4))
	y_pos = np.arange(len(devices)) * 0.2

	bars = ax.barh(y_pos, times, color=colors, height=0.15)

	ax.set_yticks(y_pos)
	ax.set_yticklabels([f"{device} devices" for device in devices])
	ax.set_title("Impact of the Number of Clients on Update Exchange Time", pad=15)
	ax.set_xlabel("Avg Update Exchange Time (s)")
	ax.set_ylabel("Number of Devices")

	max_time = max(times)
	ax.set_xlim((0, max_time * 1.15))

	for i, bar in enumerate(bars):
		width = bar.get_width()
		label = f"{times[i]:.2f}"
		ax.text(
			width + (max_time * 0.01),
			bar.get_y() + bar.get_height() / 2,
			label,
			va='center',
			ha='left',
			color='black',
			fontsize=9
		)

	plt.tight_layout()
	output_path = "client_vs_update_exchange_time.png"
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	print(f"Figure saved as '{output_path}'")


def validate_args(args: list[str]) -> str:
	if len(args) != 2:
		raise ValueError("Usage: python client_vs_update_exchange_time.py <path_to_experiment_result.json>")
	return args[1]


if __name__ == "__main__":
	try:
		file_path = validate_args(sys.argv)
		experiment_result = load_experiment_data(file_path)
		experiment_result = dict(sorted(experiment_result.items(), key=lambda item: item[0], reverse=True))
		avg_times = compute_avg_exchange_time(experiment_result)
		plot_horizontal_bar_chart(avg_times)
	except (ValueError, FileNotFoundError) as e:
		print(f"Error: {e}", file=sys.stderr)
		sys.exit(1)
	except Exception as e:
		print(f"Unexpected error: {e}", file=sys.stderr)
		sys.exit(1)
