import json
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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


def load_experiment_data(file_path: str):
	path = Path(file_path)
	if not path.is_file():
		print(f"Error: File not found - {file_path}")
		sys.exit(1)
	with open(path, 'r') as f:
		return json.load(f)


def plot_accuracy_vs_round(experiment_result):
	fig, ax = plt.subplots(figsize=(7.5, 4))

	for device, data in experiment_result.items():
		rounds = [entry["round"] for entry in data]
		accuracies = [entry["accuracy"] for entry in data]

		marker_rounds = [r for r in rounds if r % 50 == 0]
		marker_accuracies = [a for r, a in zip(rounds, accuracies) if r % 50 == 0]

		ax.plot(
			rounds,
			accuracies,
			label=f"{device} devices",
			color=device_colors.get(device, "#000000"),
			linewidth=1.5,
		)
		ax.plot(
			marker_rounds,
			marker_accuracies,
			linestyle='',
			marker='o',
			color=device_colors.get(device, "#000000"),
			markersize=4,
		)

	ax.set_title("Impact of the Number of Clients on Test Accuracy", pad=15)
	ax.set_xlabel("Number of Rounds")
	ax.set_ylabel("Test Accuracy")
	ax.legend(loc="lower right", frameon=False)

	ax.xaxis.set_major_locator(mticker.MultipleLocator(100))

	plt.tight_layout()
	output_path = "accuracy_vs_round.png"
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	print(f"Figure saved as '{output_path}'")


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python accuracy_vs_round.py <path_to_experiment_result.json>")
		sys.exit(1)

	file_path = sys.argv[1]
	experiment_result = load_experiment_data(file_path)
	experiment_result = dict(sorted(experiment_result.items(), key=lambda item: item[0], reverse=True))
	plot_accuracy_vs_round(experiment_result)
