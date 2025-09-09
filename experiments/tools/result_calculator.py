import argparse
import json
from pathlib import Path
from statistics import mean


def calculate_general_results(data, max_cpus, max_memory_mb):
	train_data = data["train"]
	eval_data = data["evaluate"]

	test_accuracy = eval_data[-1]["accuracy"]
	convergence_speed = len(eval_data) - 1
	avg_train_time = mean(d["train_time"] for d in train_data)

	avg_memory = mean(d["memory_avg_mb"] for d in train_data)
	avg_memory_percent = (avg_memory / max_memory_mb) * 100

	avg_cpu = mean(d["cpu_avg_percent"] for d in train_data)
	avg_cpu_percent = avg_cpu / max_cpus

	exchange_times = [d["exchange_time"] for d in train_data if "exchange_time" in d]
	avg_exchange_time = mean(exchange_times) if exchange_times else -1.0

	metrics = [
		{"name": "Test Accuracy", "value": f"{test_accuracy}"},
		{"name": "Convergence Speed (no. rounds)", "value": f"{convergence_speed}"},
		{"name": "Avg Training Time (s)", "value": f"{avg_train_time}"},
		{"name": "Avg Memory Utilization (%)", "value": f"{avg_memory_percent}"},
		{"name": "Avg CPU Utilization (%)", "value": f"{avg_cpu_percent}"},
		{"name": "Avg Update Exchange Time (s)", "value": f"{avg_exchange_time}"}
	]

	return {"metrics": metrics}


def main():
	parser = argparse.ArgumentParser(description="Process JSON metrics and generate results.")
	parser.add_argument("input_path", help="Path to JSON metrics")
	parser.add_argument("--cpu", type=int, required=True, help="Max device CPU cores")
	parser.add_argument("--memory", type=int, required=True, help="Max device memory in MB")

	args = parser.parse_args()

	input_file = Path(args.input_path)

	output_file = input_file.with_name(f"{input_file.stem}-general-results.json")

	with open(input_file, "r") as f:
		data = json.load(f)

	result = calculate_general_results(data, args.cpu, args.memory)

	with open(output_file, "w") as f:
		json.dump(result, f, indent=2)

	print(f"Results written to: {output_file}")

if __name__ == "__main__":
	main()
