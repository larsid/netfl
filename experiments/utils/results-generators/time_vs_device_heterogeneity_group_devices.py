import json
import sys
from pathlib import Path
from collections import defaultdict


def load_experiment_data(file_path: str):
	path = Path(file_path)
	if not path.is_file():
		print(f"Error: File not found - {file_path}")
		sys.exit(1)

	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def group_by_client_prefix(data):
	if not isinstance(data, list):
		raise ValueError("Input JSON must be a list of objects.")

	grouped = defaultdict(list)
	for record in data:
		client_name = record.get("client_name")
		if not client_name:
			raise ValueError(f"Missing 'client_name' in record: {record}")

		client_prefix = client_name.split("_", 1)[0]
		grouped[client_prefix].append(record)

	return grouped


def save_grouped_data(grouped_data, output_path: Path):
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(grouped_data, f, indent=2)
	print(f"Grouped data written to '{output_path}'")


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python time_vs_device_heterogeneity_group_devices.py <input_json_file>")
		sys.exit(1)

	file_path = sys.argv[1]
	data = load_experiment_data(file_path)
	grouped = group_by_client_prefix(data)

	output_path = Path.cwd() / "time_vs_device_heterogeneity_group_devices_result.json"
	save_grouped_data(grouped, output_path)
