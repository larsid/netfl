import json
import sys
from pathlib import Path
from collections import defaultdict


def load_experiment_data(file_path: str) -> dict:
	path = Path(file_path)
	if not path.is_file():
		raise FileNotFoundError(f"File not found: {file_path}")

	try:
		with open(path, "r", encoding="utf-8") as f:
			return json.load(f)
	except json.JSONDecodeError as e:
		raise ValueError(f"Invalid JSON in file {file_path}: {e}") from e


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


def validate_args(args: list[str]) -> str:
	if len(args) != 2:
		raise ValueError("Usage: python time_vs_device_heterogeneity_group_devices.py <input_json_file>")
	return args[1]


if __name__ == "__main__":
	try:
		file_path = validate_args(sys.argv)
		data = load_experiment_data(file_path)
		grouped = group_by_client_prefix(data)

		output_path = Path.cwd() / "time_vs_device_heterogeneity_group_devices_result.json"
		save_grouped_data(grouped, output_path)
	except (ValueError, FileNotFoundError) as e:
		print(f"Error: {e}", file=sys.stderr)
		sys.exit(1)
	except Exception as e:
		print(f"Unexpected error: {e}", file=sys.stderr)
		sys.exit(1)
