import time
import csv
from pathlib import Path

# Define path for execution times log
csv_path = Path("reports/experiments/execution_times.csv")

# Ensure the folder exists
csv_path.parent.mkdir(parents=True, exist_ok=True)

# Logs execution time into a CSV file.
def log_execution_time(model_name, dataset_name, start_time, end_time):
	"""
	Args:
		model_name (str): The name of the model used.
		dataset_name (str): The dataset used.
		start_time (float): The start time (use time.time()).
		end_time (float): The end time (use time.time()).
	"""
	execution_time = end_time - start_time
	hours = int(execution_time // 3600)
	minutes = int((execution_time % 3600) // 60)
	seconds = int(execution_time % 60)
	formatted_time = f"{hours}H:{minutes}M:{seconds}S"

	# Check if CSV exists to add headers if necessary
	file_exists = csv_path.exists()

	# Save execution time with model and dataset info
	with csv_path.open(mode="a", newline="") as file:
		writer = csv.writer(file)

		# Write headers only if the file is new
		if not file_exists:
			writer.writerow(
				["Model", "Dataset", "Timestamp", "Execution Time"])

		writer.writerow([model_name, dataset_name, time.strftime(
			"%Y-%m-%d %H:%M:%S"), formatted_time])

	print(f"✅ Execution time logged: {formatted_time}")
