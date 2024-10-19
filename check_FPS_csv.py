import os

import pandas as pd

result2 = []
def check_fps(file_path):
    """Checks if the FPS in the CSV file is consistent.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        bool: True if the FPS is consistent, False otherwise.
    """

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        return False  # Handle empty files gracefully

    fps_values = df['fps'].unique()
    if len(fps_values) > 1:
        return False  # FPS is inconsistent
    if fps_values[0] != 25.0:
        if fps_values[0] != 30.0:
            result2.append(file_path)
        return False
    return True

# Replace 'your_folder_path' with the actual path to your folder
folder_path = './map-keyframes/'
result = []
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        if not check_fps(file_path):
            print(f"File {file_name} has inconsistent FPS.")
            result.append(file_name[:-4])
import json

with open('fps30.json', 'w') as f:
    json.dump(result, f)
print(result)
print(result2)

import os

import pandas as pd


def check_n_sequence(csv_file):
  df = pd.read_csv(csv_file)
  n_series = df['n']
  return (n_series == range(1, len(n_series) + 1)).all()

# Replace 'your_folder_path' with the actual path to your folder
folder_path = './map-keyframes/'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for csv_file in csv_files:
  file_path = os.path.join(folder_path, csv_file)
  if not check_n_sequence(file_path):
    print(f"{file_path} has non-consecutive 'n' values.")




print("Done!")