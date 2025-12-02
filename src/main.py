import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    dir_path = Path.cwd()
    path_to_data = dir_path / "src" / "synthetic_fraud_dataset.csv"

    df = pd.read_csv(path_to_data)
