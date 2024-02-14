import os
import pandas as pd
from pathlib import Path


class DatasetPreviewer:
    def __init__(self):
        pass

    def preview(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        print(df.columns)
        print(df.head())


if __name__ == "__main__":
    previewer = DatasetPreviewer()
    datasets_root = Path(__file__).parents[1] / "datasets" / "parquets"
    file_path = sorted(list(datasets_root.glob("train-*.parqLuet")))[0]
    previewer.preview(parquet_path=(datasets_root / file_path).resolve())

    # python -m documents.dataset_previewer
