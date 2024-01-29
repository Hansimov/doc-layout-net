from pathlib import Path
import pandas as pd


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
    file_path = "test-00000-of-00002-635b47e9044a436c.parquet"
    previewer.preview(parquet_path=datasets_root / file_path)

    # python -m documents.dataset_previewer
