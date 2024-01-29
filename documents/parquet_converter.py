from pathlib import Path
import pandas as pd


class ParquetToCSVConverter:
    def __init__(self):
        pass

    def convert(self, parquet_path, csv_path):
        df = pd.read_parquet(parquet_path)
        df.to_csv(csv_path)


if __name__ == "__main__":
    converter = ParquetToCSVConverter()
    datasets_root = Path(__file__).parents[1] / "datasets"

    parquet_path = "test-00000-of-00002-635b47e9044a436c.parquet"
    csv_path = parquet_path.replace(".parquet", ".csv")
    converter.convert(
        parquet_path=datasets_root / "parquets" / parquet_path,
        csv_path=datasets_root / "csvs" / csv_path,
    )

    # python -m documents.parquet_converter
