import pandas as pd
import torchvision.transforms.functional as torch_func

from pathlib import Path


def decompose_dataset_row(row):
    row_dict = {
        "image_bytes": row["image"]["bytes"],
        "category_ids": row["category_id"],
        "bboxes": row["bboxes"],
        "doc_category": row["metadata"]["doc_category"],
        "metadata": row["metadata"],
    }
    return row_dict


def image_to_tensor(image):
    image_tensor = torch_func.to_tensor(image)
    return image_tensor


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
