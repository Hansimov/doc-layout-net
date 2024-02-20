import pandas as pd
import torchvision.transforms.functional as torch_func

from pathlib import Path


def xywh_to_x1y1x2y2(xywh, spacing=2):
    x1y1x2y2 = [
        xywh[0] - spacing,
        xywh[1] - spacing,
        xywh[0] + xywh[2] + spacing,
        xywh[1] + xywh[3] + spacing,
    ]
    return x1y1x2y2


def normalize_x1y1x2y2(x1y1x2y2, width, height):
    x1, y1, x2, y2 = x1y1x2y2
    norm_x1 = x1 / width
    norm_y1 = y1 / height
    norm_x2 = x2 / width
    norm_y2 = y2 / height
    return [norm_x1, norm_y1, norm_x2, norm_y2]


def denormalize_x1y1x2y2(normalized_x1y1x2y2, width, height):
    norm_x1, norm_y1, norm_x2, norm_y2 = normalized_x1y1x2y2
    x1 = int(norm_x1 * width)
    y1 = int(norm_y1 * height)
    x2 = int(norm_x2 * width)
    y2 = int(norm_y2 * height)
    return [x1, y1, x2, y2]


def x1y1x2y2_with_spacing(x1y1x2y2, spacing=2):
    x1, y1, x2, y2 = x1y1x2y2
    x1 -= spacing
    y1 -= spacing
    x2 += spacing
    y2 += spacing
    return [x1, y1, x2, y2]


def decompose_dataset_row(row):
    row_dict = {
        "image_bytes": row["image"]["bytes"],
        "category_ids": row["category_id"],
        "bboxes": row["bboxes"],
        "doc_category": row["metadata"]["doc_category"],
    }
    return row_dict


def image_to_tensor(image, device):
    image_tensor = torch_func.to_tensor(image).to(device)
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
