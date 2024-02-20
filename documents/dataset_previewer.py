import io
import os
import pandas as pd

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pprint import pprint
from documents.parquet_converter import decompose_dataset_row, xywh_to_x1y1x2y2
from constants.dataset_info import (
    CATEGORY_COLORS,
    CATEGORY_NAMES,
    CATEGORY_NEW_NAMES,
    PARQUETS_ROOT,
)


class CategoryBoxViewer:
    def view(self, image_bytes, category_ids, bboxes, spacing=2):
        image = Image.open(io.BytesIO(image_bytes))
        draw = ImageDraw.Draw(image, "RGBA")

        for idx, (category_id, bbox_array) in enumerate(zip(category_ids, bboxes)):
            category = CATEGORY_NAMES[category_id]
            bbox = list(bbox_array)
            print(f"{category}: {bbox}")
            color = CATEGORY_COLORS[category]
            rect_box = xywh_to_x1y1x2y2(bbox, spacing)
            draw.rectangle(rect_box, outline=color, fill=(*color, 64), width=2)

        text_font = ImageFont.truetype("times.ttf", 15)
        for idx, (category_id, bbox_array) in enumerate(zip(category_ids, bboxes)):
            category_name = CATEGORY_NAMES[category_id]
            if category_name in CATEGORY_NEW_NAMES:
                category_name = CATEGORY_NEW_NAMES[category]
            bbox = list(bbox_array)
            text_str = f"{idx+1}.{category_name}"
            draw.text(
                (bbox[0] - 2 * spacing, bbox[1] - 2 * spacing),
                text_str,
                fill="black",
                font=text_font,
                anchor="rt",
            )
        image.show()


class DatasetPreviewer:
    def __init__(self):
        self.category_box_viewer = CategoryBoxViewer()

    def preview(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        print(df.columns)
        head_rows = df.head(40)
        print(head_rows)
        sample_row = head_rows.iloc[31]
        image_label_info = decompose_dataset_row(sample_row)

        self.category_box_viewer.view(
            image_bytes=image_label_info["image_bytes"],
            category_ids=image_label_info["category_ids"],
            bboxes=image_label_info["bboxes"],
        )


if __name__ == "__main__":
    previewer = DatasetPreviewer()
    file_path = sorted(list(PARQUETS_ROOT.glob("train-*.parquet")))[0]
    previewer.preview(parquet_path=(PARQUETS_ROOT / file_path).resolve())

    # python -m documents.dataset_previewer
