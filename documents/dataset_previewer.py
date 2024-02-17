import io
import os
import pandas as pd

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pprint import pprint

datasets_root = Path(__file__).parents[1] / "datasets"


class DatasetRowDecomposer:
    def decompose(self, row):
        row_dict = {
            "image_bytes": row["image"]["bytes"],
            "category_ids": row["category_id"],
            "bboxes": row["bboxes"],
            "doc_category": row["metadata"]["doc_category"],
        }
        return row_dict


class CategoryBoxViewer:
    CATEGORY_NAMES = {
        1: "Caption",
        2: "Footnote",
        3: "Formula",
        4: "List-item",
        5: "Page-footer",
        6: "Page-header",
        7: "Picture",
        8: "Section-header",
        9: "Table",
        10: "Text",
        11: "Title",
    }
    # https://www.rapidtables.com/web/color/RGB_Color.html
    CATEGORY_COLORS = {
        "Caption": (255, 178, 102),
        "Footnote": (64, 255, 64),
        "Formula": (0, 255, 255),
        "List-item": (128, 255, 128),
        "Page-footer": (100, 100, 100),
        "Page-header": (100, 100, 100),
        "Picture": (255, 102, 255),
        "Section-header": (255, 128, 0),
        "Table": (255, 255, 102),
        "Text": (204, 255, 204),
        "Title": (255, 128, 0),
    }
    CATEGORY_NEW_NAMES = {
        "Section-header": "Header",
        "List-item": "List",
    }

    def calc_rect_box(self, bbox, spacing=2):
        rect_box = [
            bbox[0] - spacing,
            bbox[1] - spacing,
            bbox[0] + bbox[2] + spacing,
            bbox[1] + bbox[3] + spacing,
        ]
        return rect_box

    def view(self, image_bytes, category_ids, bboxes, spacing=2):
        image = Image.open(io.BytesIO(image_bytes))
        draw = ImageDraw.Draw(image, "RGBA")

        for idx, (category_id, bbox_array) in enumerate(zip(category_ids, bboxes)):
            category = self.CATEGORY_NAMES[category_id]
            bbox = list(bbox_array)
            print(f"{category}: {bbox}")
            color = self.CATEGORY_COLORS[category]
            rect_box = self.calc_rect_box(bbox, spacing)
            draw.rectangle(rect_box, outline=color, fill=(*color, 64), width=2)

        text_font = ImageFont.truetype("times.ttf", 15)
        for idx, (category_id, bbox_array) in enumerate(zip(category_ids, bboxes)):
            category_name = self.CATEGORY_NAMES[category_id]
            if category_name in self.CATEGORY_NEW_NAMES:
                category_name = self.CATEGORY_NEW_NAMES[category]
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
        self.row_decomposer = DatasetRowDecomposer()

    def preview(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        print(df.columns)
        head_rows = df.head(10)
        print(head_rows)
        sample_row = head_rows.iloc[7]
        image_label_info = self.row_decomposer.decompose(sample_row)

        self.category_box_viewer.view(
            image_bytes=image_label_info["image_bytes"],
            category_ids=image_label_info["category_ids"],
            bboxes=image_label_info["bboxes"],
        )


if __name__ == "__main__":
    previewer = DatasetPreviewer()
    parquets_root = datasets_root / "parquets"
    file_path = sorted(list(parquets_root.glob("train-*.parquet")))[0]
    previewer.preview(parquet_path=(parquets_root / file_path).resolve())

    # python -m documents.dataset_previewer
