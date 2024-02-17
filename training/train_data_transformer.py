import io
import re
import torch
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from constants.dataset_info import (
    CATEGORY_COLORS,
    CATEGORY_NAMES,
    CATEGORY_NEW_NAMES,
    PARQUETS_ROOT,
    TRANSFORMED_PARQUETS_ROOT,
)
from documents.parquet_converter import (
    DatasetRowDecomposer,
    xywh_to_x1y1x2y2,
    normalize_x1y1x2y2,
)


class TrainDataTransformer:
    def __init__(self):
        self.transformed_df = pd.DataFrame(
            columns=[
                "image_tensor",
                "bboxes_tensor",
                "category_ids_tensor",
            ]
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_output_path(self, parquet_path):
        output_parent = TRANSFORMED_PARQUETS_ROOT
        output_name = parquet_path.name
        output_path = output_parent / output_name
        return output_path

    def load_parquet_as_df(self, suffix="train"):
        # TODO: batch processing for multiple parquets
        self.parquet_path = sorted(list(PARQUETS_ROOT.glob(f"{suffix}-*.parquet")))[0]
        self.transformed_parquet_path = self.get_output_path(self.parquet_path)
        df = pd.read_parquet(self.parquet_path)
        return df

    def transform(self):
        df = self.load_parquet_as_df(suffix="train")
        decomposer = DatasetRowDecomposer()
        for row_idx, row in tqdm(df.iterrows(), total=len(df)):
            row_dict = decomposer.decompose(row)
            # process image: bytes_to_np_array, transpose, to_tensor, normalize
            image = Image.open(io.BytesIO(row_dict["image_bytes"]))
            image_width, image_height = image.size
            image_np = np.array(image)
            if len(image_np.shape) == 2:
                # add channel dimension
                image_np = np.expand_dims(image_np, axis=0)
            else:
                # transpose to channel-first
                image_np = np.transpose(image_np, (2, 0, 1))
            image_tensor = torch.tensor(image_np).to(self.device)
            image_tensor = image_tensor.float() / 255

            # process bboxes: xywh_to_x1y1x2y2, normalize, to_tensor
            if len(row_dict["bboxes"]) == 0:
                bboxes_tensor = torch.tensor([]).to(self.device)
            else:
                bboxes_stack = np.stack(row_dict["bboxes"]).astype(np.float32)
                bboxes_transformed = [
                    normalize_x1y1x2y2(
                        xywh_to_x1y1x2y2(bbox), image_width, image_height
                    )
                    for bbox in bboxes_stack
                ]
                bboxes_tensor = torch.tensor(bboxes_transformed).to(self.device)

            # process category_ids: to_tensor, validate
            category_ids = row_dict["category_ids"]
            category_ids_tensor = torch.tensor(category_ids).to(self.device)

            # concate transformed_df
            transformed_row = {
                "image_tensor": image_tensor,
                "bboxes_tensor": bboxes_tensor,
                "category_ids_tensor": category_ids_tensor,
            }
            self.transformed_df = pd.concat(
                [self.transformed_df, pd.DataFrame([transformed_row])]
            )

        # # save transformed_df to parquet
        # if not self.transformed_parquet_path.exists:
        #     self.transformed_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        # print(self.transformed_df.head())
        # self.transformed_df.to_parquet(self.transformed_parquet_path)


if __name__ == "__main__":
    transformer = TrainDataTransformer()
    transformer.transform()

    # python -m training.train_data_transformer
