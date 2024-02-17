import numpy as np
import io
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from documents.parquet_converter import (
    DatasetRowDecomposer,
    xywh_to_x1y1x2y2,
    normalize_x1y1x2y2,
)
from constants.dataset_info import (
    CATEGORY_COLORS,
    CATEGORY_NAMES,
    CATEGORY_NEW_NAMES,
    PARQUETS_ROOT,
)


class DocElementDetector:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = len(CATEGORY_NAMES.keys()) + 1
        self.model = fasterrcnn_resnet50_fpn(num_classes=self.num_classes)
        self.model.to(self.device)
        self.decomposer = DatasetRowDecomposer()

    def load_parquet_as_df(self, suffix="train"):
        parquet_path = sorted(list(PARQUETS_ROOT.glob(f"{suffix}-*.parquet")))[0]
        df = pd.read_parquet(parquet_path)
        return df

    # def prepare_data(self):
    #     self.df_train = self.load_parquet_as_df(suffix="train")
    #     self.df_val = self.load_parquet_as_df(suffix="val")
    #     self.df_test = self.load_parquet_as_df(suffix="test")

    def train(self, batch_size=8, num_epochs=10, learning_rate=0.001):
        self.df_train = self.load_parquet_as_df(suffix="train")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # shuffle df_train, then split rows to batches
        self.df_train_shuffled = self.df_train.sample(frac=1).reset_index(drop=True)
        self.df_train_batches = [
            self.df_train.iloc[i : i + batch_size]
            for i in range(0, len(self.df_train_shuffled), batch_size)
        ]
        self.model.train()

        for batch_idx, batch in enumerate(self.df_train_batches):
            image_batch, bboxes_batch, category_ids_batch = [], [], []
            for row_idx, row in batch.iterrows():
                row_dict = self.decomposer.decompose(row)
                # skip if no bboxes
                if len(row_dict["bboxes"]) == 0:
                    continue

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

                image_batch.append(image_tensor)

                # process bboxes: xywh_to_x1y1x2y2, normalize, to_tensor
                bboxes_stack = np.stack(row_dict["bboxes"]).astype(np.float32)
                bboxes_transformed = [
                    normalize_x1y1x2y2(
                        xywh_to_x1y1x2y2(bbox), image_width, image_height
                    )
                    for bbox in bboxes_stack
                ]
                bboxes_tensor = torch.tensor(bboxes_transformed).to(self.device)
                bboxes_batch.append(bboxes_tensor)

                # process category_ids: to_tensor, validate
                category_ids = row_dict["category_ids"]
                category_ids_tensor = torch.tensor(category_ids).to(self.device)
                category_ids_batch.append(category_ids_tensor)

            # target: boxes, labels
            target_batch = [
                {
                    "boxes": bboxes_tensor,
                    "labels": category_ids_tensor,
                }
                for bboxes_tensor, category_ids_tensor in zip(
                    bboxes_batch, category_ids_batch
                )
            ]
            self.optimizer.zero_grad()
            loss_dict = self.model(image_batch, target_batch)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            self.optimizer.step()
            print(f"Batch: {batch_idx}, loss: {loss.item()}")
        print("Training complete.")


if __name__ == "__main__":
    detector = DocElementDetector()
    detector.train()
    # python -m training.doc_element_detector