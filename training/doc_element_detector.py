import io
import json
import numpy as np
import pandas as pd
import torch

from datetime import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
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
    WEIGHTS_ROOT,
    CHECKPOINTS_ROOT,
)
from utils.logger import logger, Runtimer, DummySummaryWriter


class DocElementDetector:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = len(CATEGORY_NAMES.keys()) + 1
        self.decomposer = DatasetRowDecomposer()

    def load_parquet_as_df(self, suffix="train", num=1):
        all_parquet_paths = list(PARQUETS_ROOT.glob(f"{suffix}-*.parquet"))
        num = min(num, len(all_parquet_paths))
        loaded_parquet_paths = sorted(all_parquet_paths)[:num]

        dfs = []
        for idx, path in enumerate(loaded_parquet_paths):
            try:
                logger.file(f"  - [{idx+1}/{len(loaded_parquet_paths)}] {path}")
                df = pd.read_parquet(path)
                dfs.append(df)
            except Exception as e:
                logger.err(f"Error when loading: {path}")
                raise e

        df_concat = pd.concat(dfs, ignore_index=True)
        logger.success(f"  + Loaded {num} {suffix} parquets with {len(df_concat)} rows")
        return df_concat

    def batchize_df(self, df, batch_size=8, shuffle=True, seed=None):
        # shuffle df, then split rows to batches
        if shuffle:
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        if batch_size <= 1 or len(df) <= batch_size:
            df_batches = [df]
        else:
            df_batches = [
                df.iloc[i : i + batch_size] for i in range(0, len(df), batch_size)
            ]
        return df_batches

    def tensorize_row_dict(self, row_dict):
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
        bboxes_stack = np.stack(row_dict["bboxes"]).astype(np.float32)
        bboxes_transformed = [
            normalize_x1y1x2y2(xywh_to_x1y1x2y2(bbox), image_width, image_height)
            for bbox in bboxes_stack
        ]
        bboxes_tensor = torch.tensor(bboxes_transformed).to(self.device)

        # process category_ids: to_tensor, validate
        category_ids = row_dict["category_ids"]
        category_ids_tensor = torch.tensor(category_ids).to(self.device)

        return {
            "image_tensor": image_tensor,
            "bboxes_tensor": bboxes_tensor,
            "category_ids_tensor": category_ids_tensor,
        }

    def batch_to_inputs(self, batch):
        image_batch, bboxes_batch, category_ids_batch = [], [], []
        for row_idx, row in batch.iterrows():
            row_dict = self.decomposer.decompose(row)
            # skip if no bboxes
            if len(row_dict["bboxes"]) == 0:
                continue
            try:
                tensorized_row_dict = self.tensorize_row_dict(row_dict)
                image_batch.append(tensorized_row_dict["image_tensor"])
                bboxes_batch.append(tensorized_row_dict["bboxes_tensor"])
                category_ids_batch.append(tensorized_row_dict["category_ids_tensor"])
            except Exception as e:
                logger.warn(f"  x Error when tensorizing row {row_idx}: {e}")
                continue

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
        return (image_batch, target_batch)

    def save_checkpoint_info(self, checkpoint_path):
        checkpoint_info_path = self.checkpoint_parent / "checkpoint_info.json"
        checkpoint_info = {
            "checkpoint_path": str(checkpoint_path),
            "saved_datetime": datetime.now().isoformat(),
        }
        logger.success(f"  > Saving checkpoint info: {checkpoint_info_path}")
        with open(checkpoint_info_path, "w") as wf:
            json.dump(checkpoint_info, wf, indent=4)

    def save_checkpoint(self, epoch_idx, train_batch_idx):
        # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html#save-the-general-checkpoint
        if not self.checkpoint_parent.exists():
            self.checkpoint_parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = (
            self.checkpoint_parent
            / f"checkpoint_epoch_{epoch_idx+1}_batch_{train_batch_idx+1}.pth"
        )
        checkpoint_dict = {
            "epoch_idx": epoch_idx,
            "train_batch_idx": train_batch_idx,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        logger.success(f"  > Saving checkpoint: {checkpoint_path}")
        torch.save(checkpoint_dict, checkpoint_path)
        self.save_checkpoint_info(checkpoint_path)

    def load_checkpoint(self):
        checkpoint_info_path = self.checkpoint_parent / "checkpoint_info.json"
        if checkpoint_info_path.exists():
            with open(checkpoint_info_path, "r") as rf:
                checkpoint_info = json.load(rf)
                logger.note(f"> Loading checkpoint info: {checkpoint_info_path}")
            checkpoint_path = Path(checkpoint_info["checkpoint_path"])
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                logger.success(f"  > Checkpoint loaded: {checkpoint_path}")
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                epoch_idx_offset = checkpoint["epoch_idx"]
                train_batch_idx_offset = checkpoint["train_batch_idx"] + 1
                return epoch_idx_offset, train_batch_idx_offset
            else:
                logger.warn(f"  x Not found checkpoint: {checkpoint_path}")
                return 0, 0
        else:
            logger.warn(f"x Not found checkpoint info: {checkpoint_info_path}")
            return 0, 0

    def train(
        self,
        epoch_count=1,
        batch_size=8,
        learning_rate=1e-6,
        train_parquets_num=1,
        df_shuffle_seed=None,
        val_batches_num=30,
        val_batch_interval=20,
        save_checkpoint_batch_interval=100,
        show_in_board=True,
        resume_from_checkpoint=False,
    ):
        # weights file name, checkpoint parent
        self.weights_name = f"pq_{train_parquets_num}_sd_{df_shuffle_seed}_ep_{epoch_count}_bs_{batch_size}_lr_{learning_rate}"
        self.checkpoint_parent = CHECKPOINTS_ROOT / self.weights_name

        # init model and optimizer, enter train mode
        self.model = fasterrcnn_resnet50_fpn(num_classes=self.num_classes)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        if resume_from_checkpoint:
            epoch_idx_offset, train_batch_idx_offset = self.load_checkpoint()
        self.model.train()
        logger.success("> Start training ...")

        # load train data
        logger.note("> Loading parquests ...")
        self.df_train = self.load_parquet_as_df(suffix="train", num=train_parquets_num)
        self.df_train_batches = self.batchize_df(
            self.df_train, batch_size=batch_size, seed=df_shuffle_seed
        )

        # load validation data
        self.df_val = self.load_parquet_as_df(suffix="val", num=1)
        self.df_val_batches = self.batchize_df(
            self.df_val, batch_size=batch_size, seed=df_shuffle_seed
        )
        self.df_val_batches = self.df_val_batches[:val_batches_num]

        # tensorboard
        if show_in_board:
            self.summary_writer = SummaryWriter()
        else:
            self.summary_writer = DummySummaryWriter()

        # train loop
        for epoch_idx in range(epoch_idx_offset, epoch_count):
            logger.mesg(f"> Epoch: {epoch_idx+1}/{epoch_count}")
            train_batch_count = len(self.df_train_batches)
            if epoch_idx > epoch_idx_offset:
                train_batch_idx_offset = 0
            for train_batch_idx, train_batch in enumerate(
                self.df_train_batches[train_batch_idx_offset:],
                start=train_batch_idx_offset,
            ):
                # train: forward, backward, step
                train_inputs = self.batch_to_inputs(train_batch)
                self.optimizer.zero_grad()
                train_loss_dict = self.model(*train_inputs)
                train_loss = sum(loss for loss in train_loss_dict.values())
                train_loss.backward()
                self.optimizer.step()
                # validate
                if ((train_batch_idx + 1) % val_batch_interval == 0) or (
                    train_batch_idx + 1 == train_batch_count
                ):
                    with torch.no_grad():
                        val_loss = 0
                        for val_batch_idx, val_batch in enumerate(self.df_val_batches):
                            val_batch_inputs = self.batch_to_inputs(val_batch)
                            val_loss_dict = self.model(*val_batch_inputs)
                            val_loss += sum(
                                val_loss for val_loss in val_loss_dict.values()
                            )
                        val_loss /= len(self.df_val_batches)
                    self.model.train()
                    # log loss, and update tensorboard
                    train_loss_value = train_loss.item()
                    val_loss_value = val_loss.item()
                    logger.line(
                        f"  - [{train_batch_idx+1}/{train_batch_count}] "
                        f"train: {round(train_loss_value,6)}, "
                        f"val: {round(val_loss_value,6)}"
                    )
                    self.summary_writer.add_scalars(
                        "Loss",
                        {
                            "train": train_loss_value,
                            "val": val_loss_value,
                        },
                        train_batch_idx + 1,
                    )
                # save checkpoints
                if ((train_batch_idx + 1) % save_checkpoint_batch_interval == 0) or (
                    train_batch_idx + 1 == train_batch_count
                ):
                    self.save_checkpoint(
                        epoch_idx=epoch_idx, train_batch_idx=train_batch_idx
                    )

        # save weights
        self.weights_path = (
            self.checkpoint_parent.parent / f"weights_{self.weights_name}.pth"
        )
        logger.success(f"> Saving weights: {self.weights_path}")
        torch.save(self.model.state_dict(), self.weights_path)

        self.summary_writer.close()
        logger.success("[Finished]")


if __name__ == "__main__":
    detector = DocElementDetector()
    with Runtimer():
        detector.train(
            epoch_count=1,
            batch_size=16,
            learning_rate=1e-6,
            train_parquets_num=30,
            df_shuffle_seed=1,
            val_batches_num=30,
            val_batch_interval=20,
            save_checkpoint_batch_interval=100,
            show_in_board=True,
            resume_from_checkpoint=True,
        )

    # python -m training.doc_element_detector
    # tensorboard --logdir=runs --host=0.0.0.0 --port=16006 --load_fast=true
