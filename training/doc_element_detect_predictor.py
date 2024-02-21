import json
import torch

from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)

from documents.parquet_converter import image_to_tensor
from constants.dataset_info import (
    CATEGORY_COLORS,
    CATEGORY_NAMES,
    CATEGORY_NEW_NAMES,
    NUM_CLASSES,
    WEIGHTS_ROOT,
    SAMPLES_ROOT,
)
from utils.logger import logger, Runtimer
from utils.transforms import int_x1y1x2y2, x1y1x2y2_with_spacing


class DocElementDetectPredictor:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = NUM_CLASSES

    def get_latest_weights_path(self):
        weights_paths = list(WEIGHTS_ROOT.glob("*.pth"))
        weights_paths = sorted(
            weights_paths, key=lambda x: x.stat().st_mtime, reverse=True
        )
        latest_weights_path = weights_paths[0]
        return latest_weights_path

    def load_weights(self, weights_path):
        self.model = fasterrcnn_resnet50_fpn(num_classes=self.num_classes)
        # self.model = fasterrcnn_resnet50_fpn_v2(num_classes=self.num_classes)
        state_dict = torch.load(weights_path)
        if weights_path.stem.startswith("checkpoint"):
            weights = state_dict["model_state_dict"]
        else:
            weights = state_dict
        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.eval()

    def image_to_tensor(self, image_path):
        image = Image.open(image_path)
        image_tensor = image_to_tensor(image).to(self.device)
        return image_tensor

    def prediction_to_dict_list(self, prediction, threshold=0.5):
        # get prediction items, filter by threshold, and convert to list
        boxes, labels, scores = list(map(prediction.get, ["boxes", "labels", "scores"]))
        boxes, labels, scores = list(
            map(lambda x: x[scores > threshold], [boxes, labels, scores])
        )
        boxes, labels, scores = list(map(lambda x: x.tolist(), [boxes, labels, scores]))
        boxes = [int_x1y1x2y2(box) for box in boxes]

        # predict results as dict, and dump to json
        predict_results = []
        for box, label, score in zip(boxes, labels, scores):
            predict_results.append(
                {
                    "box": box,
                    "label": label,
                    "score": score,
                }
            )
        return predict_results

    def calc_avg_score(self, predict_results):
        scores = list(map(lambda x: x["score"], predict_results))
        avg_score = sum(scores) / len(scores)
        return avg_score

    def prediction_to_image(self, image_path, predict_results, bbox_spacing=2):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image, "RGBA")

        for idx, predict_result in enumerate(predict_results):
            box, label, score = list(map(predict_result.get, ["box", "label", "score"]))
            category = CATEGORY_NAMES[label]
            category_name = CATEGORY_NAMES[label]
            if category_name in CATEGORY_NEW_NAMES:
                category_name = CATEGORY_NEW_NAMES[category_name]
            logger.line(f"  - {idx+1}.{category_name} ({round(score,2)})", end="")
            color = CATEGORY_COLORS[category]
            rect_box = x1y1x2y2_with_spacing(box, spacing=bbox_spacing)
            draw.rectangle(rect_box, outline=color, fill=(*color, 64), width=2)
        logger.line("")

        # sudo apt install ttf-mscorefonts-installer
        text_font = ImageFont.truetype("times.ttf", 15)
        for idx, predict_result in enumerate(predict_results):
            box, label, score = list(map(predict_result.get, ["box", "label", "score"]))
            category_name = CATEGORY_NAMES[label]
            if category_name in CATEGORY_NEW_NAMES:
                category_name = CATEGORY_NEW_NAMES[category_name]
            text_str = f"{idx+1}.{category_name} ({round(score, 2)})"
            draw.text(
                (box[0] - 2 * bbox_spacing, box[1] - 2 * bbox_spacing),
                text_str,
                fill="black",
                font=text_font,
                anchor="lb",
            )
        return image

    def predict(self, image_path, weights_path=None, threshold=0.5):
        if not hasattr(self, "model"):
            if not weights_path:
                weights_path = self.get_latest_weights_path()
            logger.note(f"> Loading weights from: {weights_path}")
            self.load_weights(weights_path)

        logger.note(f"> Predicting image: {image_path}")
        predict_json_path = image_path.parent / f"{image_path.stem}_predict.json"
        predict_image_path = image_path.parent / f"{image_path.stem}_predict.png"
        image_tensor = self.image_to_tensor(image_path)
        with torch.no_grad():
            prediction = self.model([image_tensor])[0]
        predict_results = self.prediction_to_dict_list(prediction, threshold)

        # save predict results to json
        with open(predict_json_path, "w") as wf:
            json.dump(predict_results, wf, indent=4)
        logger.success(f"  + Predict results saved to: {predict_json_path}")

        # save predict results to image
        image = self.prediction_to_image(
            image_path=image_path, predict_results=predict_results
        )
        image.save(predict_image_path)
        logger.success(f"  + Predict image saved to: {predict_image_path}")

        return predict_results


if __name__ == "__main__":
    with Runtimer():
        predictor = DocElementDetectPredictor()
        image_pattern = "*"
        image_paths = sorted(
            [
                path
                for path in SAMPLES_ROOT.glob(image_pattern)
                if not path.stem.endswith("_predict")
            ]
        )

        chekpoint_name = "fasterrcnn_resnet50_fpn_pq-10_sd-1_ep-3_bs-8_lr-0.001-auto/checkpoint_epoch_3_batch_2600.pth"
        weights_path = WEIGHTS_ROOT / chekpoint_name
        weights_path = None

        total_avg_score = 0
        for image_path in image_paths:
            predict_results = predictor.predict(
                image_path=image_path, weights_path=weights_path, threshold=0.6
            )
            total_avg_score += predictor.calc_avg_score(predict_results)
        avg_avg_score = total_avg_score / len(image_paths)
        logger.success(f"Average avg_score: {avg_avg_score:.2f}")

    # python -m training.doc_element_detect_predictor
