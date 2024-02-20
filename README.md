# Doc-Layout-Net
Codes for Document Layout Analysis data processing and model training.


## Environments

<div align="center">

![](https://img.shields.io/badge/GPU-RTX%204080-green?logo=nvidia) ![](https://img.shields.io/badge/NVIDIA%20Driver-535.154.05-blue?logo=nvidia) ![](https://img.shields.io/badge/CUDA-12.2-blue?logo=nvidia)

![](https://img.shields.io/badge/Ubuntu-22.04.3%20LTS-blue?logo=ubuntu) ![](https://img.shields.io/badge/Python-3.11.7-blue?logo=python) ![](https://img.shields.io/badge/PyTorch-2.1.2-blue?logo=pytorch) ![](https://img.shields.io/badge/DocLayNet-1.1-blue?logo=ibm)

</div>

## Download dataset

<div align="center">

DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis

[arXiv](https://arxiv.org/abs/2206.01062)
· [PDF](https://arxiv.org/pdf/2206.01062.pdf)
· [GitHub](https://github.com/DS4SD/DocLayNet)
· [HF-Dataset](https://huggingface.co/datasets/ds4sd/DocLayNet)
· [HF-Dataset-v1](https://huggingface.co/datasets/ds4sd/DocLayNet-v1.1)

</div>

```sh
# pip install huggingface_hub
# For PRC users, hf-mirror is recommended
HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ds4sd/DocLayNet-v1.1 --include "*.parquet" --repo-type dataset --local-dir ./datasets/parquets --local-dir-use-symlinks False
```

More details about huggingface-cli and hf-mirror:
- https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-download
- https://hf-mirror.com/  

I recommend to use huggingface-cli as it supports resuming download and saving to specific directory.

## Process dataset

### Formats of DocLayNet-v1.1

See more: [DocLayNet-v1.1-Dataset-Card](https://huggingface.co/datasets/ds4sd/DocLayNet-v1.1?row=0#dataset-card-for-doclaynet-v11)


#### Rows of dataset

The dataset is provided with parquets (train + test + val), each row in the parquet follows the format below:

|    Field     |      Type      |                                           Description                                           |
| ------------ | -------------- | ----------------------------------------------------------------------------------------------- |
| image        | (bytes)        | Page-level PIL image                                                                            |
| bboxes       | (2d-list)      | List of layout bounding boxes                                                                   |
| category_id  | (list)         | List of class ids corresponding to the bounding boxes                                           |
| segmentation | (2d-list)      | List of layout segmentation polygons                                                            |
| area         | (list)         |                                                                                                 |
| pdf_cells    | (list of dict) | List of lists corresponding to bbox. Each list contains the PDF cells (content) inside the bbox |
| metadata     | (dict)         | Meta infos of Page and document                                                                 |

#### bboxes classes / categories


1: Caption · 2: Footnote · 3: Formula · 4: List-item · 5: Page-footer · 6: Page-header · 7: Picture · 8: Section-header · 9: Table · 10: Text
· 11: Title

#### doc_category enums

1: financial_reports · 2: scientific_articles · 3: laws_and_regulations · 4: government_tenders · 5: manuals · 6: patents

## References

* fasterrcnn_resnet50_fpn — Torchvision main documentation
  * https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html

* ReduceLROnPlateau — PyTorch 2.2 documentation
  * https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
