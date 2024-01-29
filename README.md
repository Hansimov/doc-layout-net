# Doc-Layout-Net
Codes for Document Layout Analysis data processing and model training.

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
HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ds4sd/DocLayNet-v1.1 --include "*.parquet" --repo-type dataset --local-dir ./datasets
```

More details about huggingface-cli and hf-mirror:
- https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-download
- https://hf-mirror.com/  

I recommend to use huggingface-cli as it supports resuming download and saving to specific directory.

## Process dataset