# Doc-Layout-Net
Codes for Document Layout Analysis dataset preparation and model training

## Download dataset

<div align="center">

DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis

[arXiv](https://arxiv.org/abs/2206.01062)
· [PDF](https://arxiv.org/pdf/2206.01062.pdf)
· [GitHub](https://github.com/DS4SD/DocLayNet)
· [HuggingFace](https://huggingface.co/datasets/ds4sd/DocLayNet)


```sh
python -m setups.doclaynet_dataset_downloader

# For PRC users, hf-mirror is recommended
HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 python -m setups.doclaynet_dataset_downloader
```

</div>