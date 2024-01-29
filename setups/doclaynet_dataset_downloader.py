from datasets import load_dataset


class DoclaynetDatasetDownloader:
    def __init__(self):
        self.dataset_name = "ds4sd/DocLayNet"

    def download(self):
        load_dataset(self.dataset_name, trust_remote_code=True)


if __name__ == "__main__":
    downloader = DoclaynetDatasetDownloader()
    downloader.download()

    # HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 python -m setups.doclaynet_dataset_downloader
