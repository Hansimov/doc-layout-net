from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]

DATASETS_ROOT = REPO_ROOT / "datasets"
PARQUETS_ROOT = DATASETS_ROOT / "parquets"
TRANSFORMED_PARQUETS_ROOT = DATASETS_ROOT / "transformed_parquets"

WEIGHTS_ROOT = REPO_ROOT / "weights"
CHECKPOINTS_ROOT = WEIGHTS_ROOT / "checkpoints"

OUTPUTS_ROOT = REPO_ROOT / "outputs"
RUNS_ROOT = REPO_ROOT / "runs"

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
