from pathlib import Path
from pdf2image import convert_from_path


def pdf_to_images(pdf_path, output_path=None):
    pdf_path = Path(pdf_path)
    if not output_path:
        output_path = pdf_path.parent / f"{pdf_path.stem}"
    if not output_path.exists():
        output_path.mkdir(parents=True)

    images = convert_from_path(
        pdf_path,
        fmt="jpg",
        size=(1025, None),
        output_folder=output_path,
        output_file="page_",
    )
    return images


if __name__ == "__main__":
    pdf_root = Path(__file__).parents[1] / "samples"
    pdf_name = "2402.13255 - How NeRFs and 3D Gaussian Splatting are Reshaping SLAM - a Survey.pdf"
    pdf_path = pdf_root / pdf_name
    print(f"> Converting pdf to images: {pdf_path}")
    images = pdf_to_images(pdf_path)
    # python -m documents.pdf_converter
