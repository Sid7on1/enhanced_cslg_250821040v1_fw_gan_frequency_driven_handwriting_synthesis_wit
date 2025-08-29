import logging
import os
import tempfile
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_image(image_path: str) -> bool:
    """
    Validate the input image.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    bool: True if the image is valid, False otherwise.
    """
    try:
        image = Image.open(image_path)
        if image.mode not in ("L", "RGB"):
            raise ValueError("Image must be in L or RGB mode.")
        return True
    except IOError:
        return False

def extract_text_from_pdf(pdf_path: str, output_dir: str) -> List[str]:
    """
    Extract images from a PDF file and save them to the output directory.

    Parameters:
    pdf_path (str): Path to the input PDF file.
    output_dir (str): Directory to save the extracted images.

    Returns:
    List[str]: List of paths to the extracted image files.
    """
    if not os.path.exists(pdf_path):
        raise ValueError("PDF file not found.")

    # Create temporary directory to save PDF pages
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert PDF to images and save in temporary directory
        os.system(f"pdftoppm {pdf_path} {temp_dir}/page")

        # Get list of image files in the temporary directory
        image_files = [
            os.path.join(temp_dir, file) for file in os.listdir(temp_dir) if file.endswith(".ppm")
        ]

        # Convert images to JPEG format and save in output directory
        extracted_images = []
        for image_file in image_files:
            base_name = os.path.basename(image_file)
            output_path = os.path.join(output_dir, f"{base_name}.jpg")
            os.system(f"convert {image_file} {output_path}")
            extracted_images.append(output_path)

    return extracted_images

def preprocess_image(
    image_path: str,
    output_path: str,
    resize_dim: Optional[Tuple[int, int]] = None,
    grayscale: bool = False,
) -> None:
    """
    Preprocess an image by resizing and converting to grayscale if specified.

    Parameters:
    image_path (str): Path to the input image file.
    output_path (str): Path to save the preprocessed image.
    resize_dim (Optional[Tuple[int, int]]): Dimension to resize the image.
    grayscale (bool): Whether to convert the image to grayscale.
    """
    try:
        image = Image.open(image_path)
        if resize_dim:
            image = image.resize(resize_dim)
        if grayscale:
            image = image.convert("L")
        image.save(output_path)
        logger.info(f"Preprocessed image saved to {output_path}")
    except IOError as e:
        logger.error(f"Error preprocessing image: {e}")

def batch_preprocess_images(
    input_dir: str,
    output_dir: str,
    resize_dim: Optional[Tuple[int, int]] = None,
    grayscale: bool = False,
) -> None:
    """
    Preprocess a batch of images in the input directory and save them to the output directory.

    Parameters:
    input_dir (str): Directory containing the input images.
    output_dir (str): Directory to save the preprocessed images.
    resize_dim (Optional[Tuple[int, int]]): Dimension to resize the images.
    grayscale (bool): Whether to convert the images to grayscale.
    """
    if not os.path.exists(input_dir):
        raise ValueError("Input directory not found.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_images = [
        os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.lower().endswith((".jpg", ".png", ".bmp", ".tiff"))
    ]

    for image_path in input_images:
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, base_name)
        preprocess_image(image_path, output_path, resize_dim, grayscale)

def apply_canny_edge_detection(image_path: str, output_path: str, low_threshold: int, high_threshold: int) -> None:
    """
    Apply Canny edge detection to an image and save the result.

    Parameters:
    image_path (str): Path to the input image file.
    output_path (str): Path to save the edge-detected image.
    low_threshold (int): Lower threshold for hysteresis procedure.
    high_threshold (int): Higher threshold for hysteresis procedure.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Invalid image format.")

        edges = cv2.Canny(image, low_threshold, high_threshold)
        cv2.imwrite(output_path, edges)
        logger.info(f"Edge-detected image saved to {output_path}")
    except ValueError as e:
        logger.error(f"Error applying Canny edge detection: {e}")

def batch_apply_canny_edge_detection(
    input_dir: str, output_dir: str, low_threshold: int, high_threshold: int
) -> None:
    """
    Apply Canny edge detection to a batch of images in the input directory and save the results.

    Parameters:
    input_dir (str): Directory containing the input images.
    output_dir (str): Directory to save the edge-detected images.
    low_threshold (int): Lower threshold for hysteresis procedure.
    high_threshold (int): Higher threshold for hysteresis procedure.
    """
    if not os.path.exists(input_dir):
        raise ValueError("Input directory not found.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_images = [
        os.path.join(input_dir, file)
        for file in os.listdir(input_dir)
        if file.lower().endswith((".jpg", ".png", ".bmp"))
    ]

    for image_path in input_images:
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"canny_{base_name}")
        apply_canny_edge_detection(image_path, output_path, low_threshold, high_threshold)

class ImagePreprocessor:
    """
    Image preprocessor class providing various preprocessing techniques.
    """

    def __init__(self, config: Dict):
        """
        Initialize the image preprocessor with configuration settings.

        Parameters:
        config (Dict): Preprocessor configuration settings.
        """
        self.config = config

    def resize_images(self, input_dir: str, output_dir: str) -> None:
        """
        Resize images in the input directory and save them to the output directory.

        Parameters:
        input_dir (str): Directory containing the input images.
        output_dir (str): Directory to save the resized images.
        """
        resize_dim = self.config.get("resize_dim")
        if resize_dim is None:
            raise ValueError("Resize dimension not specified in configuration.")

        batch_preprocess_images(input_dir, output_dir, resize_dim=resize_dim)

    def convert_to_grayscale(self, input_dir: str, output_dir: str) -> None:
        """
        Convert images in the input directory to grayscale and save them to the output directory.

        Parameters:
        input_dir (str): Directory containing the input images.
        output_dir (str): Directory to save the grayscale images.
        """
        batch_preprocess_images(input_dir, output_dir, grayscale=True)

    def apply_edge_detection(self, input_dir: str, output_dir: str) -> None:
        """
        Apply Canny edge detection to images in the input directory and save the results.

        Parameters:
        input_dir (str): Directory containing the input images.
        output_dir (str): Directory to save the edge-detected images.
        """
        thresholds = self.config.get("canny_thresholds")
        if thresholds is None:
            raise ValueError("Canny thresholds not specified in configuration.")

        low_threshold, high_threshold = thresholds
        batch_apply_canny_edge_detection(input_dir, output_dir, low_threshold, high_threshold)

def main() -> None:
    """
    Main function to preprocess images based on configuration settings.
    """
    config = {
        "input_dir": "input_images",
        "output_dir": "preprocessed_images",
        "resize_dim": (300, 300),
        "grayscale": True,
        "canny_thresholds": (50, 150),
    }

    preprocessor = ImagePreprocessor(config)
    preprocessor.resize_images(config["input_dir"], config["output_dir"])
    preprocessor.convert_to_grayscale(config["output_dir"], config["output_dir"])
    preprocessor.apply_edge_detection(config["output_dir"], config["output_dir"])

if __name__ == "__main__":
    main()