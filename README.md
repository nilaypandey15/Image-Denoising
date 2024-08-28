
# Image Denoising Using Wavelet Transform in Python

## Introduction
This project implements image denoising techniques using Wavelet Transform in Python. Wavelet Transform is a powerful tool for noise reduction, allowing for the removal of noise while preserving important image features like edges and textures.

## Features
- **Wavelet Transform Denoising**: Apply Wavelet Transform to decompose images into different frequency components, allowing for effective noise reduction.
- **Multiple Wavelet Types**: Support for various wavelet families (e.g., Haar, Daubechies, Symlets).
- **Thresholding Techniques**: Implementation of soft and hard thresholding methods to control noise reduction.
- **Performance Comparison**: Compare the denoised images using metrics like Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).
- **Support for Various Image Formats**: Work with common image formats (JPEG, PNG, BMP, etc.).

## Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/image-denoising-wavelet.git
   cd image-denoising-wavelet
   ```

2. **Install Dependencies**
   Make sure you have Python 3.x installed. Install the required libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Prepare Your Image**
   Place your noisy image in the `images/` directory.

2. **Run the Denoising Script**
   You can denoise an image by running the following command:
   ```bash
   python denoise.py --image images/noisy_image.jpg --wavelet db1 --threshold soft
   ```
   - `--image`: Path to the noisy image.
   - `--wavelet`: Choose the wavelet type (e.g., `haar`, `db1`, `sym2`).
   - `--threshold`: Type of thresholding (`soft` or `hard`).

3. **View Results**
   The denoised image will be saved in the `output/` directory, and metrics will be displayed in the console.

## Examples
Here are some examples of how to use the script:

- **Denoising with Haar Wavelet and Soft Thresholding**:
  ```bash
  python denoise.py --image images/noisy_image.jpg --wavelet haar --threshold soft
  ```

- **Denoising with Daubechies Wavelet and Hard Thresholding**:
  ```bash
  python denoise.py --image images/noisy_image.jpg --wavelet db2 --threshold hard
  ```

## Dependencies
- Python 3.x
- NumPy
- PyWavelets
- OpenCV
- Matplotlib

## Results
You can compare the original noisy image with the denoised image using the provided metrics (PSNR, SSIM) to evaluate the effectiveness of the denoising process.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.

## Acknowledgments
- **PyWavelets**: A Python library for Wavelet Transform.
- **OpenCV**: Used for image processing tasks.
- **Matplotlib**: Used for visualizing the results.

