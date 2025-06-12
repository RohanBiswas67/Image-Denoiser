# Image Denoiser

A web application for denoising images using various state-of-the-art algorithms, built with Flask and OpenCV.

## Features

- Drag & drop or browse to upload images
- Choose from multiple denoising algorithms:
  - Non-local Means (OpenCV)
  - Bilateral Filter
  - Gaussian Blur
  - Median Blur
  - Wavelet Denoising
  - Total Variation
  - BM3D (slow but high quality)
  - Non-local Means (Scikit-image)
  - Deep CNN Super-Resolution (requires EDSR_x3.pb model)
  - Anisotropic Diffusion
- Adjustable denoising level
- Side-by-side comparison of original and denoised images

## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Image-Denoiser.git
    cd Image-Denoiser
    ```

2. **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the EDSR_x3.pb model** (for Deep CNN Super-Resolution) and place it in a `models` directory in the project root.  
   You can get it from [OpenCV's GitHub](https://github.com/opencv/opencv_contrib/tree/master/modules/dnn_superres/samples).

5. **Run the application:**
    ```bash
    python app.py
    ```

6. **Open your browser and go to:**  
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Folder Structure
Image-Denoiser/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── models/
│   └── EDSR_x3.pb
├── static/
│   ├── styles.css
│   ├── uploads/
│   └── results/
└── templates/
└── index.html