import os
from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
from skimage.restoration import denoise_wavelet, denoise_tv_chambolle, denoise_nl_means
from bm3d import bm3d_rgb
from skimage.restoration import denoise_tv_bregman
import cv2.dnn_superres

app = Flask(__name__, static_url_path='/static')


UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

MODEL_FOLDER = '/'
os.makedirs(MODEL_FOLDER, exist_ok=True)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

import signal
from contextlib import contextmanager
import time

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


    

# Function to denoise the image
@app.route('/denoise', methods=['POST'])
def denoise_image(img, algorithm, denoising_level=10):
    # Example parameter defaults
    h = denoising_level
    hColor = denoising_level
    templateWindowSize = 7
    searchWindowSize = 21
    d = 9
    sigmaColor = 75
    sigmaSpace = 75

    if algorithm == 'nlm':
        denoised = cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)
    elif algorithm == 'bilateral':
        denoised = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
    elif algorithm == 'gaussian':
        denoised = cv2.GaussianBlur(img, (5, 5), 0)
    elif algorithm == 'median':
        denoised = cv2.medianBlur(img, 5)
    elif algorithm == 'wavelet':
        denoised = denoise_wavelet(img/255.0, channel_axis=-1)
        denoised = (denoised * 255).astype(np.uint8)
    elif algorithm == 'tv':
        denoised = denoise_tv_chambolle(img/255.0, channel_axis=-1)
        denoised = (denoised * 255).astype(np.uint8)
    elif algorithm == 'nl_means_skimage':
        patch_kw = dict(patch_size=5, patch_distance=6, channel_axis=-1)
        denoised = denoise_nl_means(img/255.0, h=1.15 * np.std(img/255.0), fast_mode=True, **patch_kw)
        denoised = (denoised * 255).astype(np.uint8)
    
    elif algorithm == 'anisotropic':
        denoised = denoise_tv_bregman(img/255.0, weight=0.1, eps=1e-3, channel_axis=-1)
        denoised = (denoised * 255).astype(np.uint8)

    elif algorithm == 'deep_cnn':
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        model_path = os.path.join(MODEL_FOLDER, "EDSR_x3.pb")
        if not os.path.exists(model_path):
            return img  # Return original image if model not found
        sr.readModel(model_path)
        sr.setModel("edsr", 3)
        try:
            denoised = sr.upsample(img)
            denoised = cv2.resize(denoised, (img.shape[1], img.shape[0]))
        except Exception:
            denoised = img

    elif algorithm == 'bm3d':
        try:
            with time_limit(30):  # 30 seconds timeout
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                denoised_rgb = bm3d_rgb(img_rgb, sigma_psd=30/255)
                denoised = (denoised_rgb * 255).astype(np.uint8)
                denoised = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
        except TimeoutException:
            denoised = img  # Return original image if timeout
        except Exception:
            denoised = img

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return '<div class="alert alert-danger">No file uploaded</div>'
    file = request.files['image']
    if file.filename == '':
        return '<div class="alert alert-danger">No file selected</div>'
    if file:
        try:
            unique_filename = f"{time.time()}_{file.filename}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(image_path)
            denoising_level = int(request.form.get('denoising_level', 10))
            algorithm = request.form.get('algorithm', 'nlm')
            image = cv2.imread(image_path)
            if image is None:
                return '<div class="alert alert-danger">Invalid image file or unsupported format.</div>'
            denoised_image = denoise_image(image, algorithm, denoising_level)
            if denoised_image is None or not isinstance(denoised_image, np.ndarray) or denoised_image.size == 0:
                return '<div class="alert alert-danger">Denoising failed or produced an empty image.</div>'
            result_path = os.path.join(app.config['RESULT_FOLDER'], unique_filename)
            cv2.imwrite(result_path, denoised_image)
            return f'<img src="/static/results/{unique_filename}" class="img-fluid" alt="Denoised Image">'
        except Exception as e:
            return f'<div class="alert alert-danger">Error processing image: {str(e)}</div>'
    return '<div class="alert alert-danger">Error uploading file</div>'

@app.route('/results/<filename>')
def uploaded_file(filename):
    """
    Serve the denoised image from the results folder.

    :param filename: Name of the denoised image file.
    :return: Denoised image file.
    """
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/')
def index():
    """
    Render the main page.
    """
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
