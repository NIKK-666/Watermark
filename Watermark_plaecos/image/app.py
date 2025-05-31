from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from scipy.fftpack import dct, idct
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Invisible DCT Watermark Embedding ---
def embed_dct_watermark(image_path, text):
    img = cv2.imread(image_path)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = np.float32(y)
    h, w = y.shape

    watermark = ''.join(format(ord(c), '08b') for c in text)
    wm_idx = 0
    block_size = 8
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if wm_idx >= len(watermark): break
            block = y[i:i+8, j:j+8]
            if block.shape != (8, 8): continue
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            coeff = dct_block[4][4]
            dct_block[4][4] = coeff - coeff % 2 + int(watermark[wm_idx])
            wm_idx += 1
            idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            y[i:i+8, j:j+8] = idct_block
    y = np.clip(y, 0, 255).astype(np.uint8)
    watermarked = cv2.merge([y, cr, cb])
    return cv2.cvtColor(watermarked, cv2.COLOR_YCrCb2BGR)

# --- Visible Watermark Overlay ---
def add_visible_logo(image, logo_path):
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        raise ValueError("❌ Logo image not found or invalid path!")

    # Resize logo
    logo_h, logo_w = logo.shape[:2]
    scale = image.shape[1] // 5
    logo = cv2.resize(logo, (scale, int(scale * logo_h / logo_w)))

    y_offset = image.shape[0] - logo.shape[0] - 10
    x_offset = image.shape[1] - logo.shape[1] - 10

    # If logo has alpha channel
    if logo.shape[2] == 4:
        for c in range(3):  # BGR channels
            image[y_offset:y_offset+logo.shape[0], x_offset:x_offset+logo.shape[1], c] = (
                logo[:, :, c] * (logo[:, :, 3] / 255.0) +
                image[y_offset:y_offset+logo.shape[0], x_offset:x_offset+logo.shape[1], c] * (1.0 - logo[:, :, 3] / 255.0)
            )
    else:
        # No alpha: blend with 30% transparency
        overlay = image.copy()
        overlay[y_offset:y_offset+logo.shape[0], x_offset:x_offset+logo.shape[1]] = logo
        image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)

    return image


# --- Extract Hidden Watermark ---
def extract_dct_watermark(image_path, length):
    img = cv2.imread(image_path)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, _, _ = cv2.split(ycrcb)
    y = np.float32(y)
    watermark_bits = ''
    extracted = 0
    total_bits = length * 8
    for i in range(0, y.shape[0], 8):
        for j in range(0, y.shape[1], 8):
            if extracted >= total_bits: break
            block = y[i:i+8, j:j+8]
            if block.shape != (8, 8): continue
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            watermark_bits += str(int(dct_block[4][4]) % 2)
            extracted += 1
    chars = [chr(int(watermark_bits[i:i+8], 2)) for i in range(0, len(watermark_bits), 8)]
    return ''.join(chars)

# --- Flask Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        watermark_type = request.form["type"]
        watermark_text = request.form.get("watermark_text", "")
        img_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(img_path)
        img = cv2.imread(img_path)

        if watermark_type == "invisible":
            result = embed_dct_watermark(img_path, watermark_text)
        elif watermark_type == "visible":
            result = add_visible_logo(img, os.path.join(STATIC_FOLDER, "watermark_logo.png"))
        elif watermark_type == "both":
            result = embed_dct_watermark(img_path, watermark_text)
            result = add_visible_logo(result, os.path.join(STATIC_FOLDER, "watermark_logo.png"))

        output_path = os.path.join(STATIC_FOLDER, "watermarked.jpg")
        cv2.imwrite(output_path, result)
        return render_template("index.html", result_img="static/watermarked.jpg")

    return render_template("index.html", result_img=None)

@app.route("/extract", methods=["POST"])
def extract():
    image = request.files["extract_image"]
    length = int(request.form["length"])
    img_path = os.path.join(UPLOAD_FOLDER, "extract.jpg")
    image.save(img_path)
    extracted = extract_dct_watermark(img_path, length)
    return render_template("index.html", extracted=extracted)

if __name__ == "__main__":
    app.run(debug=True)
