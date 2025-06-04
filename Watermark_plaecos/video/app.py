import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
from scipy.fftpack import dct, idct
import tempfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Helper function to embed invisible watermark
def embed_invisible_watermark(frame, watermark_bits, bit_index):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    Y = np.float32(Y)
    h, w = Y.shape

    block_size = 8
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            if bit_index >= len(watermark_bits):
                break
            block = Y[i:i+block_size, j:j+block_size]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            coeff = dct_block[4][4]
            bit = int(watermark_bits[bit_index])
            dct_block[4][4] = coeff - coeff % 2 + bit
            idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            Y[i:i+block_size, j:j+block_size] = idct_block
            bit_index += 1
        if bit_index >= len(watermark_bits):
            break

    Y = np.clip(Y, 0, 255).astype(np.uint8)
    watermarked_frame = cv2.cvtColor(cv2.merge([Y, Cr, Cb]), cv2.COLOR_YCrCb2BGR)
    return watermarked_frame, bit_index

# Helper function to overlay visible text watermark
def overlay_text_watermark(frame, text):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_x = w - text_size[0] - 10
    text_y = h - 10
    cv2.putText(overlay, text, (text_x, text_y), font, scale, (255, 255, 255), thickness)
    return overlay

# Helper function to overlay logo watermark
def overlay_logo_watermark(frame, logo_path):
    overlay = frame.copy()
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        return frame

    # Resize logo to small size (like YouTube's watermark)
    target_height = int(frame.shape[0] * 0.08)  # ~8% of video height
    aspect_ratio = logo.shape[1] / logo.shape[0]
    target_width = int(target_height * aspect_ratio)
    logo = cv2.resize(logo, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Coordinates to place the logo in bottom-right corner
    h_logo, w_logo = logo.shape[:2]
    h_frame, w_frame = frame.shape[:2]
    x_offset = w_frame - w_logo - 10
    y_offset = h_frame - h_logo - 10

    # Blend logo with alpha channel
    if logo.shape[2] == 4:
        for c in range(0, 3):
            alpha = logo[:, :, 3] / 255.0
            overlay[y_offset:y_offset+h_logo, x_offset:x_offset+w_logo, c] = \
                alpha * logo[:, :, c] + (1 - alpha) * overlay[y_offset:y_offset+h_logo, x_offset:x_offset+w_logo, c]
    else:
        overlay[y_offset:y_offset+h_logo, x_offset:x_offset+w_logo] = logo

    return overlay

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_file = request.files['video']
        watermark_text = request.form.get('watermark_text', '')
        visible_option = request.form.get('visible_option', 'none')

        filename = secure_filename(video_file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(input_path)

        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_filename = f"watermarked_{filename}"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        watermark_bits = ''.join(format(ord(c), '08b') for c in watermark_text)
        bit_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if bit_index < len(watermark_bits):
                frame, bit_index = embed_invisible_watermark(frame, watermark_bits, bit_index)
            if visible_option == 'text':
                frame = overlay_text_watermark(frame, "© YourBrand")
            elif visible_option == 'logo':
                frame = overlay_logo_watermark(frame, os.path.join('static', 'logo.png'))
            out.write(frame)

        cap.release()
        out.release()
        return send_file(output_path, as_attachment=True)

    return render_template('index.html')

# Route to extract hidden watermark
@app.route('/extract', methods=['POST'])
def extract():
    video_file = request.files['video']
    length = int(request.form.get('length', 0))

    filename = secure_filename(video_file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    watermark_bits = ''
    bit_index = 0
    required_bits = length * 8

    while True:
        ret, frame = cap.read()
        if not ret or bit_index >= required_bits:
            break
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        Y = np.float32(ycrcb[:, :, 0])
        h, w = Y.shape
        block_size = 8
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                if bit_index >= required_bits:
                    break
                block = Y[i:i+block_size, j:j+block_size]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                bit = int(dct_block[4][4]) % 2
                watermark_bits += str(bit)
                bit_index += 1
            if bit_index >= required_bits:
                break

    cap.release()

    if len(watermark_bits) < required_bits:
        watermark_text = "❌ Not enough bits found in video"
    else:
        watermark_text = ''.join(chr(int(watermark_bits[i:i+8], 2)) for i in range(0, required_bits, 8))

    # Render result.html with extracted watermark
    return render_template("result.html", watermark_text=watermark_text)


if __name__ == '__main__':
    app.run(debug=True)
