# 🔐 DCT Watermarking Web App (Invisible + Visible Watermarks)

A Flask-based web application that allows users to:

✅ Embed **invisible watermarks** into color images using **DCT (Discrete Cosine Transform)**  
✅ Overlay **visible logo watermarks**  
✅ Extract the hidden invisible watermark from the watermarked image  
✅ Export high-quality watermarked images matching original resolution

---

## 🚀 Features

- 🖼️ Upload any JPG/PNG color image from your device
- 🧊 DCT-based invisible text watermarking (stored in luminance channel)
- 👁️ Visible logo watermark overlay (supports transparency)
- 🔓 Extract hidden watermark from images
- 📦 Download final watermarked image in original quality
- 💡 Supports both watermark types individually or combined

---

## 📸 Example

| Feature                  | Example Description                    |
|--------------------------|----------------------------------------|
| 🔍 Invisible Watermark   | Encoded in Y channel using DCT         |
| 🖼️ Visible Watermark     | Logo overlay in bottom-right corner    |
| 🎯 Dual Mode             | Apply both in one step                 |

---

## 🛠️ Tech Stack

- Python 3.8+
- Flask
- OpenCV (cv2)
- SciPy (for DCT)
- HTML/CSS (Jinja2 templates)
- Bootstrap (for responsive UI)

---

## 🧪 Installation

```bash
git clone https://github.com/yourusername/dct-watermark-webapp.git
cd dct-watermark-webapp
pip install -r requirements.txt
