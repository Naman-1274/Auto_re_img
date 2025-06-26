import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"

import streamlit as st
from PIL import Image, ImageFont, ImageDraw
import io, zipfile
import cv2
import numpy as np
from rembg import remove
from ultralytics import YOLO
import ssl, warnings

warnings.filterwarnings("ignore")
ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(
    page_title="AI Cropper + Brand Generator",
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n-seg.pt")

model = load_yolo_model()

# =================== UTILITIES ===================

def compute_center_of_bbox(bbox):
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

def enhanced_subject_detection(model, img):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = model.predict(img_cv, classes=0, verbose=False)
    for r in results:
        if r.masks is not None:
            masks = r.masks.xy
            if len(masks) > 0:
                largest_mask = max(masks, key=lambda m: cv2.contourArea(m))
                x, y, w, h = cv2.boundingRect(largest_mask.astype(np.int32))
                return (x, y, x + w, y + h)

    bg_removed = remove(img, post_process_mask=True)
    alpha = bg_removed.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        dx = int((bbox[2] - bbox[0]) * 0.05)
        dy = int((bbox[3] - bbox[1]) * 0.05)
        x0 = max(0, bbox[0] - dx)
        y0 = max(0, bbox[1] - dy)
        x1 = min(img.width, bbox[2] + dx)
        y1 = min(img.height, bbox[3] + dy)
        return (x0, y0, x1, y1)
    return None

def smart_resize_no_padding(image, bbox, target_size, zoom=1.2, top_space=0, bottom_space=0):
    img_w, img_h = image.size
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2

    # Adjust for headspace
    cy += int(((bottom_space - top_space) / 2))

    crop_w = int((bbox[2] - bbox[0]) * zoom)
    crop_h = int((bbox[3] - bbox[1]) * zoom)

    aspect_target = target_size[0] / target_size[1]
    aspect_crop = crop_w / crop_h

    if aspect_crop > aspect_target:
        crop_h = int(crop_w / aspect_target)
    else:
        crop_w = int(crop_h * aspect_target)

    left = max(0, cx - crop_w // 2)
    right = min(img_w, cx + crop_w // 2)
    top = max(0, cy - crop_h // 2)
    bottom = min(img_h, cy + crop_h // 2)

    cropped = image.crop((left, top, right, bottom))
    resized = cropped.resize(target_size, Image.LANCZOS)
    return resized

def optimize_image(img, max_size_kb):
    buffer = io.BytesIO()
    quality = 95
    img.save(buffer, "JPEG", quality=quality, optimize=True, progressive=True)
    while (buffer.tell() / 1024) > max_size_kb and quality > 10:
        buffer.seek(0)
        buffer.truncate()
        quality -= 5
        img.save(buffer, "JPEG", quality=quality, optimize=True, progressive=True)
    buffer.seek(0)
    return buffer

def apply_branding(img, logo=None, **kwargs):
    composite = img.convert("RGBA")

    if kwargs.get("add_padding", False):
        pad = kwargs.get("padding", 0)
        color = kwargs.get("padding_color", (255, 255, 255, 0))
        new_w = composite.width + 2 * pad
        new_h = composite.height + 2 * pad
        base = Image.new("RGBA", (new_w, new_h), color)
        base.paste(composite, (pad, pad))
        composite = base

    if logo is not None:
        logo = logo.convert("RGBA")
        logo_w = int((kwargs["logo_scale"] / 100) * composite.width)
        logo_h = int(logo_w * (logo.height / logo.width))
        logo_resized = logo.resize((logo_w, logo_h), Image.LANCZOS)
        x_px = int((kwargs["x_offset"] / 100) * (composite.width - logo_w))
        y_px = int((kwargs["y_offset"] / 100) * (composite.height - logo_h))
        composite.paste(logo_resized, (x_px, y_px), logo_resized)

    if kwargs.get("add_text", False) and kwargs.get("text", ""):
        draw = ImageDraw.Draw(composite)
        try:
            font = ImageFont.truetype("arial.ttf", kwargs["font_size"])
        except:
            font = ImageFont.load_default()
        tx = int((kwargs["text_x"] / 100) * composite.width)
        ty = int((kwargs["text_y"] / 100) * composite.height)
        draw.text(
            (tx, ty), kwargs["text"], fill=kwargs["text_color"],
            font=font, stroke_width=2, stroke_fill="white"
        )

    return composite.convert("RGB")

def preprocess_uploaded_image(img: Image.Image, max_dim: int = 2048) -> Image.Image:
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    return img.convert("RGB")

# =================== UI + APP ===================

if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "processed_results" not in st.session_state:
    st.session_state.processed_results = []

with st.sidebar:
    st.markdown("## 🎛️ Select App Mode")
    mode = st.selectbox("Choose an action:", ["🎯 Smart Cropper + Branding"], index=0)
    st.markdown("---")
    if st.button("🗑️ Clear Uploaded Files"):
        st.session_state.upload_key += 1
        st.session_state.processed_results = []
        st.rerun()

st.title("📸 AI‑Powered Smart Cropper + Branding")
st.info("Use the sidebar to upload and process images.", icon="🛠️")

uploaded_files = st.file_uploader(
    "📸 Upload Image(s) (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.upload_key}"
)

def load_image_from_uploaded(upl):
    return Image.open(upl).convert("RGB")

if uploaded_files:
    st.subheader("🔍 Uploaded Image Preview")
    cols = st.columns(min(4, len(uploaded_files)))
    for idx, upl in enumerate(uploaded_files):
        img = preprocess_uploaded_image(load_image_from_uploaded(upl))
        cols[idx % len(cols)].image(img, use_container_width=True, caption=upl.name)

if mode == "🎯 Smart Cropper + Branding":
    st.sidebar.markdown("## ✂️ Smart Crop Settings")
    with st.sidebar.expander("📐 Output Dimensions"):
        target_width = st.number_input("Width", 512, 4096, 1200, step=100)
        target_height = st.number_input("Height", 512, 4096, 1800, step=100)
        zoom_factor = st.slider("Zoom Level", 0.5, 3.0, 1.2, 0.1)
        st.markdown("---")
        max_size_kb = st.number_input("Max File Size (KB)", 100, 5000, 800, step=50)

    with st.sidebar.expander("🧠 Headspace Settings"):
        use_headspace = st.checkbox("Add Headspace (Top/Bottom)")
        if use_headspace:
            top_space = st.number_input("Top Headspace", 0, 1000, 10)
            bottom_space = st.number_input("Bottom Headspace", 0, 1000, 10)
        else:
            top_space = 0
            bottom_space = 0

    st.sidebar.markdown("## 🎨 Branding Options")
    with st.sidebar.expander("🏷️ Logo Settings"):
        logo_file = st.file_uploader("Upload Logo (PNG)", type=["jpg", "jpeg", "png"])
        logo_scale = st.slider("Logo Size (% of width)", 5, 50, 25)
        x_offset = st.slider("Logo Horizontal Pos (%)", 0, 100, 50)
        y_offset = st.slider("Logo Vertical Pos (%)", 0, 100, 90)

    with st.sidebar.expander("🔤 Text Overlay"):
        add_text = st.checkbox("Add Text")
        if add_text:
            text = st.text_input("Text Content", "Your Brand Message")
            font_size = st.slider("Font Size", 10, 150, 90)
            text_color = st.color_picker("Text Color", "#000000")
            text_x = st.slider("Text Horizontal Pos (%)", 0, 100, 50)
            text_y = st.slider("Text Vertical Pos (%)", 0, 100, 90)
        else:
            text = ""
            font_size = 40
            text_color = "#000000"
            text_x = 5
            text_y = 5
            
    with st.sidebar.expander("🧱 Padding"):
        add_padding = st.checkbox("Add Padding")
        if add_padding:
            padding = st.slider("Padding (px)", 0, 300, 50)
            padding_color = st.color_picker("Padding Color", "#FFFFFF")
        else:
            padding = 0
            padding_color = "#FFFFFF"
            add_padding = False

    if uploaded_files and st.button("🚀 Process Images"):
        results = []
        logo_img = Image.open(logo_file).convert("RGBA") if logo_file else None
        progress = st.progress(0, text="Processing…")
        for i, upl in enumerate(uploaded_files):
            base_img = preprocess_uploaded_image(load_image_from_uploaded(upl))
            if max(base_img.size) > 3000:
                base_img = base_img.resize((base_img.width // 2, base_img.height // 2), Image.LANCZOS)

            bbox = enhanced_subject_detection(model, base_img)
            if not bbox:
                w, h = base_img.size
                bbox = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)

            cropped = smart_resize_no_padding(base_img, bbox, (target_width, target_height),
                                              zoom=zoom_factor, top_space=top_space, bottom_space=bottom_space)

            branded_img = apply_branding(
                cropped, logo_img,
                logo_scale=logo_scale, x_offset=x_offset, y_offset=y_offset,
                add_text=add_text, text=text, font_size=font_size,
                text_color=text_color, text_x=text_x, text_y=text_y,
                add_padding=add_padding, padding=padding, padding_color=padding_color
            )

            buf = optimize_image(branded_img, max_size_kb)
            results.append((upl.name, branded_img, buf))
            progress.progress((i + 1) / len(uploaded_files), text=f"Processed {i+1}/{len(uploaded_files)}")

        progress.empty()
        st.session_state.processed_results = results

    if st.session_state.processed_results:
        st.subheader("🎨 Branded Output Preview")
        preview_cols = st.columns(min(4, len(st.session_state.processed_results)))
        for idx, (fname, img_obj, buff) in enumerate(st.session_state.processed_results):
            with preview_cols[idx % len(preview_cols)]:
                st.image(img_obj, caption=fname, use_container_width=True)
                st.download_button(
                    label="⬇️ Download",
                    data=buff.getvalue(),
                    file_name=f"branded_{fname}",
                    mime="image/jpeg",
                    key=f"download_{idx}"
                )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for fname, _, buff in st.session_state.processed_results:
                zf.writestr(f"branded_{fname}", buff.getvalue())
        zip_buf.seek(0)
        st.download_button(
            "📦 Download All as ZIP",
            data=zip_buf.getvalue(),
            file_name="branded_images.zip",
            mime="application/zip"
        )
