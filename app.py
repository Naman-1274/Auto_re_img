import os
from PIL import ImageFilter
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

def smart_resize_preserve_background(image, bbox, target_size, top_space=0, bottom_space=0):
    img_w, img_h = image.size
    target_w, target_h = target_size
    target_ratio = target_w / target_h

    x0, y0, x1, y1 = bbox
    y0 = max(0, y0 - top_space)
    y1 = min(img_h, y1 + bottom_space)

    box_w = x1 - x0
    box_h = y1 - y0
    box_cx = (x0 + x1) // 2
    box_cy = (y0 + y1) // 2

    new_box_w = box_w
    new_box_h = box_h

    if (box_w / box_h) < target_ratio:
        new_box_w = int(box_h * target_ratio)
    else:
        new_box_h = int(box_w / target_ratio)

    margin_w = int(new_box_w * 0.1)
    margin_h = int(new_box_h * 0.1)
    new_box_w += margin_w
    new_box_h += margin_h

    left = max(0, box_cx - new_box_w // 2)
    right = min(img_w, box_cx + new_box_w // 2)
    top = max(0, box_cy - new_box_h // 2)
    bottom = min(img_h, box_cy + new_box_h // 2)

    expanded_crop = image.crop((left, top, right, bottom))
    final = expanded_crop.resize(target_size, Image.LANCZOS)
    return final

def add_black_glow_around_logo(base_img, logo_img, x_px, y_px, blur_radius=8, glow_opacity=100):
    """
    Adds a multiply-blended black shadow (glow) under the transparent edges of the logo.
    The shadow automatically darkens the underlying image, blending naturally with clothing.
    """
    base_img = base_img.convert("RGBA")
    logo_img = logo_img.convert("RGBA")
    logo_w, logo_h = logo_img.size

    # 1. Extract alpha channel and blur it for soft edges
    alpha = logo_img.getchannel('A')
    blurred_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 2. Make a black shadow image using blurred alpha
    shadow = Image.new('RGBA', (logo_w, logo_h), (0, 0, 0, 0))
    shadow.putalpha(blurred_alpha.point(lambda p: p * glow_opacity // 100))

    # 3. Prepare black layer for multiply
    black_layer = Image.new('RGBA', (logo_w, logo_h), (0, 0, 0, 255))
    black_layer.putalpha(shadow.getchannel('A'))

    # 4. Composite multiply blend onto base region
    region_box = (x_px, y_px, x_px + logo_w, y_px + logo_h)
    region = base_img.crop(region_box)

    region_np = np.array(region).astype(np.float32)
    black_layer_np = np.array(black_layer).astype(np.float32) / 255

    # Multiply blend only where shadow alpha exists
    region_np[..., :3] = region_np[..., :3] * (1 - black_layer_np[..., 3:]) + (region_np[..., :3] * black_layer_np[..., 3:] * 0.5)
    result_region = Image.fromarray(np.clip(region_np, 0, 255).astype(np.uint8))

    # Paste the blended region back
    base_img.paste(result_region, region_box)

    # 5. Finally paste the actual logo on top
    base_img.paste(logo_img, region_box, logo_img)

    return base_img.convert("RGB")




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
if "stored_uploaded_files" not in st.session_state:
    st.session_state.stored_uploaded_files = []

with st.sidebar:
    st.markdown("## 🎛️ Select App Mode")
    mode = st.selectbox("Choose an action:", ["🎯 Smart Cropper + Branding"], index=0)
    st.markdown("---")
    if st.button("🗑️ Clear Uploaded Files"):
        st.session_state.upload_key += 1
        st.session_state.processed_results = []
        st.session_state.stored_uploaded_files = []
        st.rerun()

st.title("📸 AI‑Powered Smart Cropper + Branding")
st.info("Use the sidebar to upload and process images.", icon="🛠️")

uploaded_files = st.file_uploader(
    "📸 Upload Image(s) (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.upload_key}"
)

if uploaded_files:
    st.session_state.stored_uploaded_files = uploaded_files

# Always use stored list so UI doesn't reset
if st.session_state.stored_uploaded_files:
    st.subheader("🔍 Uploaded Image Preview")
    cols = st.columns(min(4, len(st.session_state.stored_uploaded_files)))
    for idx, upl in enumerate(st.session_state.stored_uploaded_files):
        img = preprocess_uploaded_image(Image.open(upl))
        cols[idx % len(cols)].image(img, use_container_width=True, caption=upl.name)


if mode == "🎯 Smart Cropper + Branding":
    st.sidebar.markdown("## ✂️ Smart Crop Settings")
    with st.sidebar.expander("📐 Output Dimensions"):
        target_width = st.number_input("Width", 512, 4096, 1200, step=100)
        target_height = st.number_input("Height", 512, 4096, 1800, step=100)
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
        logo_scale = st.slider("Logo Size (% of width)", 5, 50, 30)
        x_offset = st.slider("Logo Horizontal Pos (%)", 0, 100, 50)
        y_offset = st.slider("Logo Vertical Pos (%)", 0, 100, 90)
        st.markdown("---")
        enable_edge_glow = st.checkbox("Enable Black Edge Glow", value=True)
        enable_edge_glow = st.checkbox("Enable Logo Shadow (Multiply Blend)", value=True)
        if enable_edge_glow:
            glow_radius = st.slider("Shadow Blur Radius", 2, 50, 25)
            glow_opacity = st.slider("Shadow Opacity (%)", 0, 100, 30)
        else:
            glow_radius = 0
            glow_opacity = 0



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

    if st.session_state.stored_uploaded_files and st.button("🚀 Process Images"):
        results = []
        logo_img = Image.open(logo_file).convert("RGBA") if logo_file else None
        progress = st.progress(0, text="Processing…")
        for i, upl in enumerate(st.session_state.stored_uploaded_files):
            base_img = preprocess_uploaded_image(Image.open(upl))
            if max(base_img.size) > 3000:
                base_img = base_img.resize((base_img.width // 2, base_img.height // 2), Image.LANCZOS)

            bbox = enhanced_subject_detection(model, base_img)
            if not bbox:
                w, h = base_img.size
                bbox = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)

            cropped = smart_resize_preserve_background(
                        base_img, bbox, (target_width, target_height), top_space, bottom_space
                    )

            composite = cropped.convert("RGBA")
            if logo_img is not None:
                logo_resized_w = int((logo_scale / 100) * composite.width)
                logo_resized_h = int(logo_resized_w * (logo_img.height / logo_img.width))
                logo_resized = logo_img.resize((logo_resized_w, logo_resized_h), Image.LANCZOS)
                x_px = int((x_offset / 100) * (composite.width - logo_resized_w))
                y_px = int((y_offset / 100) * (composite.height - logo_resized_h))
                if enable_edge_glow and glow_radius > 0:
                    composite = add_black_glow_around_logo(
                        composite,
                        logo_resized,
                        x_px,
                        y_px,
                        blur_radius=glow_radius,
                        glow_opacity=glow_opacity
                    )
                else:
                    composite.paste(logo_resized, (x_px, y_px), logo_resized)


            if add_text and text:
                draw = ImageDraw.Draw(composite)
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                tx = int((text_x / 100) * composite.width)
                ty = int((text_y / 100) * composite.height)
                draw.text(
                    (tx, ty), text, fill=text_color,
                    font=font, stroke_width=2, stroke_fill="white"
                )

            final_img = composite.convert("RGB")
            if add_padding:
                new_w = final_img.width + 2 * padding
                new_h = final_img.height + 2 * padding
                base = Image.new("RGB", (new_w, new_h), padding_color)
                base.paste(final_img, (padding, padding))
                final_img = base

            buf = optimize_image(final_img, max_size_kb)
            results.append((upl.name, final_img, buf))
            progress.progress((i + 1) / len(st.session_state.stored_uploaded_files), text=f"Processed {i+1}/{len(st.session_state.stored_uploaded_files)}")

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
