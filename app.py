import os
import io, zipfile
import ssl, warnings
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rembg import remove
from ultralytics import YOLO

warnings.filterwarnings("ignore")
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"

# =================== Streamlit Config ===================
st.set_page_config(
    page_title="AI Cropper + Brand Generator",
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n-seg.pt")

model = load_yolo_model()

# =================== Utility Functions ===================

def preprocess_uploaded_image(img, max_dim=2048):
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    return img.convert("RGB")

def optimize_image(img, max_size_kb):
    buf = io.BytesIO()
    q = 95
    img.save(buf, "JPEG", quality=q, optimize=True, progressive=True)
    while buf.tell()/1024 > max_size_kb and q > 10:
        buf.seek(0); buf.truncate()
        q -= 5
        img.save(buf, "JPEG", quality=q, optimize=True, progressive=True)
    buf.seek(0)
    return buf

def enhanced_subject_detection(model, img):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = model.predict(img_cv, classes=0, verbose=False)
    for r in results:
        if hasattr(r, 'masks') and r.masks is not None:
            masks = r.masks.xy
            if masks:
                largest_mask = max(masks, key=lambda m: cv2.contourArea(m.astype(np.int32)))
                x, y, w, h = cv2.boundingRect(largest_mask.astype(np.int32))
                return (x, y, x + w, y + h)
    bg_removed = remove(img, post_process_mask=True)
    alpha = bg_removed.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        dx = int((bbox[2] - bbox[0]) * 0.05)
        dy = int((bbox[3] - bbox[1]) * 0.05)
        x0, y0 = max(0, bbox[0] - dx), max(0, bbox[1] - dy)
        x1 = min(img.width, bbox[2] + dx)
        y1 = min(img.height, bbox[3] + dy)
        return (x0, y0, x1, y1)
    return None

def smart_resize_preserve_background(image, bbox, target_size, top_space=0, bottom_space=0):
    img_w, img_h = image.size
    target_w, target_h = target_size
    target_ratio = target_w / target_h

    x0, y0, x1, y1 = bbox
    y0, y1 = max(0, y0 - top_space), min(img_h, y1 + bottom_space)
    box_w, box_h = x1 - x0, y1 - y0
    box_cx, box_cy = (x0 + x1) // 2, (y0 + y1) // 2

    if (box_w / box_h) < target_ratio:
        new_box_w = int(box_h * target_ratio)
        new_box_h = box_h
    else:
        new_box_w = box_w
        new_box_h = int(box_w / target_ratio)

    margin_w, margin_h = int(new_box_w * 0.1), int(new_box_h * 0.1)
    new_box_w += margin_w
    new_box_h += margin_h

    left = max(0, box_cx - new_box_w // 2)
    right = min(img_w, box_cx + new_box_w // 2)
    top = max(0, box_cy - new_box_h // 2)
    bottom = min(img_h, box_cy + new_box_h // 2)

    cropped = image.crop((left, top, right, bottom))
    return cropped.resize(target_size, Image.LANCZOS)

def add_black_glow_around_logo(base_img, logo_img, x_px, y_px, blur_radius=8, glow_opacity=100):
    base = base_img.convert("RGBA")
    logo = logo_img.convert("RGBA")
    w, h = logo.size
    alpha = logo.split()[-1]
    blurred = alpha.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    shadow = Image.new('RGBA', (w, h), (0,0,0,0))
    shadow.putalpha(blurred.point(lambda p: p * glow_opacity // 100))
    shadow_layer = Image.new('RGBA', (w, h), (0,0,0,255))
    shadow_layer.putalpha(shadow.split()[-1])
    region = base.crop((x_px, y_px, x_px + w, y_px + h))
    region_np = np.array(region).astype(np.float32)
    sh_np = np.array(shadow_layer).astype(np.float32) / 255
    region_np[..., :3] = region_np[..., :3] * (1 - sh_np[..., 3:]) + region_np[..., :3] * sh_np[..., 3:] * 0.5
    base.paste(Image.fromarray(region_np.clip(0,255).astype(np.uint8)), (x_px, y_px))
    base.paste(logo, (x_px, y_px), logo)
    return base.convert("RGB")

def add_blur_background_under_logo(base_img, logo_img, x_px, y_px, blur_radius=10, mask_margin=5):
    base = base_img.convert("RGBA")
    logo = logo_img.convert("RGBA")
    w, h = logo.size
    alpha = logo.split()[-1]
    mask = alpha.point(lambda p: 255 if p > 0 else 0)
    mask = mask.filter(ImageFilter.MaxFilter(mask_margin*2 + 1))
    region = base.crop((x_px, y_px, x_px + w, y_px + h))
    blurred = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blended = Image.composite(blurred, region, mask)
    base.paste(blended, (x_px, y_px))
    return base.convert("RGB")

def merge_logos_horizontally(logo1, logo2, padding=20, separator_text="×", separator_font_size=40):
    try:
        font = ImageFont.truetype("arial.ttf", separator_font_size)
    except:
        font = ImageFont.load_default()

    # Measure text size using getbbox (Pillow ≥ 10.0.0)
    bbox = font.getbbox(separator_text)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    h = max(logo1.height, logo2.height, text_height)
    total_width = logo1.width + logo2.width + text_width + 2 * padding
    merged = Image.new("RGBA", (total_width, h), (0, 0, 0, 0))

    y1 = (h - logo1.height) // 2
    y2 = (h - logo2.height) // 2
    text_y = (h - text_height) // 2

    merged.paste(logo1, (0, y1), logo1)

    draw = ImageDraw.Draw(merged)
    text_x = logo1.width + padding
    draw.text((text_x, text_y), separator_text, font=font, fill=(255, 255, 255, 255))

    merged.paste(logo2, (text_x + text_width + padding, y2), logo2)

    return merged




# =================== UI State ===================
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "stored_files" not in st.session_state:
    st.session_state.stored_files = []
if "results" not in st.session_state:
    st.session_state.results = []

with st.sidebar:
    st.markdown("## 🎛️ Select Mode")
    mode = st.selectbox("Action:", ["🎯 Smart Cropper + Branding"])
    if st.button("🗑️ Clear Files"):
        st.session_state.upload_key += 1
        st.session_state.stored_files = []
        st.session_state.results = []
        st.rerun()

st.title("📸 AI-Powered Smart Cropper + Branding")
st.info("Upload images and customize settings in the sidebar.")

files = st.file_uploader(
    "Upload Images",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True,
    key=f"up_{st.session_state.upload_key}"
)
if files:
    st.session_state.stored_files = files

if st.session_state.stored_files:
    st.subheader("Preview")
    cols = st.columns(min(4, len(st.session_state.stored_files)))
    for i, f in enumerate(st.session_state.stored_files):
        cols[i % len(cols)].image(
            preprocess_uploaded_image(Image.open(f)),
            caption=f.name
        )

if mode == "🎯 Smart Cropper + Branding":
    with st.sidebar.expander("📐 Output Dimensions"):
        tw = st.number_input("Width", 512, 4096, 1200, 100)
        th = st.number_input("Height", 512, 4096, 1800, 100)
        max_kb = st.number_input("Max File Size (KB)", 100, 5000, 800, 50)

    with st.sidebar.expander("🧠 Headspace"):
        use_space = st.checkbox("Add Head/Foot Space")
        ts = st.number_input("Top Space", 0,1000,10) if use_space else 0
        bs = st.number_input("Bottom Space", 0,1000,10) if use_space else 0

    with st.sidebar.expander("🎨 Logo Settings"):
        collab_mode = st.checkbox("Collaboration Post")
        if collab_mode:
            logo_file1 = st.file_uploader("Upload First Logo", type=["png","jpg","jpeg"], key="logo1_up")
            logo_file2 = st.file_uploader("Upload Second Logo", type=["png","jpg","jpeg"], key="logo2_up")
            merge_padding = st.slider("Collab Logo Padding (px)", 0, 100, 20)

    # 👇 New: Custom symbol and font size
            separator_text = st.text_input("Symbol Between Logos", value="×")
            separator_font_size = st.slider("Symbol Size", 10, 200, 40)

        else:
            logo_file = st.file_uploader("Upload Logo (PNG, JPG, JPEG)", type=["png","jpg","jpeg"], key="logo_up")

        scale = st.slider("Logo % of Width", 5, 50, 30)
        x_off = st.slider("Logo Horiz Pos (%)", 0, 100, 50)
        y_off = st.slider("Logo Vert Pos (%)", 0, 100, 90)
        shadow = st.checkbox("Enable Logo Shadow", value=True)
        sr = st.slider("Shadow Blur", 2, 50, 25) if shadow else 0
        so = st.slider("Shadow Opacity %", 0, 100, 30) if shadow else 0
        bgblur = st.checkbox("Enable Background Blur Under Logo")
        br = st.slider("Blur Radius", 1, 50, 10) if bgblur else 0
        mm = st.slider("Mask Margin px", 1, 50, 5) if bgblur else 0

    with st.sidebar.expander("🖋️ Text Overlay"):
        overlay_text = st.text_input("Overlay Text")
        text_size = st.slider("Font Size", 10, 200, 40)
        text_color = st.color_picker("Text Color", "#FFFFFF")
        font_family = st.selectbox(
            "Font Family",
            [
                "Arial",
                "Helvetica",
                "Times New Roman",
                "Chronicle Display",
                "Facundo",
                "Felidae",
                "Edwardian Script ITC"
            ]
        )
        text_x_pct = st.slider("Text Horiz Pos (%)", 0, 100, 50)
        text_y_pct = st.slider("Text Vert Pos (%)", 0, 100, 95)

    if st.button("🚀 Process Images") and st.session_state.stored_files:
        res = []

        # ==== Load Logos ====
        if collab_mode and logo_file1 and logo_file2:
            tmp1 = Image.open(logo_file1).convert("RGBA")
            tmp2 = Image.open(logo_file2).convert("RGBA")
            datas1 = tmp1.getdata()
            newData1 = [(r,g,b,0) if r>240 and g>240 and b>240 else (r,g,b,a) for r,g,b,a in datas1]
            tmp1.putdata(newData1)
            datas2 = tmp2.getdata()
            newData2 = [(r,g,b,0) if r>240 and g>240 and b>240 else (r,g,b,a) for r,g,b,a in datas2]
            tmp2.putdata(newData2)
            logo_img = merge_logos_horizontally(
    tmp1, tmp2,
    padding=merge_padding,
    separator_text=separator_text,
    separator_font_size=separator_font_size
)

        elif not collab_mode and logo_file:
            tmp = Image.open(logo_file).convert("RGBA")
            datas = tmp.getdata()
            newData = [(r,g,b,0) if r>240 and g>240 and b>240 else (r,g,b,a) for r,g,b,a in datas]
            tmp.putdata(newData)
            logo_img = tmp
        else:
            logo_img = None
        prog = st.progress(0)
        for idx, f in enumerate(st.session_state.stored_files):
            img = preprocess_uploaded_image(Image.open(f))
            if max(img.size) > 3000:
                img = img.resize((img.width//2, img.height//2), Image.LANCZOS)

            bb = enhanced_subject_detection(model, img) or (
                img.width//4, img.height//4,
                3*img.width//4, 3*img.height//4
            )
            base = smart_resize_preserve_background(img, bb, (tw, th), ts, bs).convert("RGBA")

            # ---- Add Logo ----
            if logo_img:
                lw = int(scale/100 * base.width)
                lh = int(lw / logo_img.width * logo_img.height)
                logo_res = logo_img.resize((lw, lh), Image.LANCZOS)
                x_px = int((x_off/100) * (base.width - lw))
                y_px = int((y_off/100) * (base.height - lh))

                if bgblur:
                    base = add_blur_background_under_logo(base, logo_res, x_px, y_px, br, mm).convert("RGBA")
                if shadow:
                    base = add_black_glow_around_logo(base, logo_res, x_px, y_px, sr, so).convert("RGBA")
                else:
                    base.paste(logo_res, (x_px, y_px), logo_res)

            # ---- Add Text Overlay ----
            if overlay_text:
                draw = ImageDraw.Draw(base)
                font_folder = "fonts"
                font_paths = {
                    "Arial": os.path.join(font_folder, "arial.ttf"),
                    "Helvetica": os.path.join(font_folder, "helvetica.ttf"),
                    "Times New Roman": os.path.join(font_folder, "times.ttf"),
                    "Chronicle Display": os.path.join(font_folder, "Chronicle Display Black.ttf"),
                    "Facundo": os.path.join(font_folder, "Facundo.ttf"),
                    "Felidae": os.path.join(font_folder, "Felidae.ttf"),
                    "Edwardian Script ITC": os.path.join(font_folder, "Edwardian.ttf")
                }
                font_file = font_paths.get(font_family, "arial.ttf")
                try:
                    font = ImageFont.truetype(font_file, text_size)
                except:
                    font = ImageFont.load_default()

                bbox = draw.textbbox((0, 0), overlay_text, font=font)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                x_text = int((text_x_pct/100) * (base.width - w))
                y_text = int((text_y_pct/100) * (base.height - h))
                draw.text((x_text, y_text), overlay_text, font=font, fill=text_color)

            final = base.convert("RGB")
            buf = optimize_image(final, max_kb)
            res.append((f.name, final, buf))
            prog.progress((idx+1) / len(st.session_state.stored_files))

        st.session_state.results = res

    # =================== Results ===================
    if st.session_state.results:
        st.subheader("Results")
        cols = st.columns(min(4, len(st.session_state.results)))
        for i, (name, img, buf) in enumerate(st.session_state.results):
            with cols[i % len(cols)]:
                st.image(img, caption=name)
                st.download_button(
                    "Download",
                    data=buf.getvalue(),
                    file_name=f"branded_{name}",
                    mime="image/jpeg",
                    key=f"dl_{i}"
                )
        z = io.BytesIO()
        with zipfile.ZipFile(z, "w") as zf:
            for name, _, buf in st.session_state.results:
                zf.writestr(f"branded_{name}", buf.getvalue())
        z.seek(0)
        st.download_button(
            "Download All ZIP",
            data=z.getvalue(),
            file_name="branded_images.zip",
            mime="application/zip"
        )