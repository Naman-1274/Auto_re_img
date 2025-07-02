import os
from PIL import Image, ImageFont, ImageDraw, ImageFilter
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"

import streamlit as st
from PIL import Image
import io, zipfile
import cv2
import numpy as np
from rembg import remove
from ultralytics import YOLO
import ssl, warnings

warnings.filterwarnings("ignore")
ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(
    page_title="Snipster - AI Cropper Pro",
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="collapsed"
)

# Modern Dark Theme CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main-container {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .app-header {
        text-align: center;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
    }
    
    .upload-zone {
        background: linear-gradient(45deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
        margin: 2rem 0;
    }
    
    .upload-zone:hover {
        background: linear-gradient(45deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        border-color: #764ba2;
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
    }
    
    .glass-card {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        background: rgba(255,255,255,0.12);
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    .preview-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .preview-card {
        background: rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }
    
    .preview-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.3);
        background: rgba(255,255,255,0.12);
    }
    
    .preview-card img {
        border-radius: 8px;
        max-width: 100%;
        height: auto;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .preview-label {
        margin-top: 0.8rem;
        font-size: 0.9rem;
        opacity: 0.9;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        border: none !important;
        border-radius: 25px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    .control-section {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 3px solid #667eea;
    }
    
    .stProgress .st-bo {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        margin: 0.5rem 0;
    }
    
    .stSelectbox > div > div,
    .stNumberInput > div > div,
    .stTextInput > div > div,
    .stSlider > div > div,
    .stCheckbox > div {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    .stMarkdown, .stText, p, div, span, label {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n-seg.pt")

# Initialize session state
def init_session_state():
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0
    if "stored_files" not in st.session_state:
        st.session_state.stored_files = []
    if "results" not in st.session_state:
        st.session_state.results = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

init_session_state()

# Utility functions (keeping original logic)
def compute_center_of_bbox(bbox):
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

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

def add_effects_to_image(base, logo_img, x_px, y_px, shadow=True, blur_bg=False, shadow_radius=25, shadow_opacity=30, blur_radius=10, mask_margin=5):
    base = base.convert("RGBA")
    
    if blur_bg:
        w, h = logo_img.size
        alpha = logo_img.split()[-1]
        mask = alpha.point(lambda p: 255 if p > 0 else 0)
        mask = mask.filter(ImageFilter.MaxFilter(mask_margin*2 + 1))
        region = base.crop((x_px, y_px, x_px + w, y_px + h))
        blurred = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blended = Image.composite(blurred, region, mask)
        base.paste(blended, (x_px, y_px))
    
    if shadow:
        w, h = logo_img.size
        alpha = logo_img.split()[-1]
        blurred = alpha.filter(ImageFilter.GaussianBlur(radius=shadow_radius))
        shadow = Image.new('RGBA', (w, h), (0,0,0,0))
        shadow.putalpha(blurred.point(lambda p: p * shadow_opacity // 100))
        shadow_layer = Image.new('RGBA', (w, h), (0,0,0,255))
        shadow_layer.putalpha(shadow.split()[-1])
        region = base.crop((x_px, y_px, x_px + w, y_px + h))
        region_np = np.array(region).astype(np.float32)
        sh_np = np.array(shadow_layer).astype(np.float32) / 255
        region_np[..., :3] = region_np[..., :3] * (1 - sh_np[..., 3:]) + region_np[..., :3] * sh_np[..., 3:] * 0.5
        base.paste(Image.fromarray(region_np.clip(0,255).astype(np.uint8)), (x_px, y_px))
    
    base.paste(logo_img, (x_px, y_px), logo_img)
    return base.convert("RGB")

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

def preprocess_uploaded_image(img, max_dim=2048):
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    return img.convert("RGB")

# Main App Layout
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown('<h1 class="app-header">🎯 AI Cropper Pro</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; opacity: 0.8; margin-bottom: 2rem;">Transform your images with intelligent cropping and professional branding</p>', unsafe_allow_html=True)

# Upload Section
st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
st.markdown('### 📁 Drop Your Images Here')
files = st.file_uploader(
    "Choose images to process",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True,
    key=f"up_{st.session_state.upload_key}",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

if files:
    st.session_state.stored_files = files
    st.success(f"✅ {len(files)} images uploaded successfully!")

# Show Image Previews
if st.session_state.stored_files:
    st.markdown("### 🖼️ Uploaded Images")
    st.markdown('<div class="preview-grid">', unsafe_allow_html=True)
    
    # Create preview cards using columns
    cols = st.columns(min(4, len(st.session_state.stored_files)))
    for i, f in enumerate(st.session_state.stored_files):
        with cols[i % len(cols)]:
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            img = preprocess_uploaded_image(Image.open(f))
            st.image(img, use_container_width=True)
            st.markdown(f'<div class="preview-label">{f.name}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main Interface
if st.session_state.stored_files:
    # Controls in expandable sections
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Quick Settings
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Quick Settings")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            tw = st.number_input("Width", 512, 4096, 1200, 100)
            th = st.number_input("Height", 512, 4096, 1800, 100)
        with col_b:
            max_kb = st.number_input("Max Size (KB)", 100, 5000, 800, 50)
            add_space = st.checkbox("Add Spacing", value=True)
        with col_c:
            if add_space:
                ts = st.number_input("Top Space", 0, 200, 50, 10)
                bs = st.number_input("Bottom Space", 0, 200, 50, 10)
            else:
                ts = bs = 0
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("🎨 Logo & Branding", expanded=False):
        logo_file = st.file_uploader("Upload Logo", type=["png","jpg","jpeg"], key="logo")
        
        if logo_file:
            logo_preview = Image.open(logo_file)
            st.image(logo_preview, width=150)
            
            scale = st.slider("Logo Size (%)", 5, 50, 25)
            x_off = st.slider("X Position (%)", 0, 100, 50)
            y_off = st.slider("Y Position (%)", 0, 100, 85)
            
            # Effects
            shadow = st.checkbox("Shadow Effect", value=True)
            blur_bg = st.checkbox("Blur Background")


    # Text Overlay
    with st.expander("✏️ Text Overlay", expanded=False):
        overlay_text = st.text_input("Text to overlay")
        if overlay_text:
            col_x, col_y = st.columns(2)
            with col_x:
                text_size = st.slider("Font Size", 20, 100, 40)
                text_color = st.color_picker("Color", "#FFFFFF")
            with col_y:
                text_x = st.slider("Text X (%)", 0, 100, 50)
                text_y = st.slider("Text Y (%)", 0, 100, 90)

    # Process Button
    st.markdown("---")
    process_clicked = st.button("🚀 Process All Images", use_container_width=True, type="primary")

    # Processing Logic
    if process_clicked and not st.session_state.processing:
        st.session_state.processing = True
        
        # Load model
        model = load_yolo_model()
        
        # Prepare logo
        logo_img = None
        if 'logo_file' in locals() and logo_file:
            tmp = Image.open(logo_file).convert("RGBA")
            # Remove white background
            datas = tmp.getdata()
            newData = [(r,g,b,0) if r>240 and g>240 and b>240 else (r,g,b,a) for r,g,b,a in datas]
            tmp.putdata(newData)
            logo_img = tmp

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, file in enumerate(st.session_state.stored_files):
            status_text.text(f"🔄 Processing {file.name}... ({idx+1}/{len(st.session_state.stored_files)})")
            
            # Load and process image
            img = preprocess_uploaded_image(Image.open(file))
            bbox = enhanced_subject_detection(model, img) or (img.width//4, img.height//4, 3*img.width//4, 3*img.height//4)
            
            # Smart crop
            base = smart_resize_preserve_background(img, bbox, (tw, th), ts, bs).convert("RGBA")

            # Apply logo
            if logo_img:
                lw = int(scale/100 * base.width)
                lh = int(lw / logo_img.width * logo_img.height)
                logo_res = logo_img.resize((lw, lh), Image.LANCZOS)
                x_px = int((x_off/100) * (base.width - lw))
                y_px = int((y_off/100) * (base.height - lh))
                
                base = add_effects_to_image(base, logo_res, x_px, y_px, shadow, blur_bg if 'blur_bg' in locals() else False)

            # Apply text
            if 'overlay_text' in locals() and overlay_text:
                draw = ImageDraw.Draw(base)
                try:
                    font = ImageFont.truetype("arial.ttf", text_size)
                except:
                    font = ImageFont.load_default()
                bbox_text = draw.textbbox((0, 0), overlay_text, font=font)
                w, h = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
                x_text = int((text_x/100) * (base.width - w))
                y_text = int((text_y/100) * (base.height - h))
                draw.text((x_text, y_text), overlay_text, font=font, fill=text_color)

            # Optimize and save
            final = base.convert("RGB")
            buf = optimize_image(final, max_kb)
            results.append((file.name, final, buf))
            
            progress_bar.progress((idx+1) / len(st.session_state.stored_files))

        st.session_state.results = results
        st.session_state.processing = False
        status_text.text("✅ Processing completed!")

# Results Section
if st.session_state.results:
    st.markdown("---")
    st.markdown("## 🎉 Results")
    
    # Results grid
    cols = st.columns(min(4, len(st.session_state.results)))
    for i, (name, img, buf) in enumerate(st.session_state.results):
        with cols[i % len(cols)]:
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.caption(f"{img.size[0]}×{img.size[1]}px • {len(buf.getvalue())/1024:.1f} KB")
            st.download_button(
                "📥 Download",
                data=buf.getvalue(),
                file_name=f"processed_{name}",
                mime="image/jpeg",
                key=f"dl_{i}",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Bulk download
    st.markdown("---")
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w") as zf:
        for name, _, buf in st.session_state.results:
            zf.writestr(f"processed_{name}", buf.getvalue())
    z.seek(0)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            "📦 Download All as ZIP",
            data=z.getvalue(),
            file_name="processed_images.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary"
        )

# Clear button at bottom
if st.session_state.stored_files or st.session_state.results:
    st.markdown("---")
    if st.button("🗑️ Clear All & Start Over", use_container_width=True):
        st.session_state.upload_key += 1
        st.session_state.stored_files = []
        st.session_state.results = []
        st.session_state.processing = False
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Footer
if not st.session_state.stored_files:
    st.markdown("""
    <div style="text-align: center; opacity: 0.6; margin-top: 3rem;">
        <p>🚀 Upload your images to start the AI-powered transformation!</p>
        <p>Supports JPG, JPEG, PNG • Max 200MB per image</p>
    </div>
    """, unsafe_allow_html=True)