import os
import io, zipfile
import ssl, warnings
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rembg import remove
from ultralytics import YOLO
from multiprocessing import Pool
from functools import partial
import pickle
import warnings
warnings.filterwarnings("ignore",message=".*ScriptRunContext.*")


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

# Custom CSS for styling
st.markdown("""
<style>
    /* Main app styling */
    .main-header {
        background: linear-gradient(135deg, #2C3E50 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 12px;
        margin-bottom: 2rem;    
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: linear-gradient(90deg, #1E2761, #20c997);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #1E2761;
    }
    
    .sidebar-section h3 {
        color: #ffffff;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Step indicators */
    .step-container {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .step-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: black;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 1rem;
        font-size: 1.2rem;
    }
    
    .step-content h3 {
        margin: 0 0 0.5rem 0;
        color: #000000;
        font-size: 1.3rem;
    }
    
    .step-content p {
        margin: 0;
        color: #ffffff;
        font-size: 0.95rem;
    }
    
    /* Progress styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
    }
    
    .stError {
        background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    
    /* Results grid */
    .results-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .result-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }

    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: black;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    
    .info-card h3 {
        margin-bottom: 0.5rem;
        font-size: 1.4rem;
    }
    
    .info-card p {
        margin: 0;
        opacity: 0.9;
    }
    
    /* Features list */
    .features-list {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .feature-item {
        background: linear-gradient(135deg, #FF6B6B 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .feature-item h4 {
        color: #000000;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }

    .feature-item p {
        color: #000000;
        margin: 0;
        font-size: 0.9rem;
    }

    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


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


def enhanced_subject_detection_for_mp(img_array):
    """Modified version for multiprocessing that uses numpy array instead of PIL Image"""
    # Load model inside the function for multiprocessing
    local_model = YOLO("yolov8n-seg.pt")
    
    # Convert numpy array to opencv format
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    results = local_model.predict(img_cv, classes=0, verbose=False)
    
    for r in results:
        if hasattr(r, 'masks') and r.masks is not None:
            masks = r.masks.xy
            if masks:
                largest_mask = max(masks, key=lambda m: cv2.contourArea(m.astype(np.int32)))
                x, y, w, h = cv2.boundingRect(largest_mask.astype(np.int32))
                return (x, y, x + w, y + h)
    
    # Fallback to rembg if YOLO doesn't detect
    img_pil = Image.fromarray(img_array)
    bg_removed = remove(img_pil, post_process_mask=True)
    alpha = bg_removed.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        dx = int((bbox[2] - bbox[0]) * 0.05)
        dy = int((bbox[3] - bbox[1]) * 0.05)
        x0, y0 = max(0, bbox[0] - dx), max(0, bbox[1] - dy)
        x1 = min(img_pil.width, bbox[2] + dx)
        y1 = min(img_pil.height, bbox[3] + dy)
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


# =================== Multiprocessing Function ===================
def process_single_image(args):
    """
    Process a single image with all the branding effects.
    This function is designed to work with multiprocessing.
    """
    try:
        # Unpack arguments
        file_data, params = args
        filename, img_bytes = file_data
        
        # Load image from bytes
        img = Image.open(io.BytesIO(img_bytes))
        img = preprocess_uploaded_image(img)
        
        # Resize if too large
        if max(img.size) > 3000:
            img = img.resize((img.width//2, img.height//2), Image.LANCZOS)
        
        # Enhanced subject detection
        img_array = np.array(img)
        bb = enhanced_subject_detection_for_mp(img_array)
        if bb is None:
            bb = (img.width//4, img.height//4, 3*img.width//4, 3*img.height//4)
        
        # Smart resize and crop
        base = smart_resize_preserve_background(
            img, bb, (params['tw'], params['th']), 
            params['ts'], params['bs']
        ).convert("RGBA")
        
        # Add logo if provided
        if params['logo_img_bytes'] is not None:
            logo_img = pickle.loads(params['logo_img_bytes'])
            
            lw = int(params['scale']/100 * base.width)
            lh = int(lw / logo_img.width * logo_img.height)
            logo_res = logo_img.resize((lw, lh), Image.LANCZOS)
            x_px = int((params['x_off']/100) * (base.width - lw))
            y_px = int((params['y_off']/100) * (base.height - lh))
            
            if params['bgblur']:
                base = add_blur_background_under_logo(
                    base, logo_res, x_px, y_px, params['br'], params['mm']
                ).convert("RGBA")
            
            if params['shadow']:
                base = add_black_glow_around_logo(
                    base, logo_res, x_px, y_px, params['sr'], params['so']
                ).convert("RGBA")
            else:
                base.paste(logo_res, (x_px, y_px), logo_res)
        
        # Add text overlay if provided
        if params['overlay_text']:
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
            font_file = font_paths.get(params['font_family'], "arial.ttf")
            try:
                font = ImageFont.truetype(font_file, params['text_size'])
            except:
                font = ImageFont.load_default()
            
            bbox = draw.textbbox((0, 0), params['overlay_text'], font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x_text = int((params['text_x_pct']/100) * (base.width - w))
            y_text = int((params['text_y_pct']/100) * (base.height - h))
            draw.text((x_text, y_text), params['overlay_text'], font=font, fill=params['text_color'])
        
        # Convert to RGB and optimize
        final = base.convert("RGB")
        buf = optimize_image(final, params['max_kb'])
        
        return (filename, final, buf)
        
    except Exception as e:
        return (filename, None, str(e))


# =================== UI State ===================
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "stored_files" not in st.session_state:
    st.session_state.stored_files = []
if "results" not in st.session_state:
    st.session_state.results = []


# =================== Main UI ===================

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 AI-Powered Smart Cropper + Brand Generator</h1>
    <p>Professional image processing with intelligent cropping, branding, and batch optimization</p>
</div>
""", unsafe_allow_html=True)

# Features overview
st.markdown("""
<div class="features-list">
    <div class="feature-item">
        <h4>🎨 Professional Branding</h4>
        <p>Add logos, text overlays, and visual effects with pixel-perfect positioning</p>
    </div>
    <div class="feature-item">
        <h4>⚡ Batch Processing</h4>
        <p>Process multiple images simultaneously with parallel computing</p>
    </div>
    <div class="feature-item">
        <h4>📐 Smart Cropping</h4>
        <p>Intelligent aspect ratio handling while preserving important visual elements</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("""
    <div class="sidebar-section">
        <h3>🎛️ Control Center</h3>
        <p>Configure your image processing pipeline</p>
    </div>
    """, unsafe_allow_html=True)
    
    mode = st.selectbox(
        "Processing Mode:",
        ["🎯 Smart Cropper + Branding"],
        help="Select your desired processing mode"
    )
    
    if st.button("🗑️ Clear All Files", help="Reset all uploaded files and results"):
        st.session_state.upload_key += 1
        st.session_state.stored_files = []
        st.session_state.results = []
        st.rerun()

# Main workflow steps
col1, col2 = st.columns([2, 1])

with col1:
    files = st.file_uploader(
        "Choose image files",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True,
        key=f"up_{st.session_state.upload_key}",
        help="Select one or more images to process"
    )
    
    if files:
        st.session_state.stored_files = files
        st.success(f"✅ {len(files)} file(s) uploaded successfully!")
        
        # Show preview of uploaded files
        with st.expander("📁 View Uploaded Files", expanded=False):
            cols = st.columns(min(4, len(files)))
            for i, file in enumerate(files[:4]):  # Show first 4 images
                with cols[i]:
                    img = Image.open(file)
                    st.image(img, caption=file.name, use_container_width=True)
            if len(files) > 4:
                st.info(f"... and {len(files) - 4} more files")

with col2:
    if st.session_state.stored_files:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Upload Summary</h3>
            <p><strong>Files:</strong> {}</p>
            <p><strong>Status:</strong> Ready to process</p>
        </div>
        """.format(len(st.session_state.stored_files)), unsafe_allow_html=True)

# Sidebar settings (same logic, better organized)
if mode == "🎯 Smart Cropper + Branding":
    with st.sidebar:
        st.markdown("---")
        
        # Output Configuration
        st.markdown("""
        <div class="sidebar-section">
            <h3>📐 Output Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🎯 Dimensions & Quality", expanded=True):
            tw = st.number_input("Width (pixels)", 512, 4096, 1200, 100)
            th = st.number_input("Height (pixels)", 512, 4096, 1800, 100)
            max_kb = st.number_input("Max File Size (KB)", 100, 5000, 800, 50)
            
            # Show aspect ratio
            aspect_ratio = round(tw/th, 2)
            st.info(f"📏 Aspect Ratio: {aspect_ratio}:1")

        with st.expander("⚡ Performance Settings", expanded=False):
            max_workers = st.slider("Parallel Processes", 1, 8, 4, 1)
            st.info(f"🚀 Using {max_workers} processes for faster processing")

        with st.expander("🧠 Smart Cropping", expanded=False):
            use_space = st.checkbox("Add Head/Foot Space", help="Add extra space around detected subject")
            if use_space:
                ts = st.number_input("Top Space (pixels)", 0, 1000, 10)
                bs = st.number_input("Bottom Space (pixels)", 0, 1000, 10)
            else:
                ts = bs = 0

        # Branding Configuration
        st.markdown("---")
        st.markdown("""
        <div class="sidebar-section">
            <h3>🎨 Branding & Design</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🖼️ Logo Settings", expanded=True):
            collab_mode = st.checkbox("Collaboration Mode", help="Merge two logos for collaboration posts")
            
            if collab_mode:
                logo_file1 = st.file_uploader("First Logo", type=["png","jpg","jpeg"], key="logo1_up")
                logo_file2 = st.file_uploader("Second Logo", type=["png","jpg","jpeg"], key="logo2_up")
                
                if logo_file1 and logo_file2:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.image(logo_file1, caption="Logo 1", width=100)
                    with col_b:
                        st.image(logo_file2, caption="Logo 2", width=100)
                
                merge_padding = st.slider("Logo Spacing", 0, 100, 20)
                separator_text = st.text_input("Separator Symbol", value="×")
                separator_font_size = st.slider("Symbol Size", 10, 200, 40)
            else:
                logo_file = st.file_uploader("Upload Logo", type=["png","jpg","jpeg"], key="logo_up")
                if logo_file:
                    st.image(logo_file, caption="Logo Preview", width=150)

            if (collab_mode and logo_file1 and logo_file2) or (not collab_mode and logo_file):
                st.markdown("**Logo Positioning:**")
                col_scale, col_pos = st.columns(2)
                with col_scale:
                    scale = st.slider("Size (% of width)", 5, 50, 30)
                with col_pos:
                    x_off = st.slider("Horizontal Position", 0, 100, 50)
                    y_off = st.slider("Vertical Position", 0, 100, 90)

        with st.expander("✨ Visual Effects", expanded=False):
            shadow = st.checkbox("Enable Logo Shadow", value=True)
            if shadow:
                col_sr, col_so = st.columns(2)
                with col_sr:
                    sr = st.slider("Shadow Blur", 2, 50, 25)
                with col_so:
                    so = st.slider("Shadow Opacity", 0, 100, 30)
            else:
                sr = so = 0
                
            bgblur = st.checkbox("Background Blur Under Logo")
            if bgblur:
                col_br, col_mm = st.columns(2)
                with col_br:
                    br = st.slider("Blur Radius", 1, 50, 10)
                with col_mm:
                    mm = st.slider("Mask Margin", 1, 50, 5)
            else:
                br = mm = 0

        with st.expander("🖋️ Text Overlay", expanded=False):
            overlay_text = st.text_input("Overlay Text", placeholder="Enter text to overlay...")
            
            if overlay_text:
                col_size, col_color = st.columns(2)
                with col_size:
                    text_size = st.slider("Font Size", 10, 200, 40)
                with col_color:
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
                
                col_tx, col_ty = st.columns(2)
                with col_tx:
                    text_x_pct = st.slider("Text Horizontal Position", 0, 100, 50)
                with col_ty:
                    text_y_pct = st.slider("Text Vertical Position", 0, 100, 95)

# Step 3: Processing
if st.session_state.stored_files:
    st.markdown("""
    <div class="step-container">
        <div class="step-number">1</div>
        <div class="step-content">
            <h3>Process Your Images</h3>
            <p>Apply your configured settings to all uploaded images</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Processing button with enhanced styling
    process_col1, process_col2, process_col3 = st.columns([1, 2, 1])
    with process_col2:
        if st.button("🚀 Start Processing", help="Begin processing all uploaded images", use_container_width=True):
            # Prepare logo image
            logo_img_bytes = None
            if collab_mode and logo_file1 and logo_file2:
                tmp1 = Image.open(logo_file1).convert("RGBA")
                tmp2 = Image.open(logo_file2).convert("RGBA")
                
                # Remove white backgrounds
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
                logo_img_bytes = pickle.dumps(logo_img)
                
            elif not collab_mode and logo_file:
                tmp = Image.open(logo_file).convert("RGBA")
                datas = tmp.getdata()
                newData = [(r,g,b,0) if r>240 and g>240 and b>240 else (r,g,b,a) for r,g,b,a in datas]
                tmp.putdata(newData)
                logo_img_bytes = pickle.dumps(tmp)

            # Prepare parameters for multiprocessing
            params = {
                'tw': tw,
                'th': th,
                'ts': ts,
                'bs': bs,
                'max_kb': max_kb,
                'logo_img_bytes': logo_img_bytes,
                'scale': scale if ((collab_mode and logo_file1 and logo_file2) or (not collab_mode and logo_file)) else 30,
                'x_off': x_off if ((collab_mode and logo_file1 and logo_file2) or (not collab_mode and logo_file)) else 50,
                'y_off': y_off if ((collab_mode and logo_file1 and logo_file2) or (not collab_mode and logo_file)) else 90,
                'shadow': shadow,
                'sr': sr,
                'so': so,
                'bgblur': bgblur,
                'br': br,
                'mm': mm,
                'overlay_text': overlay_text,
                'text_size': text_size if overlay_text else 40,
                'text_color': text_color if overlay_text else "#FFFFFF",
                'font_family': font_family if overlay_text else "Arial",
                'text_x_pct': text_x_pct if overlay_text else 50,
                'text_y_pct': text_y_pct if overlay_text else 95
            }

            # Prepare file data for multiprocessing
            file_data_list = []
            for f in st.session_state.stored_files:
                f.seek(0)  # Reset file pointer
                file_bytes = f.read()
                file_data_list.append((f.name, file_bytes))

            # Create arguments list for multiprocessing
            args_list = [(file_data, params) for file_data in file_data_list]

            # Processing status display
            st.markdown("""
            <div class="info-card">
                <h3>⚙️ Processing Status</h3>
                <p>Processing {} images with {} parallel processes...</p>
            </div>
            """.format(len(file_data_list), max_workers), unsafe_allow_html=True)

            # Process images in parallel with real-time progress updates
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    if __name__ == "__main__" or True:  # Added for Streamlit compatibility
                        with Pool(processes=max_workers) as pool:
                            results = []
                            completed = 0
                            total = len(args_list)
                            
                            # Use imap_unordered to get results as they complete
                            for result in pool.imap_unordered(process_single_image, args_list):
                                results.append(result)
                                completed += 1
                                progress = completed / total
                                progress_bar.progress(progress)
                                status_text.text(f"🔄 Processed {completed}/{total} images...")
                        
                        # Filter out any failed results
                        successful_results = []
                        failed_results = []
                        
                        for result in results:
                            filename, img, buf_or_error = result
                            if img is not None:
                                successful_results.append(result)
                            else:
                                failed_results.append((filename, buf_or_error))
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Processing complete!")
                        
                        if failed_results:
                            st.warning(f"⚠️ Failed to process {len(failed_results)} images:")
                            for filename, error in failed_results:
                                st.error(f"❌ {filename}: {error}")
                        
                        if successful_results:
                            st.success(f"🎉 Successfully processed {len(successful_results)} images!")
                            st.session_state.results = successful_results
                        else:
                            st.error("❌ No images were processed successfully.")
                            st.session_state.results = []
                            
                except Exception as e:
                    st.error(f"❌ Multiprocessing error: {str(e)}")
                    st.info("🔄 Falling back to sequential processing...")
                    
                    # Fallback to sequential processing with progress
                    results = []
                    for i, args in enumerate(args_list):
                        result = process_single_image(args)
                        results.append(result)
                        progress = (i + 1) / len(args_list)
                        progress_bar.progress(progress)
                        status_text.text(f"🔄 Processed {i + 1}/{len(args_list)} images...")
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Sequential processing complete!")
                    st.session_state.results = [r for r in results if r[1] is not None]


# Step 4: Results
if st.session_state.results:
    st.markdown("""
    <div class="step-container">
        <div class="step-number">2</div>
        <div class="step-content">
            <h3>Download Your Results</h3>
            <p>Preview and download your processed images individually or as a batch</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Results summary
    st.markdown("""
    <div class="info-card">
        <h3>📊 Processing Results</h3>
        <p><strong>Successfully processed:</strong> {} images</p>
        <p><strong>Total file size:</strong> {} KB (estimated)</p>
        <p><strong>Ready for download</strong> ✅</p>
    </div>
    """.format(
        len(st.session_state.results),
        sum(len(buf.getvalue()) for _, _, buf in st.session_state.results) // 1024
    ), unsafe_allow_html=True)
    
    # Batch download section
    st.markdown("### 📦 Batch Download")
    
    # Create ZIP file for batch download
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w") as zf:
        for name, _, buf in st.session_state.results:
            zf.writestr(f"branded_{name}", buf.getvalue())
    z.seek(0)
    
    download_col1, download_col2, download_col3 = st.columns([1, 2, 1])
    with download_col2:
        st.download_button(
            "📥 Download All Images (ZIP)",
            data=z.getvalue(),
            file_name="branded_images.zip",
            mime="application/zip",
            use_container_width=True,
            help="Download all processed images as a ZIP file"
        )
    
    # Individual results grid
    st.markdown("### 🖼️ Individual Results")
    
    # Create responsive grid
    cols_per_row = 4
    num_results = len(st.session_state.results)
    
    for i in range(0, num_results, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (name, img, buf) in enumerate(st.session_state.results[i:i+cols_per_row]):
            with cols[j]:
                
                # Display image
                st.image(img, use_container_width=True)
                
                # Image info
                file_size_kb = len(buf.getvalue()) // 1024
                st.caption(f"📏 {img.width}×{img.height} • 📁 {file_size_kb} KB")
                
                # Download button
                st.download_button(
                    "💾 Download",
                    data=buf.getvalue(),
                    file_name=f"branded_{name}",
                    mime="image/jpeg",
                    key=f"dl_{i}_{j}",
                    use_container_width=True,
                    help=f"Download {name}"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem;">
    <p>🎯 <strong>AI-Powered Smart Cropper + Brand Generator</strong></p>
    <p>Professional image processing with intelligent automation</p>
</div>
""", unsafe_allow_html=True)