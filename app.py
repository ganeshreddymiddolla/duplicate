import streamlit as st
import numpy as np
import time
import os
from PIL import Image, ImageOps

# ==========================================
# 1. CONFIGURATION (Must be first for speed)
# ==========================================
st.set_page_config(
    page_title="Lens AI",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. INSTANT-LOAD CSS (Animations & UI)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');

    /* --- RESET & PERFORMANCE --- */
    .stApp {
        background-color: #f0f2f5;
        font-family: 'Google Sans', sans-serif;
        color: #202124;
    }
    header, footer, #MainMenu {visibility: hidden;}
    
    /* --- ANIMATIONS --- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideUp {
        from { transform: translateY(100%); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(66, 133, 244, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(66, 133, 244, 0); }
        100% { box-shadow: 0 0 0 0 rgba(66, 133, 244, 0); }
    }

    /* --- HOME SCREEN --- */
    .hero-container {
        text-align: center;
        padding: 40px 20px;
        animation: fadeIn 0.8s ease-out;
    }
    
    .lens-logo {
        font-size: 60px;
        margin-bottom: 10px;
        display: inline-block;
        animation: pulse 2s infinite;
        border-radius: 50%;
    }
    
    .upload-box {
        background: white;
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-top: 20px;
        text-align: center;
    }

    /* --- RESULT SHEET --- */
    .result-sheet {
        background: white;
        border-radius: 30px 30px 0 0;
        margin-top: -30px;
        padding: 30px;
        box-shadow: 0 -10px 40px rgba(0,0,0,0.1);
        position: relative;
        z-index: 100;
        animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1); /* Apple-style ease */
    }

    .sheet-handle {
        width: 50px;
        height: 5px;
        background: #e0e0e0;
        border-radius: 10px;
        margin: 0 auto 25px auto;
    }

    /* --- TYPOGRAPHY & BADGES --- */
    .breed-title {
        font-size: 32px;
        font-weight: 400;
        margin: 10px 0;
        color: #202124;
        letter-spacing: -0.5px;
    }
    
    .match-pill {
        display: inline-flex;
        align-items: center;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 500;
        background: #e8f0fe;
        color: #1a73e8;
    }
    .match-pill.fail {
        background: #fce8e6;
        color: #c5221f;
    }

    /* --- INFO GRID --- */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
        margin-top: 25px;
    }
    
    .info-card {
        background: #f8f9fa;
        padding: 16px;
        border-radius: 16px;
        transition: transform 0.2s;
    }
    .info-card:hover {
        transform: scale(1.02);
        background: #f1f3f4;
    }
    
    .info-label { font-size: 12px; color: #5f6368; font-weight: 600; text-transform: uppercase; }
    .info-val { font-size: 16px; color: #202124; font-weight: 500; margin-top: 4px; }

    /* --- BUTTONS --- */
    .stButton > button {
        width: 100%;
        border-radius: 50px;
        height: 56px;
        font-size: 16px;
        font-weight: 500;
        border: none;
        background-color: #1a73e8;
        color: white;
        box-shadow: 0 4px 12px rgba(26, 115, 232, 0.2);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1557b0;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(26, 115, 232, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. KNOWLEDGE GRAPH (Database)
# ==========================================
BREED_DB = {
    "Golden Retriever": {"Origin": "Scotland", "Life Span": "10-12 yrs", "Temperament": "Friendly, Intelligent, Devoted"},
    "Labrador Retriever": {"Origin": "Canada", "Life Span": "10-12 yrs", "Temperament": "Outgoing, Gentle, Agile"},
    "German Shepherd": {"Origin": "Germany", "Life Span": "7-10 yrs", "Temperament": "Confident, Smart, Brave"},
    "Bulldog": {"Origin": "UK", "Life Span": "8-10 yrs", "Temperament": "Docile, Willful, Friendly"},
    "Beagle": {"Origin": "UK", "Life Span": "10-15 yrs", "Temperament": "Merry, Curious, Friendly"},
    "Poodle": {"Origin": "France", "Life Span": "10-18 yrs", "Temperament": "Active, Proud, Smart"},
    "Rottweiler": {"Origin": "Germany", "Life Span": "9-10 yrs", "Temperament": "Loyal, Loving, Confident"},
    "Siberian Husky": {"Origin": "Siberia", "Life Span": "12-14 yrs", "Temperament": "Loyal, Mischievous"},
    "Dachshund": {"Origin": "Germany", "Life Span": "12-16 yrs", "Temperament": "Clever, Stubborn"},
    "Great Dane": {"Origin": "Germany", "Life Span": "7-10 yrs", "Temperament": "Friendly, Patient"},
    "Doberman": {"Origin": "Germany", "Life Span": "10-12 yrs", "Temperament": "Fearless, Alert"},
    "Pug": {"Origin": "China", "Life Span": "13-15 yrs", "Temperament": "Charming, Loving"},
    "Chihuahua": {"Origin": "Mexico", "Life Span": "14-16 yrs", "Temperament": "Sassy, Graceful"},
    "Shih Tzu": {"Origin": "China", "Life Span": "10-18 yrs", "Temperament": "Playful, Outgoing"},
    "Border Collie": {"Origin": "UK", "Life Span": "12-15 yrs", "Temperament": "Energetic, Alert"},
}

FALLBACK_DB = {"Origin": "Unknown", "Life Span": "10-13 yrs", "Temperament": "Loyal Companion"}

# ==========================================
# 4. OPTIMIZED BACKEND LOGIC
# ==========================================
MODEL_PATH = 'final_model.keras'
CLASSES_PATH = 'classes.txt'
CONFIDENCE_THRESHOLD = 50.0

# Initialize State
if 'page' not in st.session_state: st.session_state.page = 'HOME'
if 'result' not in st.session_state: st.session_state.result = None

def reset():
    st.session_state.page = 'HOME'
    st.session_state.result = None
    st.rerun()

def get_details(name):
    # Fuzzy search
    for key in BREED_DB:
        if key in name: return BREED_DB[key]
    return FALLBACK_DB

# ==========================================
# 5. UI: HOME SCREEN
# ==========================================
if st.session_state.page == 'HOME':
    
    st.markdown("""
        <div class="hero-container">
            <div class="lens-logo">📸</div>
            <h1 style="margin:0; font-size: 28px;">Google Lens</h1>
            <p style="color:#5f6368;">Search what you see</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image to start", type=['jpg','jpeg','png'])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        # --- LAZY LOADING & ANIMATION ---
        # The spinner holds the UI while we import the heavy libraries
        with st.spinner("Analyzing neural patterns..."):
            
            # 1. Import TF only now (Instant Load Hack)
            import tensorflow as tf
            
            # 2. Load Model
            if not os.path.exists(MODEL_PATH):
                st.error("Model missing! Please run training script.")
                st.stop()
                
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(CLASSES_PATH, 'r') as f:
                classes = [line.strip() for line in f.readlines()]

            # 3. Process
            image = Image.open(uploaded_file).convert('RGB')
            img_resized = ImageOps.fit(image, (224, 224), Image.LANCZOS)
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            # 4. Predict
            preds = model.predict(img_array)
            score = preds[0]
            
            top_idx = np.argmax(score)
            conf = 100 * np.max(score)
            raw_name = classes[top_idx]
            
            # Clean Name: "n02099-Golden_Retriever" -> "Golden Retriever"
            breed_name = raw_name.split('-', 1)[1].replace('_', ' ').title() if '-' in raw_name else raw_name.replace('_', ' ').title()

            # Get Top 3
            top_3_idx = np.argsort(score)[-3:][::-1]
            alts = [{"name": classes[i].split('-', 1)[1].replace('_', ' ').title() if '-' in classes[i] else classes[i], "conf": 100*score[i]} for i in top_3_idx]

            # 5. Save & Switch
            st.session_state.result = {
                "image": image,
                "breed": breed_name,
                "conf": conf,
                "alts": alts
            }
            st.session_state.page = 'RESULT'
            st.rerun()

# ==========================================
# 6. UI: RESULT SCREEN (Bottom Sheet)
# ==========================================
elif st.session_state.page == 'RESULT':
    data = st.session_state.result
    breed = data['breed']
    conf = data['conf']
    
    # Full Width Image
    st.image(data['image'], use_container_width=True)
    
    # Bottom Sheet
    st.markdown('<div class="result-sheet">', unsafe_allow_html=True)
    st.markdown('<div class="sheet-handle"></div>', unsafe_allow_html=True)
    
    # --- LOGIC: CONFIDENCE CHECK ---
    if conf < CONFIDENCE_THRESHOLD:
        # FAILED MATCH
        st.markdown(f"""
            <div class="match-pill fail">Low Confidence ({conf:.1f}%)</div>
            <h1 class="breed-title">Not a Dog / Unsure</h1>
            <p style="color:#5f6368; line-height:1.5;">
                I'm not confident this is a dog. The visual patterns only match <b>{breed}</b> by {conf:.0f}%.
            </p>
        """, unsafe_allow_html=True)
        
        with st.expander("View technical guesses"):
            for alt in data['alts']:
                st.write(f"{alt['name']}: {alt['conf']:.1f}%")

    else:
        # SUCCESSFUL MATCH
        info = get_details(breed)
        
        st.markdown(f"""
            <div class="match-pill">Visual Match: {conf:.1f}%</div>
            <h1 class="breed-title">{breed}</h1>
        """, unsafe_allow_html=True)
        
        # Details Grid
        st.markdown(f"""
        <div class="info-grid">
            <div class="info-card">
                <div class="info-label">Origin</div>
                <div class="info-val">{info['Origin']}</div>
            </div>
            <div class="info-card">
                <div class="info-label">Lifespan</div>
                <div class="info-val">{info['Life Span']}</div>
            </div>
            <div class="info-card" style="grid-column: span 2;">
                <div class="info-label">Temperament</div>
                <div class="info-val">{info['Temperament']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Breakdown Chart
        st.markdown("<h3 style='font-size:14px; color:#5f6368; margin-top:30px; text-transform:uppercase;'>Confidence Breakdown</h3>", unsafe_allow_html=True)
        for alt in data['alts']:
            st.write(f"**{alt['name']}**")
            st.progress(int(alt['conf']))

        # Google Search Link
        st.markdown(f"""
        <br>
        <div style="text-align:center;">
            <a href="https://www.google.com/search?q={breed}+dog" target="_blank" 
               style="color:#1a73e8; text-decoration:none; font-weight:600; font-size:15px;">
               Search on Google ➜
            </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True) # End Sheet

    # Reset Button
    st.write("")
    if st.button("Scan New Object"):
        reset()