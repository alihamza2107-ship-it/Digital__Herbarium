# Digital Herbarium - Streamlit app (for herbarium_model.keras)
# Developed by NanStack.tech  |  https://www.nanstack.tech/
#
# How to run:   python -m streamlit run app_keras.py
# herbarium_model.keras must be in this folder.
# Optional: confusion_matrix.png and training_curves.png (shown in the Model tab).
import os
import html
import hmac
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Digital Herbarium", page_icon="🌿", layout="wide",
                   initial_sidebar_state="collapsed")

MODEL_PATH = os.environ.get("HERBARIUM_KERAS", "herbarium_model.keras")
NAN_URL = "https://www.nanstack.tech/"
NAN_CONTACT = "https://www.nanstack.tech/contact"
DISCLAIMER = "This information is for educational purposes only and is not medical advice."

# Default login (can be changed with environment variables: HERBARIUM_USER, HERBARIUM_PASS)
APP_USER = os.environ.get("HERBARIUM_USER", "NanStack")
APP_PASS = os.environ.get("HERBARIUM_PASS", "AliHamza")

# Folder number (0..29) -> plant info. Row: (scientific, common, family, habitat, uses)
PLANTS = [
 ("Aerva lanata","Polpala / Mountain Knotgrass","Amaranthaceae","Tropical Asia and Africa, perennial herb","Traditionally used for kidney and urinary complaints"),
 ("Aloe vera","Aloe Vera","Asphodelaceae","Arid regions, succulent","Gel for skin care and minor burns, traditional medicine"),
 ("Andrographis paniculata","Kalmegh / King of Bitters","Acanthaceae","South and Southeast Asia, annual herb","Bitter tonic, traditionally used for fever and colds"),
 ("Bacopa monnieri","Brahmi / Water Hyssop","Plantaginaceae","Wetlands of tropical regions, creeping herb","Traditional Ayurvedic memory tonic"),
 ("Calotropis gigantea","Crown Flower / Giant Milkweed","Apocynaceae","Tropical Asia, shrub of dry wasteland","Used in folk medicine. ⚠️ Latex is toxic and irritating"),
 ("Centella asiatica","Gotu Kola","Apiaceae","Wetlands of Asia, creeping herb","Traditionally used for wound healing and as a memory tonic"),
 ("Chromolaena odorata","Siam Weed","Asteraceae","Native to the Americas, invasive in Asia","Leaf paste used in folk medicine for cuts. Invasive weed"),
 ("Cissus quadrangularis","Hadjod / Veldt Grape","Vitaceae","Tropical Asia and Africa, succulent climber","Traditionally used for bone and joint health"),
 ("Clerodendrum infortunatum","Hill Glory Bower","Lamiaceae","South and Southeast Asia, shrub","Folk medicine for skin problems and fever"),
 ("Clitoria ternatea","Butterfly Pea / Aparajita","Fabaceae","Tropical Asia, climbing vine","Blue flower tea, natural food dye, traditional tonic"),
 ("Coccinia grandis","Ivy Gourd / Tindora","Cucurbitaceae","Tropical Asia and Africa, climbing vine","Edible fruit eaten as vegetable, leaves in folk medicine"),
 ("Curcuma longa","Turmeric / Haldi","Zingiberaceae","South Asia, rhizome herb","Spice, traditional antiseptic and anti-inflammatory use"),
 ("Datura metel","Devil's Trumpet / Dhatura","Solanaceae","Tropical and subtropical regions, shrub","⚠️ Highly poisonous. Do not consume. Studied in traditional medicine"),
 ("Desmodium gangeticum","Shalaparni","Fabaceae","Tropical Asia, shrub","Root used in Ayurvedic formulations for fever and respiratory complaints"),
 ("Hemigraphis colorata","Purple Waffle Plant","Acanthaceae","Southeast Asia, creeping herb","Ornamental, leaves used in folk medicine for wounds"),
 ("Kaempferia galanga","Aromatic Ginger / Kachur","Zingiberaceae","Tropical Asia, rhizome herb","Rhizome used in cooking and for cough and indigestion"),
 ("Mentha spicata","Spearmint / Pudina","Lamiaceae","Temperate regions, herb","Culinary herb, tea, traditional digestive aid"),
 ("Ocimum gratissimum","Clove Basil / Ram Tulsi","Lamiaceae","Tropical Africa and Asia, shrub","Culinary herb, traditional remedy for colds and stomach upset"),
 ("Ocimum tenuiflorum","Tulsi / Holy Basil","Lamiaceae","Tropical Asia, herb","Herbal tea, traditional remedy for cough and cold"),
 ("Oxalis corniculata","Creeping Wood Sorrel","Oxalidaceae","Worldwide weed, creeping herb","Sour edible leaves, traditional digestive use. Contains oxalates"),
 ("Peperomia pellucida","Shiny Bush / Silver Bush","Piperaceae","Tropical regions, small succulent herb","Edible in salads, folk remedy for joint pain"),
 ("Phyllanthus niruri","Stone Breaker / Bhumi Amla","Phyllanthaceae","Tropical regions, herb","Traditionally used for kidney stones and liver complaints"),
 ("Piper betle","Betel Leaf / Paan","Piperaceae","Tropical Asia, climbing vine","Chewed with areca nut, traditional digestive and antiseptic use"),
 ("Piper longum","Long Pepper / Pippali","Piperaceae","Indian subcontinent, climber","Spice, Ayurvedic remedy for respiratory complaints"),
 ("Piper nigrum","Black Pepper / Kali Mirch","Piperaceae","South India, climbing vine","Spice, traditional digestive aid"),
 ("Plectranthus barbatus","Indian Coleus","Lamiaceae","Tropical Africa and Asia, herb","Roots traditionally used for heart and digestive complaints"),
 ("Plumbago indica","Rosy Leadwort / Lal Chitrak","Plumbaginaceae","South and Southeast Asia, shrub","Root used in traditional medicine. ⚠️ Toxic and skin irritant"),
 ("Pseudarthria viscida","Moovila","Fabaceae","Tropical Asia, herb","Root used in Ayurvedic formulations (Dashamoola group)"),
 ("Ricinus communis","Castor / Arandi","Euphorbiaceae","Tropical regions, shrub","Castor oil from seeds. ⚠️ Raw seeds are highly toxic"),
 ("Tinospora cordifolia","Giloy / Guduchi","Menispermaceae","Indian subcontinent, climbing shrub","Traditionally used for immunity and fever"),
]

# During training, class folders are sorted as strings: '0','1','10','11',...,'2','20',...
FOLDERS = sorted(str(i) for i in range(30))
NAMES = [PLANTS[int(f)][0] for f in FOLDERS]
INFO = []
for f in FOLDERS:
    sci, common, fam, hab, uses = PLANTS[int(f)]
    INFO.append({"common": common, "scientific": sci, "family": fam, "habitat": hab, "uses": uses})

# ======================= Styling =======================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif; }
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 1.1rem; padding-bottom: 1rem; max-width: 1180px; }

.hero { background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 55%, #43A047 100%); color: #fff;
  border-radius: 22px; padding: 26px 28px; box-shadow: 0 10px 30px rgba(27,94,32,.28); position: relative; overflow: hidden; }
.hero:after { content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
  background: rgba(255,255,255,.08); border-radius: 50%; }
.hero-top { display: flex; align-items: center; gap: 14px; }
.logo { width: 54px; height: 54px; border-radius: 16px; background: rgba(255,255,255,.18);
  display: flex; align-items: center; justify-content: center; flex: 0 0 auto; }
.hero h1 { margin: 0; padding: 0 !important; font-size: 1.9rem; font-weight: 800; letter-spacing: -.5px; color: #fff; line-height: 1.15; }
.hero p { margin: 4px 0 0 0 !important; opacity: .9; font-size: .98rem; color: #fff; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; position: relative; z-index: 1; }
.chip { background: rgba(255,255,255,.17); border: 1px solid rgba(255,255,255,.28); color: #fff;
  padding: 5px 12px; border-radius: 999px; font-size: .8rem; font-weight: 600; }

.card { background: rgba(46,125,50,.07); border: 1px solid rgba(46,125,50,.25); border-radius: 18px;
  padding: 18px 20px; margin-bottom: 14px; }
.plant-name { font-size: 1.6rem; font-weight: 800; line-height: 1.2; margin: 2px 0; }
.sci { font-style: italic; opacity: .8; margin-bottom: 10px; }
.muted { opacity: .65; font-size: .78rem; text-transform: uppercase; letter-spacing: .8px; font-weight: 600; }
.badge { display: inline-block; background: rgba(46,125,50,.16); color: #2E7D32; border: 1px solid rgba(46,125,50,.35);
  padding: 3px 11px; border-radius: 999px; font-size: .78rem; font-weight: 700; margin: 0 6px 6px 0; }
.badge.warn { background: rgba(211,47,47,.12); color: #D32F2F; border-color: rgba(211,47,47,.4); }
.row-k { font-weight: 700; font-size: .8rem; opacity: .7; text-transform: uppercase; letter-spacing: .6px; margin-top: 12px; }
.row-v { font-size: .98rem; margin-top: 2px; }
.notice { border-radius: 14px; padding: 11px 14px; font-size: .9rem; margin: 8px 0 14px 0; font-weight: 500; }
.notice.warn { background: rgba(249,168,37,.14); border: 1px solid rgba(249,168,37,.5); }
.notice.danger { background: rgba(211,47,47,.1); border: 1px solid rgba(211,47,47,.45); }

.conf-wrap { margin: 12px 0 4px 0; }
.conf-head { display: flex; justify-content: space-between; font-weight: 700; font-size: .85rem; margin-bottom: 5px; }
.track { height: 12px; background: rgba(128,128,128,.22); border-radius: 999px; overflow: hidden; }
.fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #66BB6A, #2E7D32); }
.fill.low { background: linear-gradient(90deg, #FFCA28, #F57C00); }
.bar-row { display: flex; align-items: center; gap: 10px; margin: 8px 0; }
.bar-label { flex: 0 0 38%; font-size: .85rem; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track { flex: 1; height: 10px; background: rgba(128,128,128,.22); border-radius: 999px; overflow: hidden; }
.bar-fill { height: 100%; background: linear-gradient(90deg, #81C784, #388E3C); border-radius: 999px; }
.bar-val { flex: 0 0 52px; text-align: right; font-size: .82rem; font-weight: 700; }

.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr)); gap: 14px; }
.grid .card { margin: 0; }
.pc-name { font-weight: 800; font-size: 1.08rem; }
.pc-sci { font-style: italic; opacity: .75; font-size: .88rem; margin-bottom: 8px; }
.pc-uses { font-size: .88rem; opacity: .92; margin-top: 6px; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; overflow-x: auto; }
.stTabs [data-baseweb="tab"] { height: 44px; border-radius: 12px; padding: 0 16px; font-weight: 600; white-space: nowrap; }
.stTabs [aria-selected="true"] { background: rgba(46,125,50,.15); color: #2E7D32 !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #2E7D32 !important; }
.st-key-logout_btn { display: flex; justify-content: flex-end; margin-top: 8px; }
[data-testid="stForm"] { background: rgba(46,125,50,.07); border: 1px solid rgba(46,125,50,.25) !important;
  border-radius: 22px; padding: 24px 24px 12px 24px; box-shadow: 0 8px 24px rgba(0,0,0,.06); margin-top: 18px; }
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button { border-radius: 12px; font-weight: 700; padding: .55rem 1.1rem; min-height: 46px; }
button[kind^="primary"] { background: linear-gradient(135deg, #2E7D32, #43A047) !important; border: none !important; color: #fff !important; }
.stFormSubmitButton > button { width: 100%; }
.stTextInput input { border-radius: 12px; min-height: 46px; }
[data-testid="stFileUploader"] section { border-radius: 16px; border: 2px dashed rgba(46,125,50,.5); }

.login-title { font-size: 1.5rem; font-weight: 800; }
.login-sub { opacity: .7; margin-bottom: 8px; font-size: .92rem; }

.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
.step-num { width: 30px; height: 30px; border-radius: 50%; background: #2E7D32; color: #fff; display: flex;
  align-items: center; justify-content: center; font-weight: 800; margin-bottom: 8px; }
.btn-link { display: inline-block; padding: 11px 20px; border-radius: 12px; font-weight: 700; text-decoration: none !important;
  background: linear-gradient(135deg, #2E7D32, #43A047); color: #fff !important; margin: 6px 8px 0 0; }
.btn-link.alt { background: transparent; color: #2E7D32 !important; border: 2px solid #2E7D32; }

.nan-footer { text-align: center; margin-top: 30px; padding: 18px 8px; border-top: 1px solid rgba(128,128,128,.25);
  font-size: .9rem; opacity: .95; }
.nan-footer a { color: #2E7D32; font-weight: 800; text-decoration: none; }
.nan-footer a:hover { text-decoration: underline; }
.nan-footer span { display: block; opacity: .6; font-size: .78rem; margin-top: 3px; }

@media (max-width: 640px) {
  .block-container { padding: .7rem .75rem 1rem .75rem; }
  .hero { padding: 18px 16px; border-radius: 18px; }
  .hero h1 { font-size: 1.4rem; }
  .hero p { font-size: .88rem; }
  .logo { width: 44px; height: 44px; border-radius: 13px; }
  .plant-name { font-size: 1.35rem; }
  .bar-label { flex-basis: 44%; }
  .stTabs [data-baseweb="tab"] { padding: 0 12px; font-size: .88rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LOGO = ('<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>'
        '<path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>')


def footer():
    st.markdown(
        '<div class="nan-footer">Developed by <a href="' + NAN_URL + '" target="_blank" rel="noopener noreferrer">NanStack.tech</a>'
        '<span>Digital Herbarium · ' + str(datetime.now().year) + '</span></div>', unsafe_allow_html=True)


def hero(subtitle, chips):
    chip_html = "".join('<span class="chip">' + html.escape(c) + '</span>' for c in chips)
    st.markdown(
        '<div class="hero"><div class="hero-top"><div class="logo">' + LOGO + '</div><div>'
        '<h1>Digital Herbarium</h1><p>' + html.escape(subtitle) + '</p></div></div>'
        '<div class="chips">' + chip_html + '</div></div>', unsafe_allow_html=True)


# ======================= Login =======================
def check_login(u, p):
    ok_u = hmac.compare_digest(u.strip().lower().encode(), APP_USER.lower().encode())
    ok_p = hmac.compare_digest(p.strip().encode(), APP_PASS.encode())
    return ok_u and ok_p


def login_screen():
    hero("Identify plant species from a leaf photo", ["AI Powered", "30 Medicinal Plants", "Mobile Friendly"])
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        with st.form("login_form"):
            st.markdown('<div class="login-title">Welcome back 👋</div>'
                        '<div class="login-sub">Sign in to continue</div>', unsafe_allow_html=True)
            u = st.text_input("Username", placeholder="Username")
            p = st.text_input("Password", type="password", placeholder="Password")
            go = st.form_submit_button("Login →", type="primary")
        lock = st.session_state.get("lock_until", 0)
        if go:
            if time.time() < lock:
                st.error("Too many failed attempts. Please wait a moment and try again.")
            elif check_login(u, p):
                st.session_state.auth = True
                st.session_state.fails = 0
                st.rerun()
            else:
                st.session_state.fails = st.session_state.get("fails", 0) + 1
                if st.session_state.fails >= 5:
                    st.session_state.lock_until = time.time() + 30
                    st.session_state.fails = 0
                st.error("Incorrect username or password.")
    footer()


if not st.session_state.get("auth"):
    login_screen()
    st.stop()


# ======================= Model =======================
@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)


def predict(img, model):
    h, w = model.input_shape[1:3]
    arr = np.asarray(img.convert("RGB"), dtype="float32")[None]
    arr = tf.image.resize(arr, (h, w)).numpy()      # preprocessing is inside the model
    return model.predict(arr, verbose=0)[0]


if not os.path.exists(MODEL_PATH):
    st.error(f"'{MODEL_PATH}' not found. Place herbarium_model.keras in the same folder as app_keras.py.")
    st.stop()
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("Could not load the model. This is usually a TensorFlow version mismatch (install the same version that Colab used).")
    st.code(str(e)[:1500])
    st.stop()
if model.output_shape[-1] != len(NAMES):
    st.error(f"The model has {model.output_shape[-1]} classes, but this app is built for 30 classes.")
    st.stop()

if "log" not in st.session_state:
    st.session_state.log = []


# ======================= Helpers (UI) =======================
def result_card(d, conf, thr):
    toxic = "⚠️" in d["uses"]
    uses = html.escape(d["uses"].replace("⚠️", "").strip())
    low = conf < thr
    s = ('<div class="card"><div class="muted">Identified plant</div>'
         '<div class="plant-name">' + html.escape(d["common"]) + '</div>'
         '<div class="sci">' + html.escape(d["scientific"]) + '</div>'
         '<span class="badge">' + html.escape(d["family"]) + '</span>')
    if toxic:
        s += '<span class="badge warn">⚠ Toxic / caution</span>'
    s += ('<div class="conf-wrap"><div class="conf-head"><span>Confidence</span><span>' + f"{conf:.1f}%" + '</span></div>'
          '<div class="track"><div class="fill' + (" low" if low else "") + '" style="width:' + f"{min(conf, 100):.1f}" + '%"></div></div></div>'
          '<div class="row-k">Habitat</div><div class="row-v">' + html.escape(d["habitat"]) + '</div>'
          '<div class="row-k">Uses</div><div class="row-v">' + uses + '</div></div>')
    return s


def top_bars(top, probs):
    s = '<div class="card"><div class="muted">Top 3 predictions</div>'
    for i in top:
        pct = float(probs[int(i)]) * 100
        s += ('<div class="bar-row"><div class="bar-label">' + html.escape(NAMES[int(i)]) + '</div>'
              '<div class="bar-track"><div class="bar-fill" style="width:' + f"{pct:.1f}" + '%"></div></div>'
              '<div class="bar-val">' + f"{pct:.1f}%" + '</div></div>')
    return s + '</div>'


def plant_grid(items):
    s = '<div class="grid">'
    for d in items:
        toxic = "⚠️" in d["uses"]
        s += ('<div class="card"><div class="pc-name">' + html.escape(d["common"]) + '</div>'
              '<div class="pc-sci">' + html.escape(d["scientific"]) + '</div>'
              '<span class="badge">' + html.escape(d["family"]) + '</span>'
              + ('<span class="badge warn">⚠ Toxic</span>' if toxic else '') +
              '<div class="pc-uses"><b>Habitat:</b> ' + html.escape(d["habitat"]) + '</div>'
              '<div class="pc-uses"><b>Uses:</b> ' + html.escape(d["uses"].replace("⚠️", "").strip()) + '</div></div>')
    return s + '</div>'


# ======================= Main page =======================
hero("Identify plant species from a leaf photo, with scientific information",
     ["👤 " + APP_USER, "30 Species", "MobileNetV2", "Mobile Friendly"])

if st.button("Logout", key="logout_btn"):
    st.session_state.auth = False
    st.rerun()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Identify", "📚 My Herbarium", "🌱 All Plants", "📊 Model", "ℹ️ About"])

# ---- Identify ----
with tab1:
    left, right = st.columns(2)
    with left:
        mode = st.radio("How would you like to add a photo?", ["Upload", "Camera"], horizontal=True)
        if mode == "Upload":
            file = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png", "webp"])
        else:
            file = st.camera_input("Take a leaf photo")
        img = Image.open(file) if file else None
        if img is not None:
            st.image(img, caption="Your photo")
        with st.expander("⚙️ Settings"):
            threshold = st.slider("Low-confidence warning (%)", 10, 95, 60)
        st.caption("Tip: a clear photo of a single leaf on a plain background works best.")
    with right:
        if img is None:
            st.markdown('<div class="card"><div class="plant-name">Waiting for a photo 🌿</div>'
                        '<div class="row-v">Upload a leaf photo and the app will show the plant name, scientific name, family and uses.</div></div>',
                        unsafe_allow_html=True)
        else:
            probs = predict(img, model)
            top = np.argsort(probs)[::-1][:3]
            idx = int(top[0])
            conf = float(probs[idx]) * 100
            d = INFO[idx]
            st.markdown(result_card(d, conf, threshold), unsafe_allow_html=True)
            if "⚠️" in d["uses"]:
                st.markdown('<div class="notice danger">⚠️ This plant may be poisonous. Do not use it without expert advice.</div>',
                            unsafe_allow_html=True)
            if conf < threshold:
                st.markdown('<div class="notice warn">Low confidence, the result may be wrong. Please try another clear photo.</div>',
                            unsafe_allow_html=True)
            st.markdown(top_bars(top, probs), unsafe_allow_html=True)
            if st.button("➕ Save to herbarium", type="primary"):
                thumb = img.convert("RGB")
                thumb.thumbnail((220, 220))
                st.session_state.log.append({
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Plant": d["common"], "Scientific Name": d["scientific"],
                    "Family": d["family"], "Confidence": f"{conf:.1f}%", "thumb": thumb})
                st.success("Saved! Check the 'My Herbarium' tab.")

# ---- My Herbarium ----
with tab2:
    log = st.session_state.log
    if not log:
        st.markdown('<div class="card"><div class="plant-name">Your herbarium is empty</div>'
                    '<div class="row-v">Upload a photo in the Identify tab and press "Save to herbarium".</div></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="muted">' + str(len(log)) + ' entries</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, e in enumerate(log):
            with cols[i % 4]:
                st.image(e["thumb"], caption=e["Plant"] + " (" + e["Confidence"] + ")")
        df = pd.DataFrame([{k: v for k, v in e.items() if k != "thumb"} for e in log])
        st.dataframe(df, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ CSV download", df.to_csv(index=False).encode("utf-8"),
                               file_name="my_herbarium.csv", mime="text/csv")
        with c2:
            if st.button("🗑️ Clear herbarium"):
                st.session_state.log = []
                st.rerun()

# ---- All Plants ----
with tab3:
    q = st.text_input("🔎 Search", placeholder="Type a name, family or use (e.g. tulsi, Lamiaceae, cough)")
    fams = st.multiselect("Family filter", sorted({d["family"] for d in INFO}))
    items = INFO
    if q:
        items = [d for d in items if q.lower() in " ".join(d.values()).lower()]
    if fams:
        items = [d for d in items if d["family"] in fams]
    st.markdown('<div class="muted" style="margin:6px 0 10px 0;">' + str(len(items)) + ' plants</div>', unsafe_allow_html=True)
    st.markdown(plant_grid(items) if items else '<div class="card">No plants found.</div>',
                unsafe_allow_html=True)
    st.caption("⚠️ " + DISCLAIMER)

# ---- Model ----
with tab4:
    st.markdown('<div class="card"><div class="plant-name">Model details</div>'
                '<div class="row-v">MobileNetV2 (ImageNet pretrained) + GlobalAveragePooling + Dropout + Dense(30, softmax). '
                'The top layer was trained first, then the last layers were fine-tuned. Input size 224×224.</div></div>', unsafe_allow_html=True)
    shown = False
    for fname, title in (("training_curves.png", "Training curves"), ("confusion_matrix.png", "Confusion Matrix (test set)")):
        if os.path.exists(fname):
            st.subheader(title)
            st.image(fname)
            shown = True
    if not shown:
        st.info("Download training_curves.png and confusion_matrix.png from Colab and place them in the app folder to see them here.")
    st.caption("Take the accuracy and classification report from your Colab notebook's evaluation output.")

# ---- About ----
with tab5:
    st.markdown(
        '<div class="card"><div class="plant-name">What is a Digital Herbarium?</div>'
        '<div class="row-v">A herbarium is a collection of preserved plant specimens. This app identifies a plant from a leaf photo '
        'and creates its herbarium entry (scientific name, family, habitat, uses).</div></div>'
        '<div class="steps">'
        '<div class="card"><div class="step-num">1</div><b>Photo</b><div class="row-v">Take a leaf photo or upload one.</div></div>'
        '<div class="card"><div class="step-num">2</div><b>AI identification</b><div class="row-v">The MobileNetV2 model matches it against 30 species.</div></div>'
        '<div class="card"><div class="step-num">3</div><b>Herbarium entry</b><div class="row-v">View the details and save them to your herbarium.</div></div>'
        '</div>'
        '<div class="card" style="margin-top:14px;"><div class="muted">Developed by</div>'
        '<div class="plant-name">NanStack</div>'
        '<div class="row-v">Web · Mobile · AI · SaaS. Turning concepts into digital products.</div>'
        '<a class="btn-link" href="' + NAN_URL + '" target="_blank" rel="noopener noreferrer">Visit NanStack.tech</a>'
        '<a class="btn-link alt" href="' + NAN_CONTACT + '" target="_blank" rel="noopener noreferrer">Contact</a></div>'
        '<div class="notice warn">⚠️ ' + DISCLAIMER + ' Dataset: MedLeaves (Kaggle, Machine Intelligence Research Laboratory).</div>',
        unsafe_allow_html=True)

footer()