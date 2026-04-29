"""
ui/streamlit_app.py
--------------------
Neuro AI — Streamlit Chatbot Interface

BUG FIX: Added file upload widget in sidebar.
This is the most reliable way to use Neuro AI — upload your own
CSV/Excel/JSON file directly instead of relying on web search.

Previous bug: Every run saved as datasets/dataset.csv (same filename).
When a new search failed, the pipeline loaded the OLD dataset.csv from
a previous run — so "Predict churn" would train on the chess dataset.

Fix: dataset_file path from upload is passed to run_pipeline().
     Search is now a FALLBACK when no file/URL is provided.
"""

import sys
import io
import traceback
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from loguru import logger

st.set_page_config(
    page_title="Neuro AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');
:root {
    --bg:#07080f; --bg2:#0e0f1a; --bg3:#151626; --card:#1a1b2e;
    --border:#252640; --green:#00ffaa; --blue:#4d9eff; --purple:#a78bfa;
    --orange:#fb923c; --red:#f87171; --yellow:#fbbf24;
    --text:#e2e4f0; --muted:#5a5c7a; --user-bg:#1a2540; --bot-bg:#141525;
}
html,body,.stApp { background:var(--bg) !important; font-family:'DM Sans',sans-serif; color:var(--text); }
#MainMenu,footer,header,.stDeployButton { visibility:hidden !important; display:none !important; }
.block-container { padding:1.5rem 2rem 2rem 2rem !important; max-width:1200px; }
[data-testid="stSidebar"] { background:var(--bg2) !important; border-right:1px solid var(--border); }
[data-testid="stSidebar"] label { color:var(--text) !important; font-size:13px !important; }

.stTextInput>div>div>input,
.stTextArea>div>div>textarea {
    background:var(--bg3) !important; border:1px solid var(--border) !important;
    border-radius:10px !important; color:var(--text) !important;
    font-family:'DM Sans',sans-serif !important; font-size:15px !important;
    padding:12px 16px !important;
}
.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus {
    border-color:var(--green) !important;
    box-shadow:0 0 0 2px rgba(0,255,170,0.12) !important;
}

.stButton>button {
    background:linear-gradient(135deg,#00ffaa22,#4d9eff22) !important;
    border:1px solid var(--green) !important; color:var(--green) !important;
    border-radius:10px !important; font-family:'Space Mono',monospace !important;
    font-size:13px !important; font-weight:700 !important;
    padding:10px 24px !important; transition:all 0.2s ease !important;
}
.stButton>button:hover {
    background:linear-gradient(135deg,#00ffaa44,#4d9eff44) !important;
    transform:translateY(-1px);
    box-shadow:0 4px 20px rgba(0,255,170,0.2) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background:var(--bg3) !important; border:1px dashed var(--blue) !important;
    border-radius:10px !important; padding:8px !important;
}
[data-testid="stFileUploader"] label { color:var(--blue) !important; }

.chat-row { display:flex; gap:12px; margin-bottom:20px; align-items:flex-start; }
.chat-row.user { flex-direction:row-reverse; }
.avatar { width:36px;height:36px;border-radius:50%;display:flex;align-items:center;
    justify-content:center;font-size:16px;flex-shrink:0;border:1px solid var(--border); }
.avatar.bot { background:linear-gradient(135deg,#00ffaa22,#4d9eff22); border-color:var(--green); }
.avatar.user { background:linear-gradient(135deg,#a78bfa22,#4d9eff22); border-color:var(--purple); }
.bubble { max-width:78%;padding:14px 18px;border-radius:14px;font-size:14.5px;
    line-height:1.65;border:1px solid var(--border); }
.bubble.bot  { background:var(--bot-bg); border-radius:4px 14px 14px 14px; }
.bubble.user { background:var(--user-bg); border-radius:14px 4px 14px 14px; color:#c4d4f0; }

.step-box { background:var(--card);border:1px solid var(--border);border-radius:12px;
    padding:12px 16px;margin:8px 0;font-size:13px;display:flex;align-items:center;gap:10px; }
.step-box.done    { border-color:var(--green); color:var(--green); }
.step-box.running { border-color:var(--blue);  color:var(--blue);  animation:pulse 1.5s infinite; }
.step-box.error   { border-color:var(--red);   color:var(--red); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

.metric-grid { display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
    gap:12px;margin:12px 0; }
.metric-card { background:var(--card);border:1px solid var(--border);border-radius:12px;
    padding:14px 16px;text-align:center; }
.metric-card .label { font-size:11px;color:var(--muted);text-transform:uppercase;
    letter-spacing:1px;margin-bottom:6px; }
.metric-card .value { font-size:22px;font-weight:700;font-family:'Space Mono',monospace;
    color:var(--green); }

.quality-bar-wrap { background:var(--bg3);border-radius:999px;height:8px;
    margin:6px 0;overflow:hidden; }
.quality-bar-fill { height:100%;border-radius:999px;transition:width 0.6s ease; }

.tag { display:inline-block;padding:3px 10px;border-radius:999px;font-size:11px;
    font-weight:600;margin:2px;letter-spacing:0.3px; }
.tag.green  { background:#00ffaa22;color:var(--green);border:1px solid var(--green); }
.tag.blue   { background:#4d9eff22;color:var(--blue); border:1px solid var(--blue); }
.tag.orange { background:#fb923c22;color:var(--orange);border:1px solid var(--orange); }
.tag.purple { background:#a78bfa22;color:var(--purple);border:1px solid var(--purple); }

.section-title { font-family:'Space Mono',monospace;font-size:11px;font-weight:700;
    color:var(--muted);text-transform:uppercase;letter-spacing:2px;margin:16px 0 8px 0; }
.code-block { background:var(--bg3);border:1px solid var(--border);border-radius:8px;
    padding:12px 16px;font-family:'Space Mono',monospace;font-size:12px;color:var(--green);
    overflow-x:auto;margin:8px 0;white-space:pre-wrap;word-break:break-all; }
.log-box { background:var(--bg3);border:1px solid var(--border);border-radius:10px;
    padding:12px 14px;font-family:'Space Mono',monospace;font-size:11px;color:#7c7e9a;
    max-height:200px;overflow-y:auto;margin-top:8px;line-height:1.8; }

.welcome-hero { text-align:center;padding:48px 20px 36px 20px; }
.welcome-hero h1 { font-family:'Space Mono',monospace;font-size:42px;font-weight:700;
    background:linear-gradient(135deg,var(--green),var(--blue));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:12px; }
.welcome-hero p { color:var(--muted);font-size:16px;max-width:520px;margin:0 auto 28px auto;line-height:1.6; }

.upload-notice { background:linear-gradient(135deg,#4d9eff11,#00ffaa11);
    border:1px solid var(--blue);border-radius:12px;padding:14px 18px;
    font-size:13px;color:var(--blue);margin:16px 0; }
hr { border-color:var(--border) !important; margin:20px 0 !important; }
[data-baseweb="select"]>div { background:var(--bg3) !important; border-color:var(--border) !important; }
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "messages":      [],
        "stage":         "welcome",
        "run_result":    None,
        "run_logs":      [],
        "dataset_url":   "",
        "target_col":    "",
        "mode":          "auto",
        "tune":          False,
        "model_choice":  "auto",
        "uploaded_file_path": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    st.session_state["messages"] = [
        m for m in st.session_state["messages"]
        if isinstance(m, dict) and "role" in m
    ]

init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px 0">
      <span style="font-family:'Space Mono',monospace;font-size:18px;font-weight:700;
        background:linear-gradient(135deg,#00ffaa,#4d9eff);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        🧠 Neuro AI
      </span>
      <div style="font-size:11px;color:#5a5c7a;margin-top:4px;letter-spacing:1px;">
        AUTONOMOUS ML ENGINEER
      </div>
    </div>
    <hr style="border-color:#252640;margin:8px 0 16px 0"/>
    """, unsafe_allow_html=True)

    # ── FILE UPLOAD (PRIMARY — most reliable) ─────────────────────────────────
    st.markdown(
        '<div class="section-title">📁 Upload Your Dataset (Recommended)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="upload-notice">'
        '⚡ <strong>Upload first</strong> — this is the most reliable way. '
        'Uploading your own CSV/Excel file bypasses all web search and '
        'download issues completely.'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload CSV / Excel / JSON",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        help="Upload your dataset file directly. This bypasses web search entirely.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Save uploaded file to datasets/ folder with original name
        from config.settings import DATASETS_DIR
        save_path = DATASETS_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["uploaded_file_path"] = str(save_path)
        st.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size/1024:.1f} KB)")
    else:
        st.session_state["uploaded_file_path"] = None

    st.markdown('<hr style="border-color:#252640;margin:12px 0"/>', unsafe_allow_html=True)

    # ── PIPELINE SETTINGS ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⚙️ Pipeline Settings</div>', unsafe_allow_html=True)

    mode = st.selectbox(
        "Training Mode",
        ["auto (FLAML AutoML)", "manual (ModelTrainer)"],
        index=0,
        help="Auto: FLAML tries all model types. Manual: trains specific sklearn models.",
    )
    st.session_state["mode"] = "auto" if "auto" in mode else "manual"

    model_choice = st.selectbox(
        "Model (manual mode only)",
        ["auto", "XGBoost", "LightGBM", "CatBoost", "RandomForest",
         "LogisticRegression", "LinearRegression", "Ridge"],
        index=0,
    )
    st.session_state["model_choice"] = model_choice

    st.session_state["tune"] = st.toggle(
        "Optuna Tuning",
        value=False,
        help="Runs Optuna hyperparameter search after training. Slower but better.",
    )

    st.markdown('<div class="section-title">🔗 Or Use a URL</div>', unsafe_allow_html=True)
    dataset_url = st.text_input(
        "Direct Dataset URL",
        placeholder="https://raw.githubusercontent.com/.../data.csv",
        help="Paste a direct CSV link. Only used if no file is uploaded above.",
        label_visibility="collapsed",
    )
    st.session_state["dataset_url"] = dataset_url

    target_col = st.text_input(
        "Target Column (optional)",
        placeholder="e.g.  Survived  or  price",
        help="Column to predict. Leave blank for auto-detection.",
        label_visibility="collapsed",
    )
    st.session_state["target_col"] = target_col

    st.markdown('<hr style="border-color:#252640;margin:16px 0"/>', unsafe_allow_html=True)

    # Show what data source will be used
    if st.session_state.get("uploaded_file_path"):
        fname = Path(st.session_state["uploaded_file_path"]).name
        st.markdown(
            f'<div style="font-size:12px;color:#00ffaa;padding:8px;'
            f'background:#00ffaa11;border-radius:8px;border:1px solid #00ffaa33">'
            f'✅ Will use uploaded file:<br><strong>{fname}</strong></div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.get("dataset_url"):
        st.markdown(
            f'<div style="font-size:12px;color:#4d9eff;padding:8px;'
            f'background:#4d9eff11;border-radius:8px;border:1px solid #4d9eff33">'
            f'🔗 Will use URL</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:12px;color:#fbbf24;padding:8px;'
            'background:#fbbf2411;border-radius:8px;border:1px solid #fbbf2433">'
            '⚠️ Will search for dataset automatically<br>'
            '<span style="color:#5a5c7a">(less reliable — recommend uploading)</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr style="border-color:#252640;margin:12px 0"/>', unsafe_allow_html=True)
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state["messages"]   = []
        st.session_state["stage"]      = "welcome"
        st.session_state["run_result"] = None
        st.session_state["run_logs"]   = []
        st.rerun()


# ── Message renderer ──────────────────────────────────────────────────────────
def render_message(msg):
    if not isinstance(msg, dict):
        return
    role    = msg.get("role", "bot")
    content = msg.get("content") or ""
    mtype   = msg.get("type") or "text"
    if not role:
        return

    if role == "user":
        st.markdown(f"""
        <div class="chat-row user">
          <div class="avatar user">👤</div>
          <div class="bubble user">{content}</div>
        </div>""", unsafe_allow_html=True)

    elif role == "bot":
        if mtype == "text":
            st.markdown(f"""
            <div class="chat-row">
              <div class="avatar bot">🧠</div>
              <div class="bubble bot">{content}</div>
            </div>""", unsafe_allow_html=True)

        elif mtype == "steps":
            steps = msg.get("steps", [])
            st.markdown(
                '<div class="chat-row">'
                '<div class="avatar bot">🧠</div>'
                '<div style="flex:1">',
                unsafe_allow_html=True,
            )
            for s in steps:
                icon = "✅" if s["status"] == "done" else (
                    "🔄" if s["status"] == "running" else "❌"
                )
                st.markdown(
                    f'<div class="step-box {s["status"]}">'
                    f'{icon} &nbsp; {s["label"]}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div></div>', unsafe_allow_html=True)

        elif mtype == "results":
            _render_results(msg)


def _render_results(msg):
    r         = msg.get("data", {})
    metrics   = r.get("metrics", {})
    best      = r.get("best_model_name", "—")
    quality   = r.get("quality_score", 0)
    mode_used = r.get("mode_used", "—")
    task      = r.get("task_type", "—")
    target    = r.get("target_column", "—")
    dataset   = r.get("dataset_used", "—")
    rows      = r.get("rows_trained_on", "—")
    qcolor    = "#00ffaa" if quality >= 70 else ("#fbbf24" if quality >= 40 else "#f87171")

    st.markdown('<div class="chat-row"><div class="avatar bot">🧠</div><div style="flex:1">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="bubble bot">
      <div style="font-family:'Space Mono',monospace;font-size:13px;color:#00ffaa;margin-bottom:12px;">
        ✅ Training Complete
      </div>
      <div class="section-title">Best Model</div>
      <span class="tag green">{best}</span>
      <span class="tag blue">{mode_used} mode</span>
      <span class="tag orange">{task}</span>

      <div class="section-title" style="margin-top:14px">Dataset Info</div>
      <div style="font-size:12px;color:var(--muted);margin-bottom:8px">
        Target: <strong style="color:var(--text)">{target}</strong> &nbsp;|&nbsp;
        Rows trained: <strong style="color:var(--text)">{rows:,}</strong> &nbsp;|&nbsp;
        File: <strong style="color:var(--text)">{Path(dataset).name if dataset and dataset != '—' else '—'}</strong>
      </div>

      <div class="section-title">Dataset Quality Score</div>
      <div style="font-size:20px;font-weight:700;font-family:'Space Mono',monospace;color:{qcolor}">
        {quality}/100
      </div>
      <div class="quality-bar-wrap">
        <div class="quality-bar-fill" style="width:{quality}%;background:{qcolor}"></div>
      </div>

      <div class="section-title" style="margin-top:14px">Metrics</div>
      <div class="metric-grid">
    """, unsafe_allow_html=True)

    icons = {"accuracy":"🎯","f1_score":"⚡","precision":"🔍","recall":"📡","r2":"📈","rmse":"📉","mae":"📊"}
    for k, v in metrics.items():
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">{icons.get(k,'📌')} {k.replace('_',' ').upper()}</div>
          <div class="value">{v}</div>
        </div>""", unsafe_allow_html=True)

    report       = r.get("report_path", "")
    model        = r.get("model_path", "")
    predict_path = r.get("predict_script_path", "")
    predict_code = r.get("predict_code", "")
    features     = r.get("feature_columns", [])
    feat_preview = ", ".join(features[:8]) + (" ..." if len(features) > 8 else "")

    st.markdown(f"""
      </div>
      <div class="section-title" style="margin-top:14px">Exported Files</div>
      <div class="code-block">Report    : {report}
Model     : {model}
predict.py: {predict_path}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # ── Show predict.py code in expandable section ────────────────────────────
    st.markdown("""
    <div style="margin-top:20px;background:#1a1b2e;border:1px solid #252640;
      border-radius:14px;padding:20px">
      <div style="font-family:'Space Mono',monospace;font-size:13px;
        color:#00ffaa;margin-bottom:6px;">
        🚀 How to use your trained model
      </div>
      <div style="font-size:12px;color:#5a5c7a;margin-bottom:14px">
        Copy and run this script to make predictions with your trained model.
        Replace the sample values with your own data.
      </div>
    """, unsafe_allow_html=True)

    if predict_code:
        st.code(predict_code, language="python")

    st.markdown(f"""
      <div style="margin-top:12px;font-size:12px;color:#5a5c7a">
        <strong style="color:#4d9eff">Features used ({len(features)} total):</strong>
        <span style="color:#a78bfa">{feat_preview}</span>
      </div>
      <div style="margin-top:8px;font-size:12px;color:#5a5c7a">
        Run it: <code style="color:#00ffaa">python models/predict.py</code>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Pipeline runner ───────────────────────────────────────────────────────────
def run_pipeline_captured(idea, dataset_file, dataset_url, target_col, mode, model_choice, tune):
    logs  = []
    error = None
    result = None
    log_id = logger.add(lambda msg: logs.append(msg.strip()), level="INFO")
    try:
        from main import run_pipeline
        result = run_pipeline(
            user_idea         = idea,
            dataset_file      = Path(dataset_file) if dataset_file else None,
            dataset_url       = dataset_url or None,
            user_model_choice = None if model_choice == "auto" else model_choice,
            target_column     = target_col or None,
            tune              = tune,
            mode              = mode,
        )
    except Exception:
        error = traceback.format_exc()
        logs.append(f"ERROR: {error[:200]}")
    finally:
        logger.remove(log_id)
    return result, logs, error


# ── Welcome screen ────────────────────────────────────────────────────────────
def show_welcome():
    st.markdown("""
    <div class="welcome-hero">
      <h1>🧠 Neuro AI</h1>
      <p>Your autonomous ML engineer. Describe your idea — or upload a dataset and I'll train the best model and give you results.</p>
    </div>
    """, unsafe_allow_html=True)

    # Show upload status prominently
    if st.session_state.get("uploaded_file_path"):
        fname = Path(st.session_state["uploaded_file_path"]).name
        st.markdown(
            f'<div style="text-align:center;margin-bottom:20px">'
            f'<span style="background:#00ffaa22;border:1px solid #00ffaa;'
            f'border-radius:999px;padding:8px 20px;font-size:13px;color:#00ffaa">'
            f'✅ Dataset ready: {fname}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align:center;margin-bottom:20px">'
            '<span style="background:#fbbf2411;border:1px solid #fbbf24;'
            'border-radius:999px;padding:8px 20px;font-size:13px;color:#fbbf24">'
            '⚠️ No dataset uploaded — upload one in the sidebar for best results</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="text-align:center;margin-bottom:8px">'
        '<span style="font-size:12px;color:#5a5c7a;letter-spacing:2px;text-transform:uppercase">'
        'Try an example</span></div>',
        unsafe_allow_html=True,
    )

    examples = [
        "Predict customer churn for a telecom company",
        "Classify whether a patient has diabetes",
        "Detect fraudulent credit card transactions",
        "Predict house prices based on features",
        "Classify emails as spam or not spam",
        "Predict employee salary from experience",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        with cols[i % 3]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                _start_chat(ex)


def _start_chat(idea: str):
    st.session_state["stage"] = "chatting"
    has_file = bool(st.session_state.get("uploaded_file_path"))
    has_url  = bool(st.session_state.get("dataset_url"))
    mode     = st.session_state["mode"]

    if has_file:
        fname = Path(st.session_state["uploaded_file_path"]).name
        data_source = f"your uploaded file <strong>{fname}</strong>"
    elif has_url:
        data_source = "the URL you provided"
    else:
        data_source = "a dataset I'll search for automatically"

    st.session_state["messages"].append({
        "role": "user", "content": idea, "type": "text"
    })
    st.session_state["messages"].append({
        "role": "bot", "type": "text",
        "content": (
            f"Got it! Working on: <strong>{idea}</strong><br><br>"
            f"📂 Data source: {data_source}<br>"
            f"🤖 Training mode: <strong>{'FLAML AutoML' if mode == 'auto' else 'ModelTrainer'}</strong><br><br>"
            f"Click <strong>🚀 Run Pipeline</strong> to start!"
        ),
    })
    st.rerun()


# ── Chat view ─────────────────────────────────────────────────────────────────
def show_chat():
    for msg in st.session_state["messages"]:
        render_message(msg)

    stage = st.session_state["stage"]

    if stage == "chatting":
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Message", placeholder="Type a new idea...",
                label_visibility="collapsed", key="chat_input",
            )
        with col2:
            send = st.button("Send", use_container_width=True)

        _, run_col, _ = st.columns([1, 2, 1])
        with run_col:
            run_btn = st.button("🚀  Run Pipeline", use_container_width=True, key="run_btn")

        if send and user_input.strip():
            st.session_state["messages"] = []
            _start_chat(user_input.strip())

        if run_btn:
            idea = ""
            for m in reversed(st.session_state["messages"]):
                if m.get("role") == "user":
                    idea = m.get("content", "")
                    break
            if idea:
                _launch_pipeline(idea)

    elif stage == "running":
        st.markdown("""
        <div style="text-align:center;padding:24px;color:#4d9eff">
          <div style="font-family:'Space Mono',monospace;font-size:14px;animation:pulse 1.5s infinite">
            ⚙️ Pipeline running — please wait (2-5 minutes)...
          </div>
          <div style="font-size:12px;color:#5a5c7a;margin-top:12px;line-height:2">
            ✅ This is normal — the pipeline is working<br>
            🔄 Preprocessing large datasets can take 1-2 min<br>
            🤖 FLAML model search takes up to 2 min<br>
            📊 Do NOT refresh the page — results will appear here
          </div>
        </div>""", unsafe_allow_html=True)
        st.spinner("Training in progress...")

    elif stage in ("done", "error"):
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([4, 1])
        with col1:
            new_idea = st.text_input(
                "New idea", placeholder="Try another ML idea...",
                label_visibility="collapsed", key="new_idea_input",
            )
        with col2:
            if st.button("New Run", use_container_width=True):
                if new_idea.strip():
                    st.session_state["messages"]   = []
                    st.session_state["stage"]      = "welcome"
                    st.session_state["run_result"] = None
                    st.session_state["run_logs"]   = []
                    _start_chat(new_idea.strip())


def _launch_pipeline(idea: str):
    st.session_state["stage"] = "running"
    steps = [
        {"label": "Parsing idea",                    "status": "running"},
        {"label": "Getting dataset",                 "status": "pending"},
        {"label": "Validating & analysing",          "status": "pending"},
        {"label": "Preprocessing (split-first)",     "status": "pending"},
        {"label": "Training models",                 "status": "pending"},
        {"label": "Evaluating & selecting best",     "status": "pending"},
        {"label": "Exporting model + report",        "status": "pending"},
    ]
    st.session_state["messages"].append({
        "role": "bot", "type": "steps", "content": "", "steps": steps
    })

    result, logs, error = run_pipeline_captured(
        idea         = idea,
        dataset_file = st.session_state.get("uploaded_file_path"),
        dataset_url  = st.session_state.get("dataset_url"),
        target_col   = st.session_state.get("target_col"),
        mode         = st.session_state.get("mode", "auto"),
        model_choice = st.session_state.get("model_choice", "auto"),
        tune         = st.session_state.get("tune", False),
    )

    st.session_state["run_logs"] = logs

    if error:
        st.session_state["stage"] = "error"
        # Extract meaningful error message
        lines = [l for l in error.split("\n") if l.strip()]
        short_error = "\n".join(lines[-8:]) if len(lines) > 8 else error

        st.session_state["messages"].append({
            "role": "bot", "type": "text",
            "content": (
                f"❌ <strong>Pipeline failed</strong><br><br>"
                f"<span style='color:#f87171;font-size:12px;font-family:monospace'>"
                f"{short_error.replace(chr(10),'<br>')}"
                f"</span><br><br>"
                f"<strong>Quick fixes:</strong><br>"
                f"• Upload your dataset file using the sidebar<br>"
                f"• Set the target column name in the sidebar<br>"
                f"• Switch to <code>manual</code> mode if FLAML fails<br>"
                f"• Check your <code>.env</code> file has a valid API key<br>"
                f"• Make sure all dependencies are installed"
            ),
        })
    else:
        st.session_state["stage"]      = "done"
        st.session_state["run_result"] = result
        st.session_state["messages"].append({
            "role": "bot", "type": "results", "content": "", "data": result
        })
        if logs:
            log_text = "\n".join(l for l in logs[-15:] if l.strip())
            st.session_state["messages"].append({
                "role": "bot", "type": "text",
                "content": (
                    f"Done! Check <strong>models/</strong> folder for all exported files.<br>"
                    f"<div class='log-box'>{log_text}</div>"
                ),
            })

    st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
  padding:12px 0 16px 0;border-bottom:1px solid #252640;margin-bottom:24px">
  <div>
    <span style="font-family:'Space Mono',monospace;font-size:20px;font-weight:700;
      background:linear-gradient(135deg,#00ffaa,#4d9eff);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
      🧠 Neuro AI
    </span>
    <span style="font-size:11px;color:#5a5c7a;margin-left:12px;letter-spacing:2px;
      text-transform:uppercase;vertical-align:middle;">
      Autonomous ML Engineer
    </span>
  </div>
  <div style="font-size:11px;color:#5a5c7a;font-family:'Space Mono',monospace">v2.0</div>
</div>
""", unsafe_allow_html=True)

stage = st.session_state["stage"]

if stage == "welcome":
    show_welcome()
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([5, 1])
    with c1:
        idea_input = st.text_input(
            "Idea", placeholder="e.g.  Predict whether a loan will default",
            label_visibility="collapsed", key="welcome_input",
        )
    with c2:
        if st.button("Start", use_container_width=True) and idea_input.strip():
            _start_chat(idea_input.strip())
else:
    show_chat()
