"""
app_phase2.py — Spectrum-SLM Phase 2 Dashboard
================================================
Streamlit app for the NEW dataset model (5-class: BPSK/QPSK/8PSK/16QAM/DQPSK).

Differences from app.py:
  - n_mod_classes = 5 (adds DQPSK)
  - Loads checkpoint from checkpoints/phase2/slm_phase2_new_best.pt
  - Loads normalizer from checkpoints/phase2/normalizer_phase2.pkl
  - Shows metrics_phase2.json in Research tab
  - Dataset Explorer tab for browsing new dataset samples

Run: streamlit run app_phase2.py

Authors : Anjani, Ashish Joshi, Mayank
Guide   : Dr. Abhinandan S.P. | IIT Palakkad
Dated   : April 2026
"""

import os, sys, pickle, time, json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from spectrum_slm_model import SpectrumSLM
from spectrum_slm_dataset import generate_synthetic_psd, N_BINS
from config import (
    CKPT_PHASE2, CKPT_PHASE2_BEST, NORMALIZER_FILE, METRICS_FILE,
    MOD_NAMES_V2, MOD_COLORS_V2, N_MOD_CLASSES_V2, MOD_MAP_V2,
    PHASE2_DATA_DIR,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spectrum-SLM Phase 2 | New Dataset",
    page_icon="📡", layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html,body,[class*="css"]{font-family:'Inter',sans-serif;}
  .main{background:#0d1117;}
  .metric-card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:16px;text-align:center;}
  .metric-value{font-size:2em;font-weight:700;color:#58a6ff;}
  .metric-label{font-size:0.8em;color:#8b949e;margin-top:4px;}
  .phase2-badge{display:inline-block;background:linear-gradient(135deg,#533483,#0f3460);
    border-radius:20px;padding:4px 14px;font-size:0.75em;color:#d2a8ff;margin:2px;}
  .stButton button{background:linear-gradient(135deg,#238636,#2ea043);color:white;
    border:none;border-radius:8px;padding:8px 20px;font-weight:600;transition:all 0.2s;}
  .stButton button:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(46,160,67,0.4);}
  .header-banner{background:linear-gradient(135deg,#0f3460,#16213e,#533483);
    padding:24px 32px;border-radius:16px;margin-bottom:20px;border:1px solid #30363d;}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1 style="color:#58a6ff;margin:0;font-size:2em;">📡 Spectrum-SLM — Phase 2</h1>
  <p style="color:#c9d1d9;margin:6px 0 10px;">New SDR Dataset · 5-Class Modulation Recognition</p>
  <span class="phase2-badge">BPSK</span><span class="phase2-badge">QPSK</span>
  <span class="phase2-badge">8PSK</span><span class="phase2-badge">16QAM</span>
  <span class="phase2-badge" style="background:linear-gradient(135deg,#533483,#21262d);">DQPSK ★ New</span>
  &nbsp;&nbsp;
  <span class="phase2-badge" style="background:#161b22;color:#8b949e;">IIT Palakkad</span>
  <span class="phase2-badge" style="background:#161b22;color:#8b949e;">~1M Parameters</span>
</div>
""", unsafe_allow_html=True)


# ─── Model + Normalizer loader ────────────────────────────────────────────────
@st.cache_resource
def load_model_and_normalizer(ckpt_path: str, norm_path: str):
    model = SpectrumSLM(
        n_bins=N_BINS, patch_size=8, d_model=128,
        nhead=4, num_layers=4, dim_feedforward=512,
        dropout=0.1, n_mod_classes=N_MOD_CLASSES_V2,
    )
    is_trained = False
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck.get("model", ck))
        is_trained = True
    model.eval()

    scaler = None
    if norm_path and os.path.exists(norm_path):
        with open(norm_path, "rb") as f:
            scaler = pickle.load(f)

    return model, scaler, is_trained


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Phase 2 Configuration")
    ckpt_path = st.text_input(
        "Checkpoint path",
        value=os.path.join(CKPT_PHASE2, CKPT_PHASE2_BEST),
    )
    norm_path = st.text_input(
        "Normalizer path",
        value=os.path.join(CKPT_PHASE2, NORMALIZER_FILE),
    )
    model, scaler, is_trained = load_model_and_normalizer(ckpt_path, norm_path)

    if is_trained:
        st.success("✅ Phase 2 model loaded (5-class)")
    else:
        st.warning("⚠️ Demo mode — untrained weights")
    if scaler:
        st.success("✅ Normalizer loaded")
    else:
        st.info("ℹ️ No normalizer — using raw PSD values")

    st.markdown("---")
    st.markdown("### 🔧 Model Info")
    n_params = sum(p.numel() for p in model.parameters())
    st.metric("Parameters",   f"{n_params/1e6:.2f}M")
    st.metric("Mod classes",  f"{N_MOD_CLASSES_V2} (with DQPSK)")
    st.metric("Patch tokens", "22 + CLS")
    st.metric("Heads/Layers", "4 / 4")
    st.markdown("---")
    st.caption("Anjani · Ashish Joshi · Mayank\nGuide: Dr. Abhinandan S.P.\nApril 2026")


# ─── Inference helper ─────────────────────────────────────────────────────────
def run_inference(psd_vec: np.ndarray) -> dict:
    p = psd_vec.reshape(1, -1).astype(np.float32)
    if scaler is not None:
        p = scaler.transform(p).astype(np.float32)
    t = torch.tensor(p, dtype=torch.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(t)
    lat = (time.perf_counter() - t0) * 1000
    pu_p  = torch.softmax(out["pu_logits"],  dim=1)[0].numpy()
    mod_p = torch.softmax(out["mod_logits"], dim=1)[0].numpy()
    snr   = float(np.clip(out["snr_pred"][0].item(), 0, 30))
    return {
        "pu_prob":    float(pu_p[1]),
        "pu_present": bool(pu_p[1] > 0.5),
        "mod_probs":  mod_p.tolist(),
        "mod_pred":   int(np.argmax(mod_p)),
        "snr_db":     snr,
        "gen_psd":    out["gen_pred"][0].numpy().tolist(),
        "latency_ms": lat,
    }


def psd_fig(psd: np.ndarray, gen: np.ndarray = None, title: str = "PSD") -> go.Figure:
    freq = np.linspace(2380, 2420, N_BINS)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq, y=psd, mode="lines", name="Input PSD",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.07)"))
    if gen is not None:
        fig.add_trace(go.Scatter(x=freq, y=gen, mode="lines",
            name="Predicted Next PSD", line=dict(color="#3fb950", width=2, dash="dash")))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#c9d1d9")),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=280,
        xaxis=dict(title="Frequency (MHz)", color="#8b949e", gridcolor="#21262d"),
        yaxis=dict(title="Power (norm.)", color="#8b949e", gridcolor="#21262d"),
        legend=dict(bgcolor="#161b22", font=dict(color="#c9d1d9")),
        font=dict(family="Inter"), margin=dict(l=40,r=20,t=40,b=40),
    )
    return fig


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_scan, tab_batch, tab_explore, tab_research = st.tabs([
    "🔭 Single Scan", "📊 Batch Analysis", "🗃️ Dataset Explorer", "📈 Research"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — Single Scan
# ════════════════════════════════════════════════════════════
with tab_scan:
    st.markdown("### 🔭 Single Spectrum Scan")
    col_ctrl, col_res = st.columns([1, 2])
    with col_ctrl:
        snr_t  = st.slider("Target SNR (dB)", 3.0, 20.0, 10.0, 0.5)
        pu_sel = st.selectbox("PU Status", ["Present (PU=1)", "Absent (PU=0)"])
        mod_sel= st.selectbox("Modulation", MOD_NAMES_V2)
        is_pu  = pu_sel == "Present (PU=1)"
        mod_id = MOD_NAMES_V2.index(mod_sel)

        if st.button("🔍 Run Inference", use_container_width=True):
            freqs = np.linspace(-1, 1, N_BINS)
            widths = [0.20, 0.25, 0.30, 0.35, 0.22]   # per-mod bandwidth
            psd = np.random.randn(N_BINS).astype(np.float32) * 1.5 - 22.0
            if is_pu:
                psd += (snr_t * 0.8) * np.exp(
                    -freqs**2 / (2 * widths[mod_id]**2)).astype(np.float32)
            st.session_state.p2_psd = psd
            st.session_state.p2_res = run_inference(psd)
            st.session_state.p2_true_pu  = is_pu
            st.session_state.p2_true_mod = mod_id

    with col_res:
        if "p2_res" in st.session_state:
            res = st.session_state.p2_res
            psd = st.session_state.p2_psd
            gen = np.array(res["gen_psd"]) * np.std(psd) + np.mean(psd)
            st.plotly_chart(psd_fig(psd, gen, "Input PSD + Predicted Next-PSD"),
                            use_container_width=True)

            m1,m2,m3,m4 = st.columns(4)
            pu_col = "#3fb950" if res["pu_present"] else "#f85149"
            m1.markdown(f"""<div class="metric-card">
              <div class="metric-value" style="color:{pu_col};">
                {'✅ YES' if res['pu_present'] else '⛔ NO'}
              </div>
              <div class="metric-label">PU Present ({res['pu_prob']*100:.1f}%)</div>
            </div>""", unsafe_allow_html=True)
            m2.markdown(f"""<div class="metric-card">
              <div class="metric-value" style="color:#ffa657;">
                {MOD_NAMES_V2[res['mod_pred']]}
              </div>
              <div class="metric-label">Modulation ({res['mod_probs'][res['mod_pred']]*100:.1f}%)</div>
            </div>""", unsafe_allow_html=True)
            m3.markdown(f"""<div class="metric-card">
              <div class="metric-value">{res['snr_db']:.1f} dB</div>
              <div class="metric-label">Estimated SNR</div>
            </div>""", unsafe_allow_html=True)
            m4.markdown(f"""<div class="metric-card">
              <div class="metric-value" style="color:#58a6ff;">{res['latency_ms']:.1f} ms</div>
              <div class="metric-label">Inference Latency</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("#### Modulation Probabilities (5-class)")
            bar = go.Figure(go.Bar(
                x=MOD_NAMES_V2, y=[p*100 for p in res["mod_probs"]],
                marker_color=MOD_COLORS_V2,
                text=[f"{p*100:.1f}%" for p in res["mod_probs"]],
                textposition="outside",
            ))
            bar.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                height=220, margin=dict(l=20,r=20,t=10,b=20),
                yaxis=dict(range=[0,115], color="#8b949e", gridcolor="#21262d"),
                xaxis=dict(color="#8b949e"),
                font=dict(color="#c9d1d9", family="Inter"),
            )
            st.plotly_chart(bar, use_container_width=True)

            # Ground truth comparison
            if st.session_state.get("p2_true_pu") is not None:
                true_mod = st.session_state.p2_true_mod
                pu_match  = res["pu_present"] == st.session_state.p2_true_pu
                mod_match = res["mod_pred"] == true_mod
                st.markdown(
                    f"**Ground truth:** PU={'Present' if st.session_state.p2_true_pu else 'Absent'}  |  "
                    f"Mod={MOD_NAMES_V2[true_mod]}  |  "
                    f"PU {'✅' if pu_match else '❌'}  Mod {'✅' if mod_match else '❌'}"
                )
        else:
            st.info("Configure input on the left and click **Run Inference**.")


# ════════════════════════════════════════════════════════════
# TAB 2 — Batch Analysis
# ════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### 📊 Batch Spectrum Analysis (5-class)")
    from sklearn.metrics import accuracy_score, mean_absolute_error
    n_samples = st.slider("Synthetic samples", 100, 2000, 500, 100)
    if st.button("▶ Run Batch Analysis", use_container_width=False):
        with st.spinner("Running batch inference …"):
            rng   = np.random.default_rng(42)
            psds_ = []; pu_ = []; mod_ = []; snr_ = []
            freqs = np.linspace(-1, 1, N_BINS)
            widths = [0.20, 0.25, 0.30, 0.35, 0.22]
            for _ in range(n_samples):
                pu  = rng.integers(0, 2)
                mod = rng.integers(0, N_MOD_CLASSES_V2)
                snr = rng.uniform(3, 20) if pu else rng.uniform(3, 8)
                p   = (rng.standard_normal(N_BINS) * 1.5 - 22.0).astype(np.float32)
                if pu:
                    p += (snr * 0.8) * np.exp(-freqs**2/(2*widths[mod]**2)).astype(np.float32)
                psds_.append(p); pu_.append(int(pu)); mod_.append(int(mod)); snr_.append(float(snr))

            pu_pred = []; mod_pred = []; snr_pred = []
            t0 = time.perf_counter()
            for p in psds_:
                r = run_inference(p)
                pu_pred.append(int(r["pu_present"]))
                mod_pred.append(r["mod_pred"])
                snr_pred.append(r["snr_db"])
            total_ms = (time.perf_counter() - t0) * 1000

        pu_a  = accuracy_score(pu_, pu_pred)
        mod_a = accuracy_score(mod_, mod_pred)
        snr_m = mean_absolute_error(snr_, snr_pred)

        r1,r2,r3,r4 = st.columns(4)
        r1.metric("PU Accuracy",  f"{pu_a*100:.2f}%")
        r2.metric("Mod Accuracy", f"{mod_a*100:.2f}%")
        r3.metric("SNR MAE",      f"{snr_m:.2f} dB")
        r4.metric("Throughput",   f"{n_samples/(total_ms/1000):.0f} samp/s")

        # Per-SNR PU accuracy
        st.markdown("#### PU Detection Accuracy vs SNR")
        snr_arr = np.array(snr_); pu_arr = np.array(pu_); pp_arr = np.array(pu_pred)
        bins = list(range(3,22,2)); accs = []
        for b in bins:
            mask = (snr_arr>=b-1)&(snr_arr<b+1)
            accs.append(accuracy_score(pu_arr[mask],pp_arr[mask])*100 if mask.sum()>3 else None)
        valid = [(b,a) for b,a in zip(bins,accs) if a is not None]
        if valid:
            vb,va = zip(*valid)
            snr_fig = go.Figure()
            snr_fig.add_trace(go.Scatter(x=list(vb),y=list(va),
                mode="lines+markers",name="PU Acc",
                line=dict(color="#58a6ff",width=2),marker=dict(size=8)))
            snr_fig.add_hline(y=90,line_dash="dash",line_color="#3fb950",
                              annotation_text="90% target")
            snr_fig.update_layout(
                xaxis_title="SNR (dB)", yaxis_title="PU Accuracy (%)",
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                yaxis=dict(range=[40,105],color="#8b949e",gridcolor="#21262d"),
                xaxis=dict(color="#8b949e"),
                height=280, font=dict(color="#c9d1d9",family="Inter"))
            st.plotly_chart(snr_fig, use_container_width=True)

        # Per-class modulation accuracy
        st.markdown("#### Per-Modulation Accuracy")
        mod_arr = np.array(mod_); mp_arr = np.array(mod_pred)
        mod_accs = []
        for i,n in enumerate(MOD_NAMES_V2):
            mask = (mod_arr == i)
            acc  = accuracy_score(mod_arr[mask], mp_arr[mask])*100 if mask.sum()>0 else 0
            mod_accs.append(acc)
        mbar = go.Figure(go.Bar(x=MOD_NAMES_V2, y=mod_accs, marker_color=MOD_COLORS_V2,
            text=[f"{a:.1f}%" for a in mod_accs], textposition="outside"))
        mbar.update_layout(plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",height=220,
            yaxis=dict(range=[0,115],color="#8b949e"),xaxis=dict(color="#8b949e"),
            font=dict(color="#c9d1d9",family="Inter"),margin=dict(l=20,r=20,t=10,b=20))
        st.plotly_chart(mbar, use_container_width=True)
    else:
        st.info("Click **Run Batch Analysis** to begin.")


# ════════════════════════════════════════════════════════════
# TAB 3 — Dataset Explorer
# ════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("### 🗃️ New Dataset Explorer")
    st.caption(f"Dataset root: `{PHASE2_DATA_DIR}`")

    col_e1, col_e2 = st.columns([1, 3])
    with col_e1:
        sel_mod = st.selectbox("Modulation", MOD_NAMES_V2, key="exp_mod")
        sel_sym = st.selectbox("Symbol dir", ["Symbol1","Symbol2","Symbol3"], key="exp_sym")
        if st.button("📂 Load Sample PSD"):
            import glob
            mod_folder_map = {"BPSK":"bpsk","QPSK":"qpsk","8PSK":"8psk",
                              "16QAM":"16qam","DQPSK":"dqpsk"}
            folder = mod_folder_map[sel_mod]
            csv_pattern = os.path.join(PHASE2_DATA_DIR, sel_sym, folder, "*.csv")
            csvs = glob.glob(csv_pattern)
            if csvs:
                df = pd.read_csv(csvs[0])
                st.session_state.exp_df = df
                st.session_state.exp_mod_name = sel_mod
                st.success(f"Loaded {len(df):,} rows from {os.path.basename(csvs[0])}")
            else:
                st.warning("No CSV found in that directory.")

    with col_e2:
        if "exp_df" in st.session_state:
            df = st.session_state.exp_df
            mod_name = st.session_state.exp_mod_name
            st.dataframe(df.head(20), use_container_width=True)
            if "Mean_PSD_dB" in df.columns:
                st.markdown(f"#### Mean PSD — {mod_name}")
                fig_e = go.Figure()
                fig_e.add_trace(go.Histogram(x=df["Mean_PSD_dB"],
                    marker_color=MOD_COLORS_V2[MOD_NAMES_V2.index(mod_name)],
                    name=mod_name, nbinsx=50))
                fig_e.update_layout(
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=250,
                    xaxis=dict(title="Mean PSD (dB)",color="#8b949e"),
                    yaxis=dict(title="Count",color="#8b949e"),
                    font=dict(color="#c9d1d9",family="Inter"),
                    margin=dict(l=40,r=20,t=20,b=40))
                st.plotly_chart(fig_e, use_container_width=True)
        else:
            st.info("Select a modulation + symbol dir and click **Load Sample PSD**.")


# ════════════════════════════════════════════════════════════
# TAB 4 — Research / Metrics
# ════════════════════════════════════════════════════════════
with tab_research:
    st.markdown("### 📈 Phase 2 Training Results")
    metrics_path = os.path.join(CKPT_PHASE2, METRICS_FILE)
    history_path = os.path.join(CKPT_PHASE2, "training_history_phase2.json")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            st.markdown("#### 📊 Test Metrics")
            mc1,mc2,mc3 = st.columns(3)
            mc1.metric("PU Accuracy",  f"{m.get('pu_accuracy',0)*100:.2f}%")
            mc2.metric("Mod Accuracy", f"{m.get('mod_accuracy',0)*100:.2f}%")
            mc3.metric("SNR MAE",      f"{m.get('snr_mae_db',0):.3f} dB")
            mc1.metric("PU F1",   f"{m.get('pu_f1',0):.4f}")
            mc2.metric("Mod F1",  f"{m.get('mod_f1_macro',0):.4f}")
            mc3.metric("PU AUC",  f"{m.get('pu_auc',0):.4f}")
            with st.expander("Full metrics JSON"):
                st.json(m)
        else:
            st.info("No metrics file yet — train the model first.\n\n"
                    f"Expected: `{metrics_path}`")

    with col_m2:
        if os.path.exists(history_path):
            with open(history_path) as f:
                hist = json.load(f)
            epochs_h = [h["epoch"] for h in hist]
            tr_h = [h.get("train_total", h.get("train",0)) for h in hist]
            vl_h = [h.get("val_total",  h.get("val",  0)) for h in hist]
            hfig = go.Figure()
            hfig.add_trace(go.Scatter(x=epochs_h, y=tr_h, name="Train Loss",
                line=dict(color="#58a6ff",width=2)))
            hfig.add_trace(go.Scatter(x=epochs_h, y=vl_h, name="Val Loss",
                line=dict(color="#3fb950",width=2,dash="dash")))
            hfig.update_layout(
                title="Training History", plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117", height=300,
                xaxis=dict(title="Epoch",color="#8b949e",gridcolor="#21262d"),
                yaxis=dict(title="Loss",color="#8b949e",gridcolor="#21262d"),
                legend=dict(bgcolor="#161b22",font=dict(color="#c9d1d9")),
                font=dict(color="#c9d1d9",family="Inter"))
            st.plotly_chart(hfig, use_container_width=True)
        else:
            st.info("No training history yet.")

    # Predictions CSV viewer
    pred_path = os.path.join(CKPT_PHASE2, "predictions_phase2.csv")
    if os.path.exists(pred_path):
        st.markdown("#### 🗒️ Sample Predictions (test set)")
        df_pred = pd.read_csv(pred_path).head(50)
        st.dataframe(df_pred, use_container_width=True)

    st.markdown("#### 🆚 Phase 2 vs Phase 1 (Original Dataset)")
    comp = {
        "Metric": ["Mod Classes", "Dataset", "PU Acc (target)", "Mod Acc (target)", "SNR MAE (target)"],
        "Phase 1 (Original)": ["4 (no DQPSK)", "Secondary_User/", "97–98%", "92–95%", "<1.5 dB"],
        "Phase 2 (New)":      ["5 (+ DQPSK)", "Symbol1/2/3", "~97%", "~92%", "<1.5 dB"],
    }
    st.dataframe(pd.DataFrame(comp), use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8b949e;font-size:0.8em;'>"
    "Spectrum-SLM Phase 2 &nbsp;|&nbsp; Anjani · Ashish Joshi · Mayank "
    "&nbsp;|&nbsp; Guide: Dr. Abhinandan S.P. &nbsp;|&nbsp; April 2026 &nbsp;|&nbsp; IIT Palakkad"
    "</p>",
    unsafe_allow_html=True,
)
