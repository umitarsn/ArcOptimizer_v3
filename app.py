# app.py
import os
import json
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, List

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import joblib
import streamlit.components.v1 as components

# =========================================================
# GENEL AYARLAR
# =========================================================
st.set_page_config(
    page_title="FeCr AI",
    page_icon="apple-touch-icon.png",
    layout="wide",
)

# Sidebar genişlik
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { width: 340px !important; }
    section[data-testid="stSidebar"] > div { width: 340px !important; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2.0rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

TZ = ZoneInfo("Europe/Istanbul")

SETUP_SAVE_PATH = "data/saved_inputs.json"
RUNTIME_SAVE_PATH = "data/runtime_data.json"
MODEL_SAVE_PATH = "models/arc_optimizer_model.pkl"

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Dijital ikiz hedefleri
DIGITAL_TWIN_HISTORICAL_HEATS = 1000
DIGITAL_TWIN_TARGET_HEATS = 10000
DIGITAL_TWIN_MIN_START = 1000

# Simülasyon
SIM_STREAM_TOTAL = DIGITAL_TWIN_TARGET_HEATS
SIM_STREAM_BATCH_DEFAULT = 25

# =========================================================
# SESSION STATE INIT
# =========================================================
def _init_state():
    defaults = {
        "info_state": {},
        "profit_info_state": {},
        "sim_data": None,
        "sim_full_data": None,
        "sim_stream_enabled": True,
        "sim_stream_autostep": True,
        "sim_stream_progress": DIGITAL_TWIN_HISTORICAL_HEATS,
        "sim_stream_last_step_progress": None,
        "sim_stream_autorefresh": False,
        "sim_stream_refresh_sec": 2,
        "model_status": "Henüz eğitilmedi.",
        "model_last_train_time": None,
        "model_last_train_rows": 0,
        "model_train_count": 0,
        "model_last_trained_rows_marker": 0,
        "view_mode": "Persona",
        "persona": "Plant Manager",
        "classic_page": "ArcOptimizer",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# =========================================================
# WIDGET BIND HELPERS
# =========================================================
def bind_toggle(label: str, state_key: str, widget_key: str, help_text: Optional[str] = None):
    def _sync():
        st.session_state[state_key] = st.session_state[widget_key]

    return st.toggle(
        label,
        value=bool(st.session_state.get(state_key, False)),
        key=widget_key,
        help=help_text,
        on_change=_sync,
    )


def bind_number_int(label: str, state_key: str, widget_key: str, min_v: int, max_v: int, step: int = 1):
    def _sync():
        st.session_state[state_key] = int(st.session_state[widget_key])

    return st.number_input(
        label,
        min_value=min_v,
        max_value=max_v,
        value=int(st.session_state.get(state_key, min_v)),
        step=step,
        key=widget_key,
        on_change=_sync,
    )


# =========================================================
# KAYITLI SETUP
# =========================================================
if os.path.exists(SETUP_SAVE_PATH):
    try:
        with open(SETUP_SAVE_PATH, "r", encoding="utf-8") as f:
            saved_inputs = json.load(f)
        if not isinstance(saved_inputs, dict):
            saved_inputs = {}
    except Exception:
        saved_inputs = {}
else:
    saved_inputs = {}


# =========================================================
# RUNTIME DATA
# =========================================================
def load_runtime_data():
    if os.path.exists(RUNTIME_SAVE_PATH):
        try:
            with open(RUNTIME_SAVE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def save_runtime_data(data_list):
    with open(RUNTIME_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)


runtime_data = load_runtime_data()


# =========================================================
# SIMÜLASYON
# =========================================================
def _make_heat_row(ts: datetime, idx: int):
    tap_weight = 35 + random.uniform(-3, 3)
    kwh_per_t = 420 + random.uniform(-25, 25)
    energy_kwh = tap_weight * kwh_per_t
    duration_min = 55 + random.uniform(-10, 10)
    tap_temp = 1610 + random.uniform(-15, 15)

    return {
        "timestamp": ts.isoformat(),
        "heat_id": f"SIM-{idx+1}",
        "tap_weight_t": tap_weight,
        "duration_min": duration_min,
        "energy_kwh": energy_kwh,
        "tap_temp_c": tap_temp,
        "kwh_per_t": kwh_per_t,
        "electrode_kg_per_heat": 2.0 + random.uniform(-0.3, 0.3),
        "slag_foaming_index": random.randint(3, 9),
        "panel_delta_t_c": 18 + random.uniform(-5, 8),
        "o2_flow_nm3h": 900 + random.uniform(-150, 150),
        "ems_on": random.choice([0, 1]),
    }


def generate_simulation_full_data(total_n=SIM_STREAM_TOTAL):
    now = datetime.now(TZ)
    start = now - timedelta(hours=total_n)
    return [_make_heat_row(start + timedelta(hours=i), i) for i in range(total_n)]


def ensure_simulation_data_initialized():
    if st.session_state.sim_full_data is None:
        st.session_state.sim_full_data = generate_simulation_full_data()
    if st.session_state.sim_data is None:
        st.session_state.sim_data = st.session_state.sim_full_data[:DIGITAL_TWIN_HISTORICAL_HEATS]


def advance_sim_stream(batch: int):
    cur = int(st.session_state.sim_stream_progress)
    nxt = min(cur + batch, SIM_STREAM_TOTAL)
    st.session_state.sim_data = st.session_state.sim_full_data[:nxt]
    st.session_state.sim_stream_progress = nxt


# =========================================================
# DF HAZIRLAMA – TÜRKİYE SAATİ FIX
# =========================================================
def to_df(data_source):
    if not data_source:
        return pd.DataFrame()

    df = pd.DataFrame(data_source).copy()
    if "timestamp" in df.columns:
        # ✅ TR saati: tz-aware ise convert, naive ise localize
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        try:
            if getattr(ts.dt, "tz", None) is not None:
                df["timestamp_dt"] = ts.dt.tz_convert(TZ)
            else:
                df["timestamp_dt"] = ts.dt.tz_localize(TZ)
        except Exception:
            df["timestamp_dt"] = ts
    else:
        df["timestamp_dt"] = pd.NaT

    df = df.sort_values("timestamp_dt")

    if "electrode_kg_per_heat" in df.columns and "tap_weight_t" in df.columns:
        df["electrode_kg_per_t"] = df["electrode_kg_per_heat"] / df["tap_weight_t"]

    return df


# =========================================================
# MODEL
# =========================================================
def train_arc_model(df: pd.DataFrame):
    feats = [
        "tap_weight_t",
        "duration_min",
        "energy_kwh",
        "o2_flow_nm3h",
        "slag_foaming_index",
        "panel_delta_t_c",
        "electrode_kg_per_heat",
    ]
    targets = ["kwh_per_t", "tap_temp_c"]

    for c in feats + targets:
        if c not in df.columns:
            return None

    X = df[feats].fillna(df[feats].mean(numeric_only=True))
    y = df[targets]
    if len(X) < 20:
        return None

    model = RandomForestRegressor(
        n_estimators=150, max_depth=7, random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    joblib.dump({"model": model, "feature_cols": feats, "target_cols": targets}, MODEL_SAVE_PATH)
    return model, feats, targets


def load_arc_model():
    if not os.path.exists(MODEL_SAVE_PATH):
        return None, None, None
    d = joblib.load(MODEL_SAVE_PATH)
    return d["model"], d["feature_cols"], d["target_cols"]


# =========================================================
# 24H + AI GRAFİK (OK DÖKÜM NOKTASININ ÜSTÜNDE)
# =========================================================
def build_24h_actual_vs_ai_chart(df, model, feat_cols, target_cols, height=420):
    if df.empty or "timestamp_dt" not in df.columns:
        st.info("Trend için veri yok.")
        return

    df = df.dropna(subset=["timestamp_dt"]).copy()
    if df.empty:
        st.info("Trend için zaman bilgisi yok.")
        return

    last_time = df["timestamp_dt"].max()
    window_start = last_time - timedelta(hours=24)
    df_24 = df[df["timestamp_dt"] >= window_start].copy()
    if df_24.empty:
        st.info("Son 24 saatlik pencerede veri yok.")
        return

    # sadece bu iki metrik demo'da
    keep = [c for c in ["kwh_per_t", "tap_temp_c"] if c in df_24.columns]
    if not keep:
        st.info("Grafik için uygun kolon yok.")
        return

    future_end = last_time + timedelta(hours=4)

    actual = df_24[["timestamp_dt"] + keep].melt(
        "timestamp_dt", keep, var_name="var", value_name="val"
    ).dropna()
    actual["type"] = "Aktüel"

    # AI potansiyel (demo)
    last = df_24.iloc[-1]
    last_kwh = float(last.get("kwh_per_t", np.nan))
    last_tap = float(last.get("tap_temp_c", np.nan))

    target_kwh = last_kwh - 5 if np.isfinite(last_kwh) else np.nan
    target_tap = last_tap + 5 if np.isfinite(last_tap) else np.nan

    future_wide = pd.DataFrame(
        {
            "timestamp_dt": [last_time, future_end],
            "kwh_per_t": [last_kwh, target_kwh],
            "tap_temp_c": [last_tap, target_tap],
        }
    )

    future = future_wide[["timestamp_dt"] + keep].melt(
        "timestamp_dt", keep, var_name="var", value_name="val"
    ).dropna()
    future["type"] = "Potansiyel (AI)"

    combined = pd.concat([actual, future], ignore_index=True)

    base = (
        alt.Chart(combined)
        .mark_line()
        .encode(
            x=alt.X(
                "timestamp_dt:T",
                title="Zaman (TR)",
                scale=alt.Scale(domain=[window_start, future_end]),
                axis=alt.Axis(format="%d.%m %H:%M", labelAngle=-35, tickCount=10),
            ),
            y=alt.Y("val:Q", title=None),
            color=alt.Color("var:N", title=None, legend=alt.Legend(orient="top", direction="horizontal")),
            strokeDash=alt.StrokeDash(
                "type:N",
                title=None,
                scale=alt.Scale(domain=["Aktüel", "Potansiyel (AI)"], range=[[1, 0], [6, 4]]),
            ),
        )
        .properties(height=height)
    )

    # now
    now_rule = (
        alt.Chart(pd.DataFrame({"timestamp_dt": [last_time]}))
        .mark_rule(color="black", strokeDash=[2, 2])
        .encode(x="timestamp_dt:T")
    )

    # tahmini döküm anı (kırmızı kesikli)
    fut_rule = (
        alt.Chart(pd.DataFrame({"timestamp_dt": [future_end]}))
        .mark_rule(color="red", strokeDash=[6, 4])
        .encode(x="timestamp_dt:T")
    )

    # OK + YAZI (döküm noktasının ÜSTÜNDE) — fontSize: 15
    tap_val = float(future_wide.iloc[-1].get("tap_temp_c", np.nan))
    tap_label = f"{tap_val:.0f} °C" if np.isfinite(tap_val) else "-"

    label_df = pd.DataFrame(
        {
            "timestamp_dt": [future_end],
            # noktanın biraz üstüne koy
            "val": [(tap_val + 40) if np.isfinite(tap_val) else 0],
            "label": [
                "⬆ Aktüel\nPotansiyel (AI)\n"
                f"Hedef Döküm: {future_end.strftime('%d.%m %H:%M')}\n"
                f"Hedef Tap: {tap_label}"
            ],
        }
    )

    label = (
        alt.Chart(label_df)
        .mark_text(
            align="left",
            dx=6,
            dy=-6,
            fontSize=15,          # ✅ 14–15 aralığı: 15 yaptım
            fontWeight="bold",
            lineBreak="\n",
        )
        .encode(x="timestamp_dt:T", y="val:Q", text="label:N")
    )

    full = alt.layer(base, now_rule, fut_rule, label).resolve_scale(y="independent")
    st.altair_chart(full, use_container_width=True)

    delta_min = (future_end - last_time).total_seconds() / 60.0
    st.caption(
        f"Sol: **aktüel (son 24 saat)** · Sağ: **AI potansiyel (kesikli)** · "
        f"'now' çizgisi: son ölçüm. Tahmini döküm anı ~ **{delta_min:.0f} dk** sonrası (kırmızı kesikli çizgi)."
    )


# =========================================================
# ARC OPTIMIZER SAYFASI
# =========================================================
def show_arc_optimizer_page(sim_mode=True):
    st.markdown("## Arc Optimizer – Trendler, KPI ve Öneriler")
    if sim_mode:
        ensure_simulation_data_initialized()
        df = to_df(st.session_state.sim_data)
    else:
        df = to_df(runtime_data)

    if df.empty:
        st.info("Veri yok.")
        return

    model, feat_cols, target_cols = load_arc_model()
    if model is None:
        trained = train_arc_model(df)
        if trained:
            model, feat_cols, target_cols = trained

    build_24h_actual_vs_ai_chart(df, model, feat_cols, target_cols)


# =========================================================
# SIDEBAR
# =========================================================
def sidebar_controls():
    sim_mode = st.toggle("Simülasyon Modu", value=True)
    if sim_mode:
        ensure_simulation_data_initialized()

    st.divider()
    batch = st.slider("Batch (şarj/adım)", 1, 500, SIM_STREAM_BATCH_DEFAULT)
    if st.button("▶️ İlerlet"):
        advance_sim_stream(batch)
        st.rerun()

    return sim_mode


# =========================================================
# MAIN
# =========================================================
def main():
    with st.sidebar:
        sim_mode = sidebar_controls()
    show_arc_optimizer_page(sim_mode)


if __name__ == "__main__":
    main()
