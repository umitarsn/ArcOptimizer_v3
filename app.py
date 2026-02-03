# app.py
import os
import json
import random
import base64
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

# ✅ Sidebar genişlik fix + ✅ Genel font (14–15px)
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 14.5px; }
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
TARGETS_SAVE_PATH = "data/targets.json"


def _safe_mkdir(path: str):
    """
    Bazı ortamlarda 'assets' vb. bir klasör yerine aynı isimde DOSYA olabiliyor.
    Bu durumda os.makedirs FileExistsError fırlatabiliyor.
    Çözüm: Eğer path bir dosya ise klasör oluşturmaya çalışma; alternatif klasör kullan.
    """
    if os.path.exists(path) and not os.path.isdir(path):
        return False
    os.makedirs(path, exist_ok=True)
    return True


_safe_mkdir("data")
_safe_mkdir("models")

# assets klasörü varsa kullan, yoksa alternatif isimle devam et
ASSETS_DIR = "assets"
if not _safe_mkdir(ASSETS_DIR):
    ASSETS_DIR = "assets_dir"
    _safe_mkdir(ASSETS_DIR)

# Repo'dan otomatik gösterilecek demo video
HSE_DEMO_VIDEO_PATH = os.path.join(ASSETS_DIR, "hse_demo.mp4")

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
        # sim akış state
        "sim_stream_enabled": True,
        "sim_stream_autostep": True,
        "sim_stream_progress": DIGITAL_TWIN_HISTORICAL_HEATS,
        "sim_stream_last_step_progress": None,
        # model meta
        "model_status": "Henüz eğitilmedi.",
        "model_last_train_time": None,
        "model_last_train_rows": 0,
        "model_train_count": 0,
        "model_last_trained_rows_marker": 0,
        # ui
        "classic_page": "ArcOptimizer",
        # targets
        "targets_loaded": False,
        "targets": None,
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


# =========================================================
# TARGETS (Hedefler / Recipe)
# =========================================================
def default_targets():
    return {
        "meta": {
            "source_mode": "Mühendis",  # "Mühendis" | "AI" | "Hibrit"
            "last_updated": datetime.now(TZ).isoformat(),
            "updated_by": "system",
            "notes": "Başlangıç hedefleri (demo).",
        },
        "targets": {
            "kwh_per_t": {"low": 400.0, "high": 430.0, "unit": "kWh/t"},
            "tap_temp_c": {"low": 1600.0, "high": 1630.0, "unit": "°C"},
            "electrode_kg_per_t": {"low": 0.040, "high": 0.060, "unit": "kg/t"},
            "o2_flow_nm3h": {"low": 700.0, "high": 1200.0, "unit": "Nm³/h"},
            "panel_delta_t_c": {"low": 0.0, "high": 25.0, "unit": "°C"},
            # Power quality / elektrik (kolon yoksa sapma hesaplamaz)
            "cos_phi_furnace": {"low": 0.80, "high": 0.92, "unit": "-"},
            "cos_phi_ladle": {"low": 0.90, "high": 0.97, "unit": "-"},
        },
    }


def load_targets():
    if os.path.exists(TARGETS_SAVE_PATH):
        try:
            with open(TARGETS_SAVE_PATH, "r", encoding="utf-8") as f:
                t = json.load(f)
            if isinstance(t, dict) and "targets" in t:
                return t
        except Exception:
            pass
    return default_targets()


def save_targets(t: dict):
    try:
        with open(TARGETS_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(t, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Hedefler kaydedilemedi: {e}")
        return False


def ensure_targets_loaded():
    if not st.session_state.targets_loaded or st.session_state.targets is None:
        st.session_state.targets = load_targets()
        st.session_state.targets_loaded = True


# =========================================================
# KAYITLI SETUP VERİLERİ
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
# RUNTIME VERİLERİ
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
    try:
        with open(RUNTIME_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Runtime verileri kaydedilemedi: {e}")


runtime_data = load_runtime_data()


# =========================================================
# SİMÜLASYON VERİLERİ
# =========================================================
def _make_heat_row(ts: datetime, idx: int):
    heat_id = f"SIM-{idx+1}"

    tap_weight = 35 + random.uniform(-3, 3)  # ton
    kwh_per_t = 420 + random.uniform(-25, 25)  # kWh/t
    energy_kwh = tap_weight * kwh_per_t  # kWh
    duration_min = 55 + random.uniform(-10, 10)  # dk
    tap_temp = 1610 + random.uniform(-15, 15)  # °C
    o2_flow = 950 + random.uniform(-150, 150)  # Nm³/h
    slag_foaming = random.randint(3, 9)  # 0–10
    panel_delta_t = 18 + random.uniform(-5, 8)  # °C
    electrode_cons = 1.9 + random.uniform(-0.3, 0.3)  # kg/heat

    return {
        "timestamp": ts.isoformat(),
        "heat_id": heat_id,
        "tap_weight_t": float(tap_weight),
        "duration_min": float(duration_min),
        "energy_kwh": float(energy_kwh),
        "tap_temp_c": float(tap_temp),
        "o2_flow_nm3h": float(o2_flow),
        "slag_foaming_index": float(slag_foaming),
        "panel_delta_t_c": float(panel_delta_t),
        "electrode_kg_per_heat": float(electrode_cons),
        "kwh_per_t": float(kwh_per_t),
        "operator_note": "Simülasyon kaydı",
        "grade": random.choice(["A", "B", "C"]),
        "ems_on": random.choice([0, 1]),
    }


def generate_simulation_full_data(total_n: int = SIM_STREAM_TOTAL):
    step_minutes = 60
    now = datetime.now(TZ)
    start = now - timedelta(minutes=step_minutes * (total_n - 1))
    return [_make_heat_row(start + timedelta(minutes=step_minutes * i), i) for i in range(total_n)]


def ensure_simulation_data_initialized():
    if st.session_state.sim_full_data is None:
        st.session_state.sim_full_data = generate_simulation_full_data(SIM_STREAM_TOTAL)

    if st.session_state.sim_data is None:
        st.session_state.sim_stream_progress = DIGITAL_TWIN_HISTORICAL_HEATS
        st.session_state.sim_data = st.session_state.sim_full_data[:DIGITAL_TWIN_HISTORICAL_HEATS]


def reset_sim_to_1000():
    ensure_simulation_data_initialized()
    st.session_state.sim_stream_progress = DIGITAL_TWIN_HISTORICAL_HEATS
    st.session_state.sim_data = st.session_state.sim_full_data[:DIGITAL_TWIN_HISTORICAL_HEATS]
    st.session_state.sim_stream_last_step_progress = None


def advance_sim_stream(batch: int):
    ensure_simulation_data_initialized()
    cur = int(st.session_state.sim_stream_progress)
    target = SIM_STREAM_TOTAL
    if cur >= target:
        return False
    nxt = min(cur + int(batch), target)
    st.session_state.sim_data = st.session_state.sim_full_data[:nxt]
    st.session_state.sim_stream_progress = nxt
    return True


# =========================================================
# MODEL
# =========================================================
def get_arc_training_data(df: pd.DataFrame):
    required_cols = [
        "tap_weight_t",
        "duration_min",
        "energy_kwh",
        "o2_flow_nm3h",
        "slag_foaming_index",
        "panel_delta_t_c",
        "electrode_kg_per_heat",
        "kwh_per_t",
        "tap_temp_c",
    ]
    for col in required_cols:
        if col not in df.columns:
            return None, None, None, None

    mask = df["kwh_per_t"].notna() & df["tap_temp_c"].notna()
    sub = df.loc[mask, required_cols].copy()
    if len(sub) < 10:
        return None, None, None, None

    feature_cols = [
        "tap_weight_t",
        "duration_min",
        "energy_kwh",
        "o2_flow_nm3h",
        "slag_foaming_index",
        "panel_delta_t_c",
        "electrode_kg_per_heat",
    ]
    target_cols = ["kwh_per_t", "tap_temp_c"]

    X = sub[feature_cols].fillna(sub[feature_cols].mean(numeric_only=True))
    y = sub[target_cols]
    if len(X) < 10:
        return None, None, None, None
    return X, y, feature_cols, target_cols


def train_arc_model(df: pd.DataFrame, note: str = "", min_samples: int = 20, silent: bool = False):
    st.session_state.model_status = "Eğitiliyor..."

    X, y, feature_cols, target_cols = get_arc_training_data(df)
    if X is None:
        st.session_state.model_status = "Eğitim için uygun veri bulunamadı."
        if not silent:
            st.error("Model eğitimi için gerekli kolonlar yok veya yeterli dolu kayıt yok.")
        return False

    if len(X) < min_samples:
        st.session_state.model_status = f"Eğitim için veri yetersiz: {len(X)} şarj (gereken ≥ {min_samples})."
        if not silent:
            st.warning(f"Bu mod için en az {min_samples} şarj gerekli, şu anda {len(X)} kayıt var.")
        return False

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=7,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    joblib.dump({"model": model, "feature_cols": feature_cols, "target_cols": target_cols}, MODEL_SAVE_PATH)

    now_str = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    rows = len(X)

    st.session_state.model_status = f"Eğitildi ✅ {note}".strip()
    st.session_state.model_last_train_time = now_str
    st.session_state.model_last_train_rows = rows
    st.session_state.model_train_count += 1
    st.session_state.model_last_trained_rows_marker = rows

    if not silent:
        st.success(f"Model {rows} şarj verisiyle {now_str} tarihinde eğitildi.")
    return True


def load_arc_model():
    if not os.path.exists(MODEL_SAVE_PATH):
        return None, None, None
    try:
        data = joblib.load(MODEL_SAVE_PATH)
        return data.get("model"), data.get("feature_cols"), data.get("target_cols")
    except Exception:
        return None, None, None


# =========================================================
# EXCEL – SETUP
# =========================================================
@st.cache_data
def load_sheets():
    file_name = "dc_saf_soru_tablosu.xlsx"
    try:
        xls = pd.read_excel(file_name, sheet_name=None)
        cleaned = {}
        for name, df in xls.items():
            df2 = df.dropna(how="all")
            if not df2.empty:
                cleaned[name] = df2
        return cleaned
    except Exception as e:
        st.error(f"Excel dosyası yüklenemedi: {e}")
        return {}


# =========================================================
# ORTAK: DF HAZIRLA
# =========================================================
def get_active_data(sim_mode: bool):
    return st.session_state.sim_data if sim_mode else runtime_data


def to_df(data_source):
    if not data_source:
        return pd.DataFrame()

    df = pd.DataFrame(data_source).copy()
    if "timestamp" in df.columns:
        try:
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TZ)
        except Exception:
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp_dt"] = pd.NaT

    df = df.sort_values("timestamp_dt")

    if "kwh_per_t" not in df.columns and "energy_kwh" in df.columns and "tap_weight_t" in df.columns:
        df["kwh_per_t"] = df.apply(
            lambda r: (r["energy_kwh"] / r["tap_weight_t"]) if r.get("tap_weight_t", 0) else np.nan,
            axis=1,
        )
    if "electrode_kg_per_heat" in df.columns and "tap_weight_t" in df.columns:
        df["electrode_kg_per_t"] = df.apply(
            lambda r: (r["electrode_kg_per_heat"] / r["tap_weight_t"]) if r.get("tap_weight_t", 0) else np.nan,
            axis=1,
        )
    return df


def kpi_pack(df: pd.DataFrame):
    if df.empty:
        return {}

    last = df.iloc[-1]
    last10 = df.tail(10)

    avg_kwh_t = float(last10["kwh_per_t"].dropna().mean()) if "kwh_per_t" in df.columns else np.nan
    avg_elec_pt = float(last10["electrode_kg_per_t"].dropna().mean()) if "electrode_kg_per_t" in df.columns else np.nan
    avg_tap = float(last10["tap_temp_c"].dropna().mean()) if "tap_temp_c" in df.columns else np.nan

    pot_kwh = max(avg_kwh_t - 5.0, 0.0) if not np.isnan(avg_kwh_t) else np.nan

    return {
        "last": last,
        "rows": len(df),
        "avg_kwh_t_10": avg_kwh_t,
        "avg_elec_pt_10": avg_elec_pt,
        "avg_tap_10": avg_tap,
        "pot_kwh_t": pot_kwh,
    }


def money_pack(df: pd.DataFrame, energy_price=0.12, electrode_price=3.0, annual_ton=250_000):
    if df.empty:
        return {"eur_per_t": 0.0, "eur_per_year": 0.0}

    kpi = kpi_pack(df)
    last = kpi["last"]

    eur_per_t = 0.0

    if "kwh_per_t" in df.columns and not pd.isna(last.get("kwh_per_t")) and not np.isnan(kpi["avg_kwh_t_10"]):
        real = float(last["kwh_per_t"])
        target = max(float(kpi["avg_kwh_t_10"]) - 5.0, 0.0)
        diff = max(real - target, 0.0)
        eur_per_t += diff * energy_price

    if "electrode_kg_per_t" in df.columns and not pd.isna(last.get("electrode_kg_per_t")):
        real_pt = float(last["electrode_kg_per_t"])
        target_pt = max(real_pt - 0.02, 0.0)  # demo
        diff = max(real_pt - target_pt, 0.0)
        eur_per_t += diff * electrode_price

    eur_per_year = eur_per_t * float(annual_ton)
    return {"eur_per_t": eur_per_t, "eur_per_year": eur_per_year}


# =========================================================
# 24H + AI TAHMİN GRAFİĞİ (KeyError fix + daha sağlam)
# =========================================================
def build_24h_actual_vs_ai_chart(
    df: pd.DataFrame,
    model,
    feat_cols: Optional[List[str]],
    target_cols: Optional[List[str]],
    height: int = 420,
):
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

    keep = []
    if "kwh_per_t" in df_24.columns:
        keep.append("kwh_per_t")
    if "electrode_kg_per_t" in df_24.columns:
        keep.append("electrode_kg_per_t")
    if "tap_temp_c" in df_24.columns:
        keep.append("tap_temp_c")

    if not keep:
        st.info("Grafik için uygun kolon yok.")
        return

    var_map = {
        "kwh_per_t": "kWh/t",
        "electrode_kg_per_t": "Elektrot (kg/t)",
        "tap_temp_c": "Tap T (°C)",
    }

    future_h = 4
    future_end = last_time + timedelta(hours=future_h)

    last_row = df_24.iloc[-1]
    tail50 = df.tail(50).copy()

    def safe_mean_from_df(d: pd.DataFrame, col: str, default: float) -> float:
        if col not in d.columns:
            return float(default)
        s = d[col].dropna()
        return float(s.mean()) if len(s) else float(default)

    # ✅ KeyError'ı bitiren kısım: col yoksa asla df[col] ile erişme.
    base_kwh = safe_mean_from_df(tail50, "kwh_per_t", safe_mean_from_df(df_24, "kwh_per_t", 420.0))
    base_elec = safe_mean_from_df(tail50, "electrode_kg_per_t", safe_mean_from_df(df_24, "electrode_kg_per_t", 0.055))
    base_tap = safe_mean_from_df(tail50, "tap_temp_c", safe_mean_from_df(df_24, "tap_temp_c", 1610.0))

    target_kwh = max(base_kwh - 5.0, 0.0)
    target_elec = max(base_elec - 0.002, 0.0)
    target_tap = base_tap + 5.0

    if model is not None and feat_cols is not None and target_cols is not None:
        feat_defaults = {}
        for c in feat_cols:
            if c in tail50.columns:
                feat_defaults[c] = safe_mean_from_df(tail50, c, safe_mean_from_df(df, c, 0.0))
            else:
                feat_defaults[c] = 0.0

        if "slag_foaming_index" in feat_defaults:
            feat_defaults["slag_foaming_index"] = 7.0
        if "panel_delta_t_c" in feat_defaults:
            feat_defaults["panel_delta_t_c"] = min(20.0, float(feat_defaults["panel_delta_t_c"]))

        row_df = pd.DataFrame([feat_defaults])[feat_cols].fillna(0.0)
        try:
            preds = model.predict(row_df)[0]
            pred_dict = dict(zip(target_cols, preds))
            if "kwh_per_t" in pred_dict and np.isfinite(pred_dict["kwh_per_t"]):
                target_kwh = max(float(pred_dict["kwh_per_t"]), 0.0)
            if "tap_temp_c" in pred_dict and np.isfinite(pred_dict["tap_temp_c"]):
                target_tap = float(pred_dict["tap_temp_c"])
            target_elec = max(base_elec - 0.002, 0.0)
        except Exception:
            pass

    actual = df_24[["timestamp_dt"] + keep].melt("timestamp_dt", var_name="var", value_name="val").dropna()
    actual["type"] = "Aktüel"
    actual["var_name"] = actual["var"].map(var_map).fillna(actual["var"])

    def get_last_val(col: str, fallback: float) -> float:
        v = last_row.get(col, np.nan)
        if pd.isna(v):
            return float(fallback)
        return float(v)

    last_kwh = get_last_val("kwh_per_t", base_kwh)
    last_elec = get_last_val("electrode_kg_per_t", base_elec)
    last_tap = get_last_val("tap_temp_c", base_tap)

    future_points = []
    steps = 8
    for i in range(steps + 1):
        frac = i / steps
        t = last_time + (future_end - last_time) * frac
        row = {"timestamp_dt": t}
        if "kwh_per_t" in keep:
            row["kwh_per_t"] = last_kwh + (target_kwh - last_kwh) * frac
        if "electrode_kg_per_t" in keep:
            row["electrode_kg_per_t"] = last_elec + (target_elec - last_elec) * frac
        if "tap_temp_c" in keep:
            row["tap_temp_c"] = last_tap + (target_tap - last_tap) * frac
        future_points.append(row)

    future_df = pd.DataFrame(future_points)
    future = future_df[["timestamp_dt"] + keep].melt("timestamp_dt", var_name="var", value_name="val").dropna()
    future["type"] = "Potansiyel (AI)"
    future["var_name"] = future["var"].map(var_map).fillna(future["var"])

    combined = pd.concat([actual, future], ignore_index=True)

    domain_min = window_start
    domain_max = future_end

    tap_point_val = float(future_df.iloc[-1].get("tap_temp_c", np.nan))
    hedef_time_str = future_end.strftime("%d.%m %H:%M")
    hedef_temp_str = f"{tap_point_val:.0f} °C" if np.isfinite(tap_point_val) else "-"

    st.markdown(
        f"""
        <div style="display:flex; justify-content:flex-end; margin-top:2px; margin-bottom:6px;">
          <div style="text-align:left; padding:6px 10px; border-radius:10px;">
            <div style="font-size:18px; font-weight:800; line-height:1.2;">Zaman (son 24 saat + AI tahmin)</div>
            <div style="font-size:13px; line-height:1.25; color:#555;">
              Sol: aktüel · Sağ: AI potansiyel (kesikli) · 'now': son ölçüm · kırmızı: hedef döküm
            </div>
            <div style="font-size:13px; line-height:1.25; margin-top:6px;">
              <b>Hedef Döküm (AI):</b> {hedef_time_str} · <b>Hedef Tap T:</b> {hedef_temp_str}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    base_chart = (
        alt.Chart(combined)
        .mark_line()
        .encode(
            x=alt.X(
                "timestamp_dt:T",
                title=None,
                scale=alt.Scale(domain=[domain_min, domain_max]),
                axis=alt.Axis(format="%d.%m %H:%M", tickCount=10, labelAngle=-35),
            ),
            y=alt.Y("val:Q", title=None),
            color=alt.Color("var_name:N", title=None, legend=alt.Legend(orient="top", direction="horizontal")),
            strokeDash=alt.StrokeDash(
                "type:N",
                title=None,
                scale=alt.Scale(domain=["Aktüel", "Potansiyel (AI)"], range=[[1, 0], [6, 4]]),
            ),
        )
        .properties(height=height)
    )

    now_df = pd.DataFrame({"timestamp_dt": [last_time]})
    now_rule = alt.Chart(now_df).mark_rule(strokeDash=[2, 2], color="black").encode(x="timestamp_dt:T")

    fut_df = pd.DataFrame({"timestamp_dt": [future_end]})
    future_rule = alt.Chart(fut_df).mark_rule(strokeDash=[6, 4], color="red").encode(x="timestamp_dt:T")

    layers = [base_chart, now_rule, future_rule]

    if "tap_temp_c" in keep and np.isfinite(tap_point_val):
        tp = pd.DataFrame({"timestamp_dt": [future_end], "val": [tap_point_val]})
        point = alt.Chart(tp).mark_point(size=120, filled=True).encode(x="timestamp_dt:T", y="val:Q")
        layers.append(point)

    full = alt.layer(*layers).resolve_scale(y="independent")
    st.altair_chart(full, use_container_width=True)

    delta_min = (future_end - last_time).total_seconds() / 60.0
    st.caption(f"now çizgisi: son ölçüm. Tahmini döküm anı ~ **{delta_min:.0f} dk** sonrası (kırmızı kesikli çizgi).")


def actual_vs_potential_last50_table(df: pd.DataFrame, model, feat_cols, target_cols):
    if df.empty:
        return

    tail = df.tail(50).copy()

    def m(col, default=np.nan):
        if col not in tail.columns:
            return default
        s = tail[col].dropna()
        return float(s.mean()) if len(s) else default

    act_kwh = m("kwh_per_t", np.nan)
    act_elec = m("electrode_kg_per_t", np.nan)
    act_tap = m("tap_temp_c", np.nan)

    pot_kwh = np.nan
    pot_elec = np.nan
    pot_tap = np.nan

    if np.isfinite(act_kwh):
        pot_kwh = max(act_kwh - 5.0, 0.0)
    if np.isfinite(act_elec):
        pot_elec = max(act_elec - 0.002, 0.0)
    if np.isfinite(act_tap):
        pot_tap = act_tap + 5.0

    if model is not None and feat_cols is not None and target_cols is not None and len(tail) >= 10:
        feat_defaults = {}
        for c in feat_cols:
            if c in tail.columns:
                s = tail[c].dropna()
                feat_defaults[c] = float(s.mean()) if len(s) else 0.0
            else:
                feat_defaults[c] = 0.0

        if "slag_foaming_index" in feat_defaults:
            feat_defaults["slag_foaming_index"] = 7.0
        if "panel_delta_t_c" in feat_defaults:
            feat_defaults["panel_delta_t_c"] = min(20.0, float(feat_defaults["panel_delta_t_c"]))

        row_df = pd.DataFrame([feat_defaults])[feat_cols].fillna(0.0)
        try:
            preds = model.predict(row_df)[0]
            pred_dict = dict(zip(target_cols, preds))
            if "kwh_per_t" in pred_dict and np.isfinite(pred_dict["kwh_per_t"]):
                pot_kwh = max(float(pred_dict["kwh_per_t"]), 0.0)
            if "tap_temp_c" in pred_dict and np.isfinite(pred_dict["tap_temp_c"]):
                pot_tap = float(pred_dict["tap_temp_c"])
            if np.isfinite(act_elec):
                pot_elec = max(act_elec - 0.002, 0.0)
        except Exception:
            pass

    rows = []
    if np.isfinite(act_kwh):
        rows.append({"Gösterge": "kWh/t", "Aktüel (son50 ort.)": f"{act_kwh:.1f}", "Potansiyel (AI)": f"{pot_kwh:.1f}" if np.isfinite(pot_kwh) else "-"})
    if np.isfinite(act_elec):
        rows.append({"Gösterge": "Elektrot (kg/t)", "Aktüel (son50 ort.)": f"{act_elec:.3f}", "Potansiyel (AI)": f"{pot_elec:.3f}" if np.isfinite(pot_elec) else "-"})
    if np.isfinite(act_tap):
        rows.append({"Gösterge": "Tap T (°C)", "Aktüel (son50 ort.)": f"{act_tap:.0f}", "Potansiyel (AI)": f"{pot_tap:.0f}" if np.isfinite(pot_tap) else "-"})

    if rows:
        st.markdown("#### 🎯 Aktüel vs Potansiyel (Son 50 Ortalama)")
        st.table(pd.DataFrame(rows))


# =========================================================
# 1) SETUP
# =========================================================
def show_setup_form():
    st.markdown("## Setup – Sabit Proses / Tasarım Verileri")
    st.markdown(
        "Bu form **dc_saf_soru_tablosu.xlsx** dosyasına göre hazırlanmıştır.\n\n"
        "🔴 Zorunlu (Önem:1) · 🟡 Faydalı (Önem:2) · ⚪ Opsiyonel (Önem:3)\n"
        "Detay için satır sonundaki ℹ️ butonu."
    )

    sheets = load_sheets()
    if not sheets:
        return

    total_fields = 0
    total_filled = 0
    required_fields = 0
    required_filled = 0

    for sheet_idx, (sheet_name, df) in enumerate(sheets.items(), start=1):
        with st.expander(f"{sheet_idx}. {sheet_name}", expanded=(sheet_idx == 1)):
            df.columns = [str(c).strip() for c in df.columns]
            unit_cols = [c for c in df.columns if "set" in c.lower()]
            unit_col_name = unit_cols[0] if unit_cols else None

            for row_idx, row in df.iterrows():
                row_key = f"{sheet_idx}_{row_idx}"

                onem_raw = row.get("Önem", 3)
                try:
                    onem = int(onem_raw)
                except Exception:
                    onem = 3
                renk = {1: "🔴", 2: "🟡", 3: "⚪"}.get(onem, "⚪")

                raw_birim = row.get(unit_col_name, "") if unit_col_name else ""
                birim = ""
                if pd.notna(raw_birim):
                    s = str(raw_birim).strip()
                    if s.lower() not in ("", "none", "nan"):
                        birim = s

                tag = row.get("Tag", "")
                val_key = f"{sheet_name}|{tag}"

                c1, c2, c3, c4, c5 = st.columns([2.2, 2.5, 4.0, 2.5, 0.7])
                c1.markdown(f"**{tag}**")
                c2.markdown(f"{renk} {row.get('Değişken', '')}")
                c3.markdown(str(row.get("Açıklama", "")))

                current_val = saved_inputs.get(val_key, "")
                with c4:
                    ic, uc = st.columns([5, 2])
                    with ic:
                        new_val = st.text_input("", value=current_val, key=val_key, label_visibility="collapsed")
                        if new_val != current_val:
                            saved_inputs[val_key] = new_val
                            try:
                                with open(SETUP_SAVE_PATH, "w", encoding="utf-8") as f:
                                    json.dump(saved_inputs, f, ensure_ascii=False, indent=2)
                            except Exception:
                                pass
                    with uc:
                        if birim:
                            st.markdown(f"**{birim}**")

                with c5:
                    if st.button("ℹ️", key=f"info_{row_key}"):
                        st.session_state.info_state[row_key] = not st.session_state.info_state.get(row_key, False)

                if st.session_state.info_state.get(row_key, False):
                    detaylar = []
                    da = row.get("Detaylı Açıklama")
                    if isinstance(da, str) and da.strip():
                        detaylar.append("🔷 **Detay:** " + da)
                    vk = row.get("Veri Kaynağı")
                    if isinstance(vk, str) and vk.strip():
                        detaylar.append("📌 **Kaynak:** " + vk)
                    ka = row.get("Kayıt Aralığı")
                    if isinstance(ka, str) and ka.strip():
                        detaylar.append("⏱️ **Aralık:** " + ka)
                    detaylar.append("🔵 **Önem:** " + str(onem))
                    st.info("\n".join(detaylar))

                total_fields += 1
                val = str(saved_inputs.get(val_key, "")).strip()
                if val:
                    total_filled += 1
                    if onem == 1:
                        required_filled += 1
                if onem == 1:
                    required_fields += 1

    with st.sidebar:
        st.subheader("📊 Setup Veri Giriş Durumu")
        pct_all = round(100 * total_filled / total_fields, 1) if total_fields else 0
        pct_req = round(100 * required_filled / required_fields, 1) if required_fields else 0
        st.metric("Toplam Giriş Oranı", f"{pct_all}%")
        st.progress(min(pct_all / 100, 1.0))
        st.metric("Zorunlu Veri Girişi", f"{pct_req}%")
        st.progress(min(pct_req / 100, 1.0))
        eksik = required_fields - required_filled
        if eksik > 0:
            st.warning(f"❗ Eksik Zorunlu Değerler: {eksik}")


# =========================================================
# 2) CANLI VERİ
# =========================================================
def show_runtime_page(sim_mode: bool):
    st.markdown("## Canlı Veri – Şarj Bazlı Anlık Veriler")
    if sim_mode:
        st.info("🧪 Simülasyon Modu Aktif. Aşağıdaki veriler simülasyon datasıdır.")
    else:
        st.markdown("Bu sayfada her şarj için veriler girilir veya otomasyondan okunur.")

    with st.form("runtime_form", clear_on_submit=True):
        st.markdown("### Yeni Şarj Kaydı Ekle")

        c1, c2, c3 = st.columns(3)
        with c1:
            heat_id = st.text_input("Heat ID / Şarj No", "")
        with c2:
            tap_weight = st.number_input("Tap Weight (ton)", min_value=0.0, step=0.1)
        with c3:
            duration_min = st.number_input("Toplam Süre (dk)", min_value=0.0, step=1.0)

        c4, c5, c6 = st.columns(3)
        with c4:
            energy_kwh = st.number_input("Toplam Enerji (kWh)", min_value=0.0, step=10.0)
        with c5:
            tap_temp = st.number_input("Tap Sıcaklığı (°C)", min_value=0.0, max_value=2000.0, step=1.0)
        with c6:
            o2_flow = st.number_input("Ortalama O2 Debisi (Nm³/h)", min_value=0.0, step=1.0)

        c7, c8, c9 = st.columns(3)
        with c7:
            slag_foaming = st.slider("Slag Foaming Seviyesi (0–10)", 0, 10, 5)
        with c8:
            panel_delta_t = st.number_input("Panel ΔT (°C)", min_value=0.0, step=0.1)
        with c9:
            electrode_cons = st.number_input("Elektrot Tüketimi (kg/şarj)", min_value=0.0, step=0.01)

        note
