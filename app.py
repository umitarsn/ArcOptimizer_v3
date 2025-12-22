import os
import json
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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

# ✅ Sidebar genişlik fix
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
        # sim akış
        "sim_stream_enabled": True,
        "sim_stream_autostep": True,
        "sim_stream_progress": DIGITAL_TWIN_HISTORICAL_HEATS,
        "sim_stream_last_step_progress": None,
        "sim_stream_autorefresh": False,
        "sim_stream_refresh_sec": 2,
        # model meta
        "model_status": "Henüz eğitilmedi.",
        "model_last_train_time": None,
        "model_last_train_rows": 0,
        "model_train_count": 0,
        "model_last_trained_rows_marker": 0,
        # ui
        "view_mode": "Persona",
        "persona": "Plant Manager",
        "classic_page": "ArcOptimizer",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# =========================================================
# KAYITLI SETUP VERİLERİ
# =========================================================
if os.path.exists(SETUP_SAVE_PATH):
    with open(SETUP_SAVE_PATH, "r", encoding="utf-8") as f:
        saved_inputs = json.load(f)
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

    tap_weight = 35 + random.uniform(-3, 3)          # ton
    kwh_per_t = 420 + random.uniform(-25, 25)        # kWh/t
    energy_kwh = tap_weight * kwh_per_t              # kWh
    duration_min = 55 + random.uniform(-10, 10)      # dk
    tap_temp = 1610 + random.uniform(-15, 15)        # °C
    o2_flow = 950 + random.uniform(-150, 150)        # Nm³/h
    slag_foaming = random.randint(3, 9)              # 0–10
    panel_delta_t = 18 + random.uniform(-5, 8)       # °C
    electrode_cons = 1.9 + random.uniform(-0.3, 0.3) # kg/heat

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
        # persona raporları için basit alanlar:
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


def html_autorefresh(seconds: int):
    # Streamlit'te st.autorefresh yok. Paketsiz çözüm: meta refresh.
    sec = max(1, int(seconds))
    components.html(f"<meta http-equiv='refresh' content='{sec}'>", height=0)


# =========================================================
# MODEL
# =========================================================
def get_arc_training_data(df: pd.DataFrame):
    required_cols = [
        "tap_weight_t", "duration_min", "energy_kwh", "o2_flow_nm3h",
        "slag_foaming_index", "panel_delta_t_c", "electrode_kg_per_heat",
        "kwh_per_t", "tap_temp_c",
    ]
    for col in required_cols:
        if col not in df.columns:
            return None, None, None, None

    mask = df["kwh_per_t"].notna() & df["tap_temp_c"].notna()
    sub = df.loc[mask, required_cols].copy()
    if len(sub) < 10:
        return None, None, None, None

    feature_cols = [
        "tap_weight_t", "duration_min", "energy_kwh", "o2_flow_nm3h",
        "slag_foaming_index", "panel_delta_t_c", "electrode_kg_per_heat",
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

    # derived
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
        return {"eur_per_t": 0.0, "eur_per_year": 0.0, "notes": []}

    kpi = kpi_pack(df)
    last = kpi["last"]

    eur_per_t = 0.0
    notes = []

    # energy
    if "kwh_per_t" in df.columns and not pd.isna(last.get("kwh_per_t")) and not np.isnan(kpi["avg_kwh_t_10"]):
        real = float(last["kwh_per_t"])
        target = max(float(kpi["avg_kwh_t_10"]) - 5.0, 0.0)
        diff = max(real - target, 0.0)
        gain = diff * energy_price
        eur_per_t += gain
        notes.append(("Energy", gain))

    # electrode
    if "electrode_kg_per_t" in df.columns and not pd.isna(last.get("electrode_kg_per_t")):
        real_pt = float(last["electrode_kg_per_t"])
        target_pt = max(real_pt - 0.02, 0.0)  # demo
        diff = max(real_pt - target_pt, 0.0)
        gain = diff * electrode_price
        eur_per_t += gain
        notes.append(("Electrode", gain))

    eur_per_year = eur_per_t * float(annual_ton)
    return {"eur_per_t": eur_per_t, "eur_per_year": eur_per_year, "notes": notes}


def trend_chart(df: pd.DataFrame, cols=("kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"), height=360):
    if df.empty or "timestamp_dt" not in df.columns:
        st.info("Trend için veri yok.")
        return

    use_cols = [c for c in cols if c in df.columns]
    if not use_cols:
        st.info("Trend için uygun kolon yok.")
        return

    tmp = df[["timestamp_dt"] + use_cols].copy()
    long = tmp.melt("timestamp_dt", var_name="var", value_name="val")
    var_map = {
        "kwh_per_t": "kWh/t",
        "tap_temp_c": "Tap T (°C)",
        "electrode_kg_per_heat": "Elektrot (kg/şarj)",
        "electrode_kg_per_t": "Elektrot (kg/t)",
        "panel_delta_t_c": "Panel ΔT (°C)",
        "o2_flow_nm3h": "O2 (Nm³/h)",
        "duration_min": "Süre (dk)",
        "slag_foaming_index": "Slag Foaming",
    }
    long["var_name"] = long["var"].map(var_map).fillna(long["var"])

    ch = (
        alt.Chart(long.dropna())
        .mark_line()
        .encode(
            x=alt.X("timestamp_dt:T", title="Zaman", axis=alt.Axis(format="%d.%m %H:%M", labelAngle=-35)),
            y=alt.Y("val:Q", title=None),
            color=alt.Color("var_name:N", title=None, legend=alt.Legend(orient="top", direction="horizontal")),
        )
        .properties(height=height)
    )
    st.altair_chart(ch.interactive(), use_container_width=True)


def distro_summary(df: pd.DataFrame):
    """Negatif psikoloji yaratmayan özet: percentile + son 3 ort."""
    out = []

    def add_metric(name, s: pd.Series, fmt="{:.2f}"):
        s = s.dropna()
        if len(s) < 5:
            return
        out.append({
            "Gösterge": name,
            "p10": fmt.format(s.quantile(0.10)),
            "p50": fmt.format(s.quantile(0.50)),
            "p90": fmt.format(s.quantile(0.90)),
            "Son 3 Ort.": fmt.format(s.tail(3).mean()) if len(s) >= 3 else "-",
        })

    if "kwh_per_t" in df.columns:
        add_metric("kWh/t", df["kwh_per_t"], fmt="{:.1f}")
    if "electrode_kg_per_t" in df.columns:
        add_metric("Elektrot (kg/t)", df["electrode_kg_per_t"], fmt="{:.3f}")
    if "tap_temp_c" in df.columns:
        add_metric("Tap T (°C)", df["tap_temp_c"], fmt="{:.0f}")

    return pd.DataFrame(out)


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
                            with open(SETUP_SAVE_PATH, "w", encoding="utf-8") as f:
                                json.dump(saved_inputs, f, ensure_ascii=False, indent=2)
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

        note = st.text_input("Operatör Notu (opsiyonel)", "")
        submitted = st.form_submit_button("Kaydet")

    if submitted:
        if not heat_id:
            st.error("Heat ID / Şarj No zorunlu.")
        else:
            if sim_mode:
                st.warning("Simülasyon modunda kayıt dosyaya yazılmaz (demo amaçlı).")
            else:
                now = datetime.now(TZ).isoformat()
                kwh_per_t = energy_kwh / tap_weight if tap_weight > 0 else None
                new_entry = {
                    "timestamp": now,
                    "heat_id": heat_id,
                    "tap_weight_t": tap_weight,
                    "duration_min": duration_min,
                    "energy_kwh": energy_kwh,
                    "tap_temp_c": tap_temp,
                    "o2_flow_nm3h": o2_flow,
                    "slag_foaming_index": slag_foaming,
                    "panel_delta_t_c": panel_delta_t,
                    "electrode_kg_per_heat": electrode_cons,
                    "kwh_per_t": kwh_per_t,
                    "operator_note": note,
                    "grade": "UNK",
                    "ems_on": 1,
                }
                runtime_data.append(new_entry)
                save_runtime_data(runtime_data)
                st.success(f"Şarj kaydı eklendi: {heat_id}")

    df = to_df(get_active_data(sim_mode))
    if df.empty:
        st.info("Henüz veri yok.")
        return

    st.markdown("### Kayıtlı Veriler")
    cols = [
        "timestamp_dt", "heat_id", "tap_weight_t", "duration_min", "energy_kwh",
        "kwh_per_t", "tap_temp_c", "electrode_kg_per_heat", "electrode_kg_per_t",
        "slag_foaming_index", "panel_delta_t_c", "o2_flow_nm3h", "grade", "ems_on",
    ]
    show_cols = [c for c in cols if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True)


# =========================================================
# 3) ARC OPTIMIZER
# - “En kötü/En iyi” yok
# - What-if sağ kutunun altında
# =========================================================
def show_arc_optimizer_page(sim_mode: bool):
    st.markdown("## 3. Arc Optimizer – Trendler, KPI ve Öneriler")
    if sim_mode:
        st.info("🧪 Simülasyon Modu Aktif. Arc Optimizer çıktıları simüle edilen veri üzerinden hesaplanır.")

    df = to_df(get_active_data(sim_mode))
    if df.empty:
        st.info("Önce veri oluşturun (simülasyon veya canlı kayıt).")
        return

    kpi = kpi_pack(df)
    last = kpi["last"]

    # ÜST KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Son Şarj kWh/t", f"{float(last.get('kwh_per_t')):.1f}" if pd.notna(last.get("kwh_per_t")) else "-")
    c2.metric("Son Şarj Elektrot", f"{float(last.get('electrode_kg_per_heat')):.2f} kg/şarj" if pd.notna(last.get("electrode_kg_per_heat")) else "-")
    c3.metric("Son Tap Sıcaklığı", f"{float(last.get('tap_temp_c')):.0f} °C" if pd.notna(last.get("tap_temp_c")) else "-")
    c4.metric("Son 10 Şarj Ort. kWh/t", f"{kpi['avg_kwh_t_10']:.1f}" if not np.isnan(kpi["avg_kwh_t_10"]) else "-")

    left, right = st.columns([3, 2])

    # -------------------------
    # SOL KOLON
    # -------------------------
    with left:
        st.markdown("### 🎛️ Operasyon Paneli")

        st.markdown("#### 📌 Özet (Dağılım + Son 3 Ortalama)")
        summ = distro_summary(df)
        if summ.empty:
            st.info("Özet için en az birkaç dolu kayıt gerekir.")
        else:
            st.table(summ)

        st.markdown("#### 🚨 Proses Durumu / Alarmlar")
        alarms = []

        if "kwh_per_t" in df.columns and df["kwh_per_t"].notna().sum() >= 10 and pd.notna(last.get("kwh_per_t")):
            ref = df["kwh_per_t"].dropna().tail(10).mean()
            if float(last["kwh_per_t"]) > ref * 1.05:
                alarms.append("⚡ kWh/t son 10 ortalamasına göre yüksek (+5%)")

        if "tap_temp_c" in df.columns and df["tap_temp_c"].notna().sum() >= 10 and pd.notna(last.get("tap_temp_c")):
            refT = df["tap_temp_c"].dropna().tail(10).mean()
            if abs(float(last["tap_temp_c"]) - float(refT)) > 15:
                alarms.append("🔥 Tap sıcaklığı sapması > 15°C")

        if pd.notna(last.get("panel_delta_t_c")) and float(last.get("panel_delta_t_c")) > 25:
            alarms.append("💧 Panel ΔT yüksek (>25°C)")

        if last.get("slag_foaming_index") is not None and pd.notna(last.get("slag_foaming_index")):
            if float(last["slag_foaming_index"]) >= 9:
                alarms.append("🌋 Slag foaming aşırı (≥9)")

        if alarms:
            for a in alarms:
                st.warning(a)
        else:
            st.success("✅ Proses stabil – belirgin alarm yok")

        st.markdown("### Proses Trendi")
        trend_chart(df, cols=("kwh_per_t", "tap_temp_c", "electrode_kg_per_heat", "panel_delta_t_c", "slag_foaming_index"), height=420)

    # -------------------------
    # SAĞ KOLON: AI MODEL + WHAT-IF
    # -------------------------
    with right:
        st.markdown("### 🤖 AI Model / Eğitim Modu")

        train_mode = st.radio(
            "Eğitim Modu",
            ["Model Eğit", "Sürekli Eğit", "Dijital İkiz Modu"],
            index=2,
            key="train_mode_arc",
        )

        current_rows = len(df)
        progress_ratio = min(current_rows / DIGITAL_TWIN_TARGET_HEATS, 1.0) if DIGITAL_TWIN_TARGET_HEATS else 0.0

        st.caption(f"Veri ilerleme durumu: **{current_rows} / {DIGITAL_TWIN_TARGET_HEATS}** şarj")
        st.progress(progress_ratio)
        st.caption(f"Eğitim ilerlemesi: **%{progress_ratio*100:.1f}**")

        if train_mode == "Model Eğit":
            st.caption("Mevcut veri setiyle modeli bir kez eğitir (demo / PoC).")
            st.caption(f"Mevcut veri sayısı: {current_rows} şarj (önerilen ≥ 20).")
            if st.button("Modeli Eğit", key="btn_train_manual"):
                train_arc_model(df, note="(Model Eğit)", min_samples=20)

        elif train_mode == "Sürekli Eğit":
            st.caption("Her sayfa yenilemesinde mevcut veriyle model güncellenir (demo modu).")
            train_arc_model(df, note="(Sürekli Eğit)", min_samples=20, silent=True)

        elif train_mode == "Dijital İkiz Modu":
            st.caption(
                f"Dijital ikiz: {DIGITAL_TWIN_HISTORICAL_HEATS} historical ile başlar, "
                f"veri geldikçe {DIGITAL_TWIN_TARGET_HEATS} hedefe kadar öğrenir."
            )

            if current_rows < DIGITAL_TWIN_MIN_START:
                st.warning(f"Dijital ikiz için en az {DIGITAL_TWIN_MIN_START} şarj gerekiyor; şu an {current_rows} var.")
            else:
                if current_rows > int(st.session_state.model_last_trained_rows_marker):
                    train_arc_model(df, note="(Dijital İkiz)", min_samples=DIGITAL_TWIN_MIN_START, silent=True)

                if current_rows < DIGITAL_TWIN_TARGET_HEATS:
                    st.session_state.model_status = (
                        f"Dijital ikiz öğrenme aşamasında "
                        f"(%{progress_ratio*100:.1f} – {current_rows}/{DIGITAL_TWIN_TARGET_HEATS})"
                    )
                else:
                    st.session_state.model_status = "Dijital ikiz hazır ✅ (10.000 şarj ile eğitildi)"

        st.write(f"**Durum:** {st.session_state.model_status}")
        if st.session_state.model_last_train_time:
            st.caption(
                f"Son eğitim: {st.session_state.model_last_train_time} · "
                f"Veri sayısı: {st.session_state.model_last_train_rows} · "
                f"Toplam eğitim: {st.session_state.model_train_count}"
            )

        # ✅ What-if
        st.markdown("---")
        st.markdown("### 🧪 What-if Simülasyonu (Arc Optimizer)")

        model, feat_cols, target_cols = load_arc_model()
        if model is None or feat_cols is None:
            st.info("What-if için önce modeli eğitin (en az ~20 şarj).")
        else:
            last_row = df.iloc[-1]

            def num_input(name, col, min_v, max_v, step, fmt="%.1f"):
                raw = last_row.get(col, (min_v + max_v) / 2)
                try:
                    v = float(raw)
                except Exception:
                    v = float((min_v + max_v) / 2)
                v = max(min_v, min(v, max_v))
                return st.number_input(name, min_v, max_v, v, step=step, format=fmt, key=f"whatif_{col}")

            w1, w2 = st.columns(2)
            with w1:
                tap_weight = num_input("Tap Weight (t)", "tap_weight_t", 20.0, 60.0, 0.5)
                duration = num_input("Süre (dk)", "duration_min", 30.0, 90.0, 1.0, "%.0f")
                energy = num_input("Enerji (kWh)", "energy_kwh", 500.0, 30000.0, 50.0)
                o2_flow = num_input("O2 (Nm³/h)", "o2_flow_nm3h", 300.0, 3000.0, 10.0)
            with w2:
                slag = num_input("Slag Foaming (0–10)", "slag_foaming_index", 0.0, 10.0, 0.5)
                panel_dT = num_input("Panel ΔT (°C)", "panel_delta_t_c", 0.0, 60.0, 0.5)
                elec = num_input("Elektrot (kg/şarj)", "electrode_kg_per_heat", 0.5, 6.0, 0.05)

            if st.button("Simülasyonu Çalıştır", key="btn_run_whatif"):
                inp = {
                    "tap_weight_t": tap_weight,
                    "duration_min": duration,
                    "energy_kwh": energy,
                    "o2_flow_nm3h": o2_flow,
                    "slag_foaming_index": slag,
                    "panel_delta_t_c": panel_dT,
                    "electrode_kg_per_heat": elec,
                }
                row_df = pd.DataFrame([inp])[feat_cols].fillna(0.0)
                try:
                    preds = model.predict(row_df)[0]
                    pred_dict = dict(zip(target_cols, preds))
                    kwh_pred = float(pred_dict.get("kwh_per_t", float("nan")))
                    tap_pred = float(pred_dict.get("tap_temp_c", float("nan")))

                    r1, r2 = st.columns(2)
                    r1.metric("AI kWh/t", f"{kwh_pred:.1f}")
                    r2.metric("AI Tap T", f"{tap_pred:.0f} °C")
                except Exception as e:
                    st.error(f"Tahmin hatası: {e}")

        st.markdown("---")
        st.markdown("### 💰 Proses Kazanç (Ton Başına)")
        m = money_pack(df)
        st.metric("Tahmini €/t (kaba)", f"{m['eur_per_t']:.2f}")
        st.metric("Tahmini €/yıl (kaba)", f"{m['eur_per_year']:,.0f}")


# =========================================================
# LAB – Simülasyon / Adhoc (İleri seviye)
# =========================================================
def show_lab_simulation(sim_mode: bool):
    st.markdown("## Lab – Simülasyon / Adhoc Analiz (İleri Seviye)")
    st.caption("Bu sayfa demo/Ar-Ge amaçlıdır. Yönetim ekranı değildir.")

    if not sim_mode:
        st.warning("Lab sayfası simülasyon modu için tasarlandı. Sidebar’dan Simülasyon Modu’nu aç.")
        return

    ensure_simulation_data_initialized()

    st.markdown("### 🔄 Veri Akışı Kontrolü")

    batch = st.slider("Akış hızı (şarj/adım)", 1, 500, SIM_STREAM_BATCH_DEFAULT, 1, key="lab_batch")

    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    with c1:
        st.toggle("9000 şarjı zamanla oku", key="sim_stream_enabled")
    with c2:
        st.toggle("Otomatik ilerlet", key="sim_stream_autostep")
    with c3:
        st.toggle("Auto-refresh", key="sim_stream_autorefresh")

    if st.session_state.sim_stream_autorefresh:
        st.session_state.sim_stream_refresh_sec = st.number_input(
            "Auto-refresh (sn)", min_value=1, max_value=60,
            value=int(st.session_state.sim_stream_refresh_sec), step=1
        )
        html_autorefresh(int(st.session_state.sim_stream_refresh_sec))

    b1, b2, b3 = st.columns([1.2, 1.2, 2.0])
    with b1:
        if st.button("▶️ İlerlet (1 adım)", key="lab_advance"):
            advance_sim_stream(batch)
            st.rerun()
    with b2:
        if st.button("⟲ Reset (1000’e dön)", key="lab_reset"):
            reset_sim_to_1000()
            st.rerun()
    with b3:
        st.caption(f"Akış ilerleme: {int(st.session_state.sim_stream_progress)} / {SIM_STREAM_TOTAL}")

    if st.session_state.sim_stream_enabled and st.session_state.sim_stream_autostep:
        cur = int(st.session_state.sim_stream_progress)
        if st.session_state.sim_stream_last_step_progress != cur:
            st.session_state.sim_stream_last_step_progress = cur
            advance_sim_stream(batch)

    df = to_df(st.session_state.sim_data)
    if df.empty:
        st.info("Veri yok.")
        return

    st.markdown("### İstatistik (etiketsiz — dağılım)")
    summ = distro_summary(df)
    if not summ.empty:
        st.table(summ)

    st.markdown("### Trend")
    trend_chart(
        df,
        cols=("kwh_per_t", "tap_temp_c", "electrode_kg_per_heat", "panel_delta_t_c", "o2_flow_nm3h", "slag_foaming_index"),
        height=440,
    )

    with st.expander("Ham tablo (lab)"):
        st.dataframe(df.tail(200), use_container_width=True)


# =========================================================
# PERSONA SAYFALARI
# =========================================================
def show_exec_page(sim_mode: bool):
    st.markdown("## Executive (CEO / CFO) – Value & Risk")

    df = to_df(get_active_data(sim_mode))
    if df.empty:
        st.info("Veri yok.")
        return

    m = money_pack(df)
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Tahmini €/t", f"{m['eur_per_t']:.2f}")
    a2.metric("Tahmini €/yıl", f"{m['eur_per_year']:,.0f}")
    a3.metric("Model", "Ready" if os.path.exists(MODEL_SAVE_PATH) else "Needs training")
    a4.metric("Veri (şarj)", f"{len(df)}")

    st.markdown("### Büyük Resim (Trend)")
    trend_chart(df, cols=("kwh_per_t", "electrode_kg_per_t"), height=320)

    st.markdown("### Özet (Dağılım)")
    summ = distro_summary(df)
    if not summ.empty:
        st.table(summ)


def show_plant_manager_page(sim_mode: bool):
    st.markdown("## Plant Manager – KPI & Performans Özeti")

    df = to_df(get_active_data(sim_mode))
    if df.empty:
        st.info("Veri yok.")
        return

    kpi = kpi_pack(df)
    m = money_pack(df)

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("İzlenen Şarj", f"{kpi['rows']}")
    top2.metric("Son10 kWh/t", f"{kpi['avg_kwh_t_10']:.1f}" if not np.isnan(kpi["avg_kwh_t_10"]) else "-")
    top3.metric("Son10 Elektrot kg/t", f"{kpi['avg_elec_pt_10']:.3f}" if not np.isnan(kpi["avg_elec_pt_10"]) else "-")
    top4.metric("Potansiyel (€/t)", f"{m['eur_per_t']:.2f}")

    st.markdown("### Trend")
    trend_chart(df, cols=("kwh_per_t", "tap_temp_c", "electrode_kg_per_t"), height=360)

    st.markdown("### Dağılım Özeti")
    summ = distro_summary(df)
    if not summ.empty:
        st.table(summ)


def show_operator_page(sim_mode: bool):
    st.markdown("## Engineer / Operator – Batch Dashboard")

    df = to_df(get_active_data(sim_mode))
    if df.empty:
        st.info("Veri yok.")
        return

    left, right = st.columns([1.2, 2.8])

    with left:
        st.markdown("### Batch Listesi")
        show = df[["timestamp_dt", "heat_id"]].dropna().tail(200).copy()
        show["label"] = show["heat_id"].astype(str) + " · " + show["timestamp_dt"].dt.strftime("%d.%m %H:%M")
        labels = show["label"].tolist()
        if not labels:
            st.info("Liste boş.")
            return
        sel = st.selectbox("Seç", labels, index=len(labels) - 1, key="op_batch_sel")
        sel_heat = sel.split(" · ")[0].strip()

    with right:
        st.markdown("### Batch Özeti")
        row = df[df["heat_id"] == sel_heat].tail(1)
        if row.empty:
            st.info("Batch bulunamadı.")
            return
        r = row.iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("kWh/t", f"{float(r.get('kwh_per_t')):.1f}" if pd.notna(r.get("kwh_per_t")) else "-")
        k2.metric("Tap T", f"{float(r.get('tap_temp_c')):.0f} °C" if pd.notna(r.get("tap_temp_c")) else "-")
        k3.metric("Elektrot", f"{float(r.get('electrode_kg_per_heat')):.2f} kg/şarj" if pd.notna(r.get("electrode_kg_per_heat")) else "-")
        k4.metric("Panel ΔT", f"{float(r.get('panel_delta_t_c')):.1f} °C" if pd.notna(r.get("panel_delta_t_c")) else "-")

        st.markdown("#### Trend (son 100)")
        trend_chart(df.tail(100), cols=("kwh_per_t", "tap_temp_c", "panel_delta_t_c", "slag_foaming_index"), height=320)

        st.markdown("#### Alarmlar (rule)")
        alarms = []
        if pd.notna(r.get("panel_delta_t_c")) and float(r.get("panel_delta_t_c")) > 25:
            alarms.append("• Panel ΔT yüksek (>25°C)")
        if pd.notna(r.get("slag_foaming_index")) and float(r.get("slag_foaming_index")) >= 9:
            alarms.append("• Slag foaming aşırı (≥9)")
        if pd.notna(r.get("kwh_per_t")) and df["kwh_per_t"].notna().sum() >= 10:
            ref = df["kwh_per_t"].dropna().tail(10).mean()
            if float(r.get("kwh_per_t")) > ref * 1.05:
                alarms.append("• kWh/t son10 ort üstünde (+5%)")
        st.write("\n".join(alarms) if alarms else "✅ Alarm yok")


# =========================================================
# SIDEBAR: NAV + HIZLI SİM AKIŞ
# =========================================================
def sidebar_controls():
    st.markdown("### FeCr AI")

    sim_mode = st.toggle(
        "Simülasyon Modu",
        value=True,
        help="Açıkken sistem canlı veri yerine simülasyon veri kullanır.",
        key="sidebar_sim_mode",
    )

    if sim_mode:
        ensure_simulation_data_initialized()
    else:
        st.session_state.sim_data = None

    st.divider()

    st.radio(
        "Görünüm",
        ["Persona", "Klasik Sayfalar"],
        index=0 if st.session_state.view_mode == "Persona" else 1,
        key="view_mode",
    )

    if st.session_state.view_mode == "Persona":
        st.selectbox(
            "Persona",
            ["CEO / CFO", "Plant Manager", "Engineer / Operator", "Lab (Advanced)"],
            index=["CEO / CFO", "Plant Manager", "Engineer / Operator", "Lab (Advanced)"].index(st.session_state.persona),
            key="persona",
        )
    else:
        st.selectbox(
            "Sayfa",
            ["Setup", "Canlı Veri", "ArcOptimizer", "Lab (Advanced)"],
            index=["Setup", "Canlı Veri", "ArcOptimizer", "Lab (Advanced)"].index(st.session_state.classic_page),
            key="classic_page",
        )

    st.divider()

    # Sim akış quick controls
    if sim_mode:
        st.markdown("### 🔄 Hızlı Akış")
        batch = st.slider("Batch (şarj/adım)", 1, 500, SIM_STREAM_BATCH_DEFAULT, 1, key="sidebar_batch")
        st.toggle("9000 şarjı zamanla oku", key="sim_stream_enabled")
        st.toggle("Otomatik ilerlet", key="sim_stream_autostep")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ İlerlet", key="sb_advance"):
                advance_sim_stream(int(batch))
                st.rerun()
        with c2:
            if st.button("⟲ Reset", key="sb_reset"):
                reset_sim_to_1000()
                st.rerun()

        st.caption(f"İlerleme: {int(st.session_state.sim_stream_progress)} / {SIM_STREAM_TOTAL}")

        # Autostep: aynı progress’te tekrar etmesin
        if st.session_state.sim_stream_enabled and st.session_state.sim_stream_autostep:
            cur = int(st.session_state.sim_stream_progress)
            if st.session_state.sim_stream_last_step_progress != cur:
                st.session_state.sim_stream_last_step_progress = cur
                advance_sim_stream(int(batch))

    return sim_mode


# =========================================================
# MAIN
# =========================================================
def main():
    with st.sidebar:
        sim_mode = sidebar_controls()

    if st.session_state.view_mode == "Persona":
        p = st.session_state.persona
        if p == "CEO / CFO":
            show_exec_page(sim_mode)
        elif p == "Plant Manager":
            show_plant_manager_page(sim_mode)
        elif p == "Engineer / Operator":
            show_operator_page(sim_mode)
        else:
            show_lab_simulation(sim_mode)
    else:
        page = st.session_state.classic_page
        if page == "Setup":
            show_setup_form()
        elif page == "Canlı Veri":
            show_runtime_page(sim_mode)
        elif page == "ArcOptimizer":
            show_arc_optimizer_page(sim_mode)
        else:
            show_lab_simulation(sim_mode)


if __name__ == "__main__":
    main()
