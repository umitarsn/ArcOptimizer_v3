import os
import json
import random
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import joblib


# ----------------------------------------------
# GENEL AYARLAR
# ----------------------------------------------
st.set_page_config(
    page_title="FeCr AI",
    page_icon="apple-touch-icon.png",
    layout="wide",
)

# ✅ Sidebar genişlik fix (uzun Türkçe metinler harf harf bölünmesin)
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { width: 340px !important; }
    section[data-testid="stSidebar"] > div { width: 340px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

TZ = ZoneInfo("Europe/Istanbul")

SETUP_SAVE_PATH = "data/saved_inputs.json"
RUNTIME_SAVE_PATH = "data/runtime_data.json"
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

MODEL_SAVE_PATH = "models/arc_optimizer_model.pkl"

# Dijital ikiz hedefleri
DIGITAL_TWIN_HISTORICAL_HEATS = 1000   # ilk historical
DIGITAL_TWIN_TARGET_HEATS = 10000      # toplam hedef (1000 + 9000)
DIGITAL_TWIN_MIN_START = 1000          # dijital ikiz eğitimine başlamak için min şarj

# Simülasyon akışı
SIM_STREAM_TOTAL = DIGITAL_TWIN_TARGET_HEATS
SIM_STREAM_BATCH_DEFAULT = 25  # her “ilerlet”te eklenecek heat sayısı


# ----------------------------------------------
# SESSION STATE (DEFAULTS)
# ----------------------------------------------
if "info_state" not in st.session_state:
    st.session_state.info_state = {}

if "profit_info_state" not in st.session_state:
    st.session_state.profit_info_state = {}

# Simülasyon cache
if "sim_data" not in st.session_state:
    st.session_state.sim_data = None

if "sim_full_data" not in st.session_state:
    st.session_state.sim_full_data = None

if "sim_mode_flag" not in st.session_state:
    st.session_state.sim_mode_flag = None

# ✅ default AÇIK
if "sim_stream_enabled" not in st.session_state:
    st.session_state.sim_stream_enabled = True

if "sim_stream_progress" not in st.session_state:
    st.session_state.sim_stream_progress = DIGITAL_TWIN_HISTORICAL_HEATS

# ✅ default AÇIK
if "sim_stream_autostep" not in st.session_state:
    st.session_state.sim_stream_autostep = True

# ✅ Lab autorefresh defaults
if "sim_stream_autorefresh" not in st.session_state:
    st.session_state.sim_stream_autorefresh = False

if "sim_stream_refresh_sec" not in st.session_state:
    st.session_state.sim_stream_refresh_sec = 2

# ✅ autostep’in aynı progress'te tekrar çalışmasını engelle
if "sim_stream_last_step_progress" not in st.session_state:
    st.session_state.sim_stream_last_step_progress = None

# Model eğitim durumu
if "model_status" not in st.session_state:
    st.session_state.model_status = "Henüz eğitilmedi."
    st.session_state.model_last_train_time = None
    st.session_state.model_last_train_rows = 0
    st.session_state.model_train_count = 0

if "model_last_trained_rows_marker" not in st.session_state:
    st.session_state.model_last_trained_rows_marker = 0


# ----------------------------------------------
# KAYITLI SETUP VERİLERİ
# ----------------------------------------------
if os.path.exists(SETUP_SAVE_PATH):
    with open(SETUP_SAVE_PATH, "r", encoding="utf-8") as f:
        saved_inputs = json.load(f)
else:
    saved_inputs = {}


# ----------------------------------------------
# RUNTIME VERİLERİ
# ----------------------------------------------
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


# ----------------------------------------------
# SİMÜLASYON VERİLERİ
# ----------------------------------------------
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
    electrode_cons = 1.9 + random.uniform(-0.3, 0.3) # kg/şarj

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
    }


def generate_simulation_full_data(total_n: int = SIM_STREAM_TOTAL):
    # 1 heat ~ 60 dk (1000 heat ~ 41-42 gün)
    step_minutes = 60
    now = datetime.now(TZ)
    start = now - timedelta(minutes=step_minutes * (total_n - 1))

    data = []
    for i in range(total_n):
        ts = start + timedelta(minutes=step_minutes * i)
        data.append(_make_heat_row(ts, i))
    return data


def ensure_simulation_data_initialized():
    if st.session_state.sim_full_data is None:
        st.session_state.sim_full_data = generate_simulation_full_data(SIM_STREAM_TOTAL)

    if st.session_state.sim_data is None:
        st.session_state.sim_stream_progress = DIGITAL_TWIN_HISTORICAL_HEATS
        st.session_state.sim_data = st.session_state.sim_full_data[:DIGITAL_TWIN_HISTORICAL_HEATS]


def reset_sim_stream():
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


# ----------------------------------------------
# MODEL FONKSİYONLARI
# ----------------------------------------------
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

    X = sub[feature_cols].fillna(sub[feature_cols].mean())
    y = sub[target_cols]

    if len(X) < 10:
        return None, None, None, None

    return X, y, feature_cols, target_cols


def train_arc_model(df: pd.DataFrame, note: str = "", min_samples: int = 20):
    st.session_state.model_status = "Eğitiliyor..."

    X, y, feature_cols, target_cols = get_arc_training_data(df)
    if X is None:
        st.session_state.model_status = "Eğitim için uygun veri bulunamadı."
        st.error("Model eğitimi için gerekli kolonlar yok veya yeterli dolu kayıt yok.")
        return False

    if len(X) < min_samples:
        st.session_state.model_status = f"Eğitim için veri yetersiz: {len(X)} şarj (gereken ≥ {min_samples})."
        st.warning(f"Bu mod için en az {min_samples} şarj gerekli, şu anda {len(X)} kayıt var.")
        return False

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=7,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "target_cols": target_cols},
        MODEL_SAVE_PATH,
    )

    now_str = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    rows = len(X)

    st.session_state.model_status = f"Eğitildi ✅ {note}".strip()
    st.session_state.model_last_train_time = now_str
    st.session_state.model_last_train_rows = rows
    st.session_state.model_train_count += 1
    st.session_state.model_last_trained_rows_marker = rows

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


# ----------------------------------------------
# EXCEL – SETUP
# ----------------------------------------------
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


# ----------------------------------------------
# 1) SETUP SAYFASI
# ----------------------------------------------
def show_setup_form():
    st.markdown("## 1. Setup – Sabit Proses / Tasarım Verileri")
    st.markdown(
        "Bu form **dc_saf_soru_tablosu.xlsx** dosyasına göre hazırlanmıştır.\n\n"
        "1. Girişi sadece **Set Değeri** alanına yapınız.\n"
        "2. 🔴 Zorunlu (Önem: 1), 🟡 Faydalı (Önem: 2), ⚪ Opsiyonel (Önem: 3).\n"
        "3. Detaylı bilgi için satır sonundaki ℹ️ butonuna tıklayın."
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
                        new_val = st.text_input(
                            label="",
                            value=current_val,
                            key=val_key,
                            label_visibility="collapsed",
                        )
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
                        detaylar.append("🔷 **Detaylı Açıklama:** " + da)
                    vk = row.get("Veri Kaynağı")
                    if isinstance(vk, str) and vk.strip():
                        detaylar.append("📌 **Kaynak:** " + vk)
                    ka = row.get("Kayıt Aralığı")
                    if isinstance(ka, str) and ka.strip():
                        detaylar.append("⏱️ **Kayıt Aralığı:** " + ka)
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

    st.sidebar.subheader("📊 Setup Veri Giriş Durumu")
    pct_all = round(100 * total_filled / total_fields, 1) if total_fields else 0
    pct_req = round(100 * required_filled / required_fields, 1) if required_fields else 0
    st.sidebar.metric("Toplam Giriş Oranı", f"{pct_all}%")
    st.sidebar.progress(min(pct_all / 100, 1.0))
    st.sidebar.metric("Zorunlu Veri Girişi", f"{pct_req}%")
    st.sidebar.progress(min(pct_req / 100, 1.0))
    eksik = required_fields - required_filled
    if eksik > 0:
        st.sidebar.warning(f"❗ Eksik Zorunlu Değerler: {eksik}")


# ----------------------------------------------
# 2) CANLI VERİ
# ----------------------------------------------
def show_runtime_page(sim_mode: bool):
    st.markdown("## 2. Canlı Veri – Şarj Bazlı Anlık Veriler")
    if sim_mode:
        st.info("🧪 **Simülasyon Modu Aktif.** Aşağıdaki veriler simülasyon datasıdır.")
    else:
        st.markdown("Bu sayfada her **şarj / heat** için veriler girilir veya otomasyondan okunur.")

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
                }
                runtime_data.append(new_entry)
                save_runtime_data(runtime_data)
                st.success(f"Şarj kaydı eklendi: {heat_id}")

    data_source = st.session_state.sim_data if sim_mode else runtime_data
    if not data_source:
        st.info("Henüz canlı veri yok.")
        return

    df = pd.DataFrame(data_source)
    try:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TZ)
        df = df.sort_values("timestamp_dt")
    except Exception:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp_dt")

    st.markdown("### Kayıtlı Veriler")
    st.dataframe(
        df[
            [
                "timestamp_dt",
                "heat_id",
                "tap_weight_t",
                "duration_min",
                "energy_kwh",
                "kwh_per_t",
                "tap_temp_c",
                "electrode_kg_per_heat",
                "slag_foaming_index",
                "panel_delta_t_c",
            ]
        ].rename(
            columns={
                "timestamp_dt": "Zaman",
                "heat_id": "Heat ID",
                "tap_weight_t": "Tap Weight (t)",
                "duration_min": "Süre (dk)",
                "energy_kwh": "Enerji (kWh)",
                "kwh_per_t": "kWh/t",
                "tap_temp_c": "Tap T (°C)",
                "electrode_kg_per_heat": "Elektrot (kg/şarj)",
                "slag_foaming_index": "Slag Foaming",
                "panel_delta_t_c": "Panel ΔT (°C)",
            }
        ),
        use_container_width=True,
    )


# ----------------------------------------------
# LAB – SİMÜLASYON / ADHOC ANALİZ
# ----------------------------------------------
def show_lab_simulation(sim_mode: bool):
    st.markdown("## Lab – Simülasyon / Adhoc Analiz (İleri Seviye)")
    st.caption("Bu sayfa demo/Ar-Ge amaçlıdır. Yönetim ekranı değildir.")

    if not sim_mode:
        st.warning("Lab sayfası simülasyon modu ile anlamlı. Sidebar’dan **Simülasyon Modu** aç.")
        return

    ensure_simulation_data_initialized()

    st.markdown("### 🔄 Veri Akışı Kontrolü")
    batch = st.slider(
        "Akış hızı (şarj / adım)",
        min_value=1,
        max_value=500,
        value=SIM_STREAM_BATCH_DEFAULT,
        step=1,
        key="lab_batch_slider",
    )

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        st.toggle("9000 şarjı zamanla oku", key="sim_stream_enabled")
    with c2:
        st.toggle("Otomatik ilerlet", key="sim_stream_autostep")
    with c3:
        st.toggle("Auto-refresh", key="sim_stream_autorefresh")

    st.number_input(
        "Auto-refresh (sn)",
        min_value=1,
        max_value=60,
        step=1,
        key="sim_stream_refresh_sec",
    )

    b1, b2, b3 = st.columns([2, 2, 6])
    with b1:
        if st.button("▶️ İlerlet (1 adım)"):
            advance_sim_stream(batch)
            st.rerun()
    with b2:
        if st.button("🔄 Reset (1000’e dön)"):
            reset_sim_stream()
            st.rerun()
    with b3:
        st.caption(f"Akış ilerleme: {int(st.session_state.sim_stream_progress)} / {SIM_STREAM_TOTAL}")

    # ✅ Autostep: aynı progress’te 2 kere çalışmasın
    if st.session_state.sim_stream_enabled and st.session_state.sim_stream_autostep:
        cur = int(st.session_state.sim_stream_progress)
        if st.session_state.sim_stream_last_step_progress != cur:
            st.session_state.sim_stream_last_step_progress = cur
            advance_sim_stream(batch)

    # ✅ Auto-refresh: streamlit core'da st.autorefresh yok → sleep + rerun
    if (
        st.session_state.sim_stream_enabled
        and st.session_state.sim_stream_autostep
        and st.session_state.sim_stream_autorefresh
    ):
        time.sleep(int(st.session_state.sim_stream_refresh_sec))
        st.rerun()

    # Quick adhoc
    df = pd.DataFrame(st.session_state.sim_data or [])
    if df.empty:
        st.info("Henüz veri yok.")
        return

    try:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TZ)
        df = df.sort_values("timestamp_dt")
    except Exception:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp_dt")

    st.markdown("### Adhoc – Trend")
    chart_df = df[["timestamp_dt", "kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"]].copy()
    long = chart_df.melt("timestamp_dt", var_name="var", value_name="val").dropna()
    var_map = {
        "kwh_per_t": "kWh/t",
        "tap_temp_c": "Tap T (°C)",
        "electrode_kg_per_heat": "Elektrot (kg/şarj)",
    }
    long["var_name"] = long["var"].map(var_map)

    ch = (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("timestamp_dt:T", title="Zaman"),
            y=alt.Y("val:Q", title=None),
            color=alt.Color("var_name:N", title="Değişken"),
        )
        .properties(height=360)
    )
    st.altair_chart(ch.interactive(), use_container_width=True)

    st.markdown("### Kayıtlar (son 30)")
    st.dataframe(df.tail(30), use_container_width=True)


# ----------------------------------------------
# 3) ARC OPTIMIZER (POC)
# ----------------------------------------------
def show_arc_optimizer_page(sim_mode: bool):
    st.markdown("## Arc Optimizer (PoC) – KPI, Trend ve What-If")

    data_source = st.session_state.sim_data if sim_mode else runtime_data
    if not data_source:
        st.info("Önce veri gerekli. (Simülasyon modunda Lab/Sidebar üzerinden ilerlet veya Canlı Veri sayfasından gir.)")
        return

    df = pd.DataFrame(data_source)
    try:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TZ)
        df = df.sort_values("timestamp_dt")
    except Exception:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp_dt")

    # ✅ PoC sayfasına şarj ilerletme (istenen)
    if sim_mode:
        with st.expander("🔄 Simülasyon Akışı (PoC)", expanded=False):
            batch = st.slider(
                "Akış hızı (şarj)",
                1, 500, SIM_STREAM_BATCH_DEFAULT, 1,
                key="poc_batch_slider",
            )
            c1, c2, c3 = st.columns([2, 2, 6])
            with c1:
                if st.button("▶️ Şarjı İlerlet"):
                    advance_sim_stream(batch)
                    st.rerun()
            with c2:
                if st.button("🔄 Reset"):
                    reset_sim_stream()
                    st.rerun()
            with c3:
                st.caption(f"{st.session_state.sim_stream_progress} / {SIM_STREAM_TOTAL}")

    last = df.iloc[-1]
    last_n = df.tail(25)

    # KPI'lar
    avg_kwh_t = float(last_n["kwh_per_t"].dropna().mean()) if last_n["kwh_per_t"].notna().any() else np.nan
    avg_electrode = float(last_n["electrode_kg_per_heat"].dropna().mean()) if last_n["electrode_kg_per_heat"].notna().any() else np.nan
    avg_tap_temp = float(last_n["tap_temp_c"].dropna().mean()) if last_n["tap_temp_c"].notna().any() else np.nan

    # Üst KPI satırı
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Son Şarj kWh/t", f"{last['kwh_per_t']:.1f}" if pd.notna(last.get("kwh_per_t")) else "-")
    c2.metric("Son Şarj Elektrot", f"{last['electrode_kg_per_heat']:.2f} kg/şarj" if pd.notna(last.get("electrode_kg_per_heat")) else "-")
    c3.metric("Son Tap Sıcaklığı", f"{last['tap_temp_c']:.0f} °C" if pd.notna(last.get("tap_temp_c")) else "-")
    c4.metric("Son 25 Ort. kWh/t", f"{avg_kwh_t:.1f}" if pd.notna(avg_kwh_t) else "-")

    # ✅ “En iyi/en kötü” psikolojisi yerine: dağılım istatistiği
    st.markdown("### 📈 KPI Dağılımı (son 100 şarj)")
    window = df.tail(100).copy()
    stats_rows = []

    def add_stats(name, series):
        s = series.dropna()
        if len(s) < 5:
            return
        stats_rows.append({
            "Gösterge": name,
            "P10": float(np.percentile(s, 10)),
            "P50 (Medyan)": float(np.percentile(s, 50)),
            "P90": float(np.percentile(s, 90)),
            "Ortalama": float(s.mean()),
        })

    add_stats("kWh/t", window["kwh_per_t"])
    add_stats("Elektrot (kg/şarj)", window["electrode_kg_per_heat"])
    add_stats("Tap T (°C)", window["tap_temp_c"])

    if stats_rows:
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)
    else:
        st.info("Dağılım istatistiği için yeterli veri yok.")

    # Trend grafiği (zaman görünür)
    st.markdown("### Proses Gidişatı – Zaman Trendi")
    trend_df = df[["timestamp_dt", "kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"]].copy()
    long = trend_df.melt("timestamp_dt", var_name="var", value_name="val").dropna()
    var_map = {
        "kwh_per_t": "kWh/t",
        "tap_temp_c": "Tap T (°C)",
        "electrode_kg_per_heat": "Elektrot (kg/şarj)",
    }
    long["var_name"] = long["var"].map(var_map)
    ch = (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("timestamp_dt:T", title="Zaman", axis=alt.Axis(format="%d.%m %H:%M", labelAngle=-35)),
            y=alt.Y("val:Q", title=None),
            color=alt.Color("var_name:N", title="Değişken"),
        )
        .properties(height=420)
    )
    st.altair_chart(ch.interactive(), use_container_width=True)

    # ------------------------------------------
    # MODEL / EĞİTİM
    # ------------------------------------------
    st.markdown("### 🤖 Model / Eğitim")
    train_mode = st.radio(
        "Eğitim Modu",
        ["Model Eğit", "Sürekli Eğit", "Dijital İkiz Modu"],
        index=0,
        key="train_mode_arc",
        horizontal=True,
    )

    current_rows = len(df)
    progress_ratio = min(current_rows / DIGITAL_TWIN_TARGET_HEATS, 1.0) if DIGITAL_TWIN_TARGET_HEATS else 0.0
    st.caption(f"Veri ilerleme: **{current_rows} / {DIGITAL_TWIN_TARGET_HEATS}** | Eğitim ilerlemesi: **%{progress_ratio*100:.1f}**")
    st.progress(progress_ratio)

    if train_mode == "Model Eğit":
        if st.button("Modeli Eğit", key="btn_train_manual"):
            train_arc_model(df, note="(Model Eğit)", min_samples=20)

    elif train_mode == "Sürekli Eğit":
        train_arc_model(df, note="(Sürekli Eğit)", min_samples=20)

    elif train_mode == "Dijital İkiz Modu":
        if current_rows < DIGITAL_TWIN_MIN_START:
            st.warning(f"Dijital ikiz için min {DIGITAL_TWIN_MIN_START} şarj gerekli. Şu an: {current_rows}")
        else:
            if current_rows > int(st.session_state.model_last_trained_rows_marker):
                train_arc_model(df, note="(Dijital İkiz Modu)", min_samples=DIGITAL_TWIN_MIN_START)

    st.write(f"**Durum:** {st.session_state.model_status}")
    if st.session_state.model_last_train_time:
        st.caption(
            f"Son eğitim: {st.session_state.model_last_train_time} · "
            f"Veri: {st.session_state.model_last_train_rows} · "
            f"Eğitim sayısı: {st.session_state.model_train_count}"
        )

    # ------------------------------------------
    # ✅ WHAT-IF (istenen)
    # ------------------------------------------
    model, feat_cols, target_cols = load_arc_model()
    if model is not None and feat_cols is not None:
        st.markdown("### 🔮 What-If Simülasyonu (PoC)")

        last_row = df.iloc[-1]

        def w_input(label, col, mn, mx, step, fmt="%.2f"):
            raw = last_row.get(col, (mn + mx) / 2)
            try:
                v = float(raw)
            except Exception:
                v = float((mn + mx) / 2)
            v = max(mn, min(v, mx))
            return st.number_input(label, min_value=mn, max_value=mx, value=v, step=step, format=fmt)

        wc1, wc2 = st.columns(2)
        with wc1:
            tap_w = w_input("Tap Weight (t)", "tap_weight_t", 20.0, 60.0, 0.5, "%.1f")
            dur = w_input("Süre (dk)", "duration_min", 30.0, 90.0, 1.0, "%.0f")
            energy = w_input("Enerji (kWh)", "energy_kwh", 500.0, 30000.0, 50.0, "%.0f")
            o2 = w_input("O₂ (Nm³/h)", "o2_flow_nm3h", 300.0, 3000.0, 10.0, "%.0f")
        with wc2:
            slag = w_input("Slag Foaming (0–10)", "slag_foaming_index", 0.0, 10.0, 0.5, "%.1f")
            dT = w_input("Panel ΔT (°C)", "panel_delta_t_c", 0.0, 60.0, 0.5, "%.1f")
            elec = w_input("Elektrot (kg/şarj)", "electrode_kg_per_heat", 0.5, 6.0, 0.05, "%.2f")

        if st.button("What-If Çalıştır", key="btn_whatif_run"):
            X = pd.DataFrame([{
                "tap_weight_t": tap_w,
                "duration_min": dur,
                "energy_kwh": energy,
                "o2_flow_nm3h": o2,
                "slag_foaming_index": slag,
                "panel_delta_t_c": dT,
                "electrode_kg_per_heat": elec,
            }])[feat_cols].fillna(0.0)

            try:
                y_pred = model.predict(X)[0]
                pred = dict(zip(target_cols, y_pred))
                st.success("AI Tahmini:")
                st.write(f"- **kWh/t:** {float(pred.get('kwh_per_t', np.nan)):.1f}")
                st.write(f"- **Tap T:** {float(pred.get('tap_temp_c', np.nan)):.0f} °C")
            except Exception as e:
                st.error(f"What-If tahmini hata verdi: {e}")
    else:
        st.info("What-If için önce modeli eğit (Model Eğit).")


# ----------------------------------------------
# MAIN
# ----------------------------------------------
def main():
    with st.sidebar:
        st.markdown("### FeCr AI")

        sim_mode = st.toggle(
            "Simülasyon Modu",
            value=True,
            help="Açıkken sistem canlı veri yerine simülasyon veri kullanır.",
        )

        if sim_mode:
            ensure_simulation_data_initialized()
        else:
            st.session_state.sim_data = None

        page = st.radio(
            "Sayfa Seç",
            [
                "Arc Optimizer (PoC)",
                "2. Canlı Veri",
                "1. Setup",
                "Lab – Simülasyon/Adhoc",
            ],
            index=0,
        )

    if page == "1. Setup":
        show_setup_form()
    elif page == "2. Canlı Veri":
        show_runtime_page(sim_mode)
    elif page == "Lab – Simülasyon/Adhoc":
        show_lab_simulation(sim_mode)
    else:
        show_arc_optimizer_page(sim_mode)


if __name__ == "__main__":
    main()
