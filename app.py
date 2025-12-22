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
    unsafe_allow_html=True
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
# SESSION STATE INIT
# ----------------------------------------------
def _init_state():
    if "info_state" not in st.session_state:
        st.session_state.info_state = {}

    if "profit_info_state" not in st.session_state:
        st.session_state.profit_info_state = {}

    # Simülasyon cache
    if "sim_data" not in st.session_state:
        st.session_state.sim_data = None
    if "sim_full_data" not in st.session_state:
        st.session_state.sim_full_data = None

    # Defaults (toggle’lar buradan beslenir)
    if "sim_stream_enabled" not in st.session_state:
        st.session_state.sim_stream_enabled = True
    if "sim_stream_autostep" not in st.session_state:
        st.session_state.sim_stream_autostep = True

    if "sim_stream_progress" not in st.session_state:
        st.session_state.sim_stream_progress = DIGITAL_TWIN_HISTORICAL_HEATS

    # ✅ autostep’in aynı progress'te tekrar çalışmasını engelle
    if "sim_stream_last_step_progress" not in st.session_state:
        st.session_state.sim_stream_last_step_progress = None

    # Auto refresh (gerçek “zamanla akış” için şart)
    if "sim_stream_autorefresh" not in st.session_state:
        st.session_state.sim_stream_autorefresh = True
    if "sim_stream_refresh_sec" not in st.session_state:
        st.session_state.sim_stream_refresh_sec = 2

    # Model eğitim durumu
    if "model_status" not in st.session_state:
        st.session_state.model_status = "Henüz eğitilmedi."
        st.session_state.model_last_train_time = None
        st.session_state.model_last_train_rows = 0
        st.session_state.model_train_count = 0

    if "model_last_trained_rows_marker" not in st.session_state:
        st.session_state.model_last_trained_rows_marker = 0


_init_state()


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


def ensure_simulation_data_initialized(force_reset: bool = False):
    """
    force_reset=True: simülasyon datasını ve progress’i sıfırdan başlatır.
    """
    if st.session_state.sim_full_data is None:
        st.session_state.sim_full_data = generate_simulation_full_data(SIM_STREAM_TOTAL)

    if force_reset or st.session_state.sim_data is None:
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

    X = sub[feature_cols].fillna(sub[feature_cols].mean(numeric_only=True))
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
# COMMON HELPERS
# ----------------------------------------------
def get_data_source(sim_mode: bool):
    return st.session_state.sim_data if sim_mode else runtime_data


def build_df(data_source):
    df = pd.DataFrame(data_source)
    if df.empty:
        return df
    try:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TZ)
    except Exception:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp_dt")
    return df


def safe_pct(series: pd.Series, q: float):
    s = series.dropna()
    if len(s) == 0:
        return None
    return float(np.percentile(s.values, q))


def money_fmt(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "-"


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

    data_source = get_data_source(sim_mode)
    if not data_source:
        st.info("Henüz canlı veri yok.")
        return

    df = build_df(data_source)

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
# EXECUTIVE DASHBOARD (İşletme Müdürü)
# ----------------------------------------------
def show_executive_dashboard(sim_mode: bool):
    st.markdown("## Executive Dashboard – Büyük Resim (KPI + Kalite + Finans)")
    if sim_mode:
        st.info("🧪 Demo/Simülasyon datası gösteriliyor.")

    data_source = get_data_source(sim_mode)
    if not data_source:
        st.info("Önce veri ekleyin (2. Canlı Veri) veya Simülasyon Modu açın.")
        return

    df = build_df(data_source)
    if df.empty:
        st.info("Veri boş.")
        return

    # Dönem filtresi (basit)
    with st.expander("📅 Dönem / Filtre", expanded=True):
        c1, c2, c3 = st.columns([1.2, 1.2, 2.0])
        with c1:
            period = st.selectbox("Dönem", ["Son 7 gün", "Son 30 gün", "Tümü"], index=0)
        with c2:
            target_kwh_reduction = st.number_input("AI hedefi (kWh/t iyileşme)", value=5.0, step=1.0)
        with c3:
            st.caption("Not: Demo’da kalite değişkenleri KPI bandlarıyla temsil edilir. Gerçek projede grade/kimya/yield eklenir.")

    max_t = df["timestamp_dt"].max()
    if period == "Son 7 gün":
        dfp = df[df["timestamp_dt"] >= (max_t - timedelta(days=7))].copy()
    elif period == "Son 30 gün":
        dfp = df[df["timestamp_dt"] >= (max_t - timedelta(days=30))].copy()
    else:
        dfp = df.copy()

    if dfp.empty:
        st.warning("Seçilen dönemde veri yok.")
        return

    # KPI’lar
    kwh = dfp["kwh_per_t"].dropna()
    elec_pt = (dfp["electrode_kg_per_heat"] / dfp["tap_weight_t"]).replace([np.inf, -np.inf], np.nan)
    dur = dfp["duration_min"].dropna()
    tap = dfp["tap_temp_c"].dropna()

    last = dfp.iloc[-1]

    # Percentile bandları (kötü/iyi demeden)
    p10_kwh = safe_pct(kwh, 10)
    p50_kwh = safe_pct(kwh, 50)
    p90_kwh = safe_pct(kwh, 90)

    p10_e = safe_pct(elec_pt, 10)
    p50_e = safe_pct(elec_pt, 50)
    p90_e = safe_pct(elec_pt, 90)

    # Hedef bandı (basit): median - target_kwh_reduction
    target_kwh = (p50_kwh - float(target_kwh_reduction)) if p50_kwh is not None else None

    # Outlier sayacı: hedefin %5 üstü
    outlier_cnt = 0
    if target_kwh is not None:
        outlier_cnt = int((kwh > target_kwh * 1.05).sum())

    # Finans (€/t)
    ENERGY_PRICE_EUR_PER_KWH = 0.12
    ELECTRODE_PRICE_EUR_PER_KG = 3.0

    fin_rows = []
    total_gain_eur_per_t = 0.0

    if target_kwh is not None and len(kwh) > 0:
        # mevcut ortalama vs hedef
        real_avg = float(kwh.mean())
        diff = max(0.0, real_avg - float(target_kwh))
        gain = diff * ENERGY_PRICE_EUR_PER_KWH
        total_gain_eur_per_t += gain
        fin_rows.append({"Kalem": "Enerji", "Mevcut": f"{real_avg:.1f} kWh/t", "Hedef": f"{target_kwh:.1f} kWh/t", "Δ": f"{diff:.1f} kWh/t", "€ / t": f"{gain:.2f}"})

    if p50_e is not None and len(elec_pt.dropna()) > 0:
        # elektrot hedef: median - 0.05 kg/t (demo)
        real_avg_e = float(elec_pt.dropna().mean())
        target_e = max(0.0, float(p50_e) - 0.05)
        diff_e = max(0.0, real_avg_e - target_e)
        gain_e = diff_e * ELECTRODE_PRICE_EUR_PER_KG
        total_gain_eur_per_t += gain_e
        fin_rows.append({"Kalem": "Elektrot", "Mevcut": f"{real_avg_e:.3f} kg/t", "Hedef": f"{target_e:.3f} kg/t", "Δ": f"{diff_e:.3f} kg/t", "€ / t": f"{gain_e:.2f}"})

    # Üst KPI kartları
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Batch (seçili dönem)", f"{len(dfp)}")
    c2.metric("kWh/t (p10 / p50 / p90)", f"{p10_kwh:.1f} / {p50_kwh:.1f} / {p90_kwh:.1f}" if p10_kwh is not None else "-")
    c3.metric("Elektrot kg/t (p10 / p50 / p90)", f"{p10_e:.3f} / {p50_e:.3f} / {p90_e:.3f}" if p10_e is not None else "-")
    c4.metric("Hedef dışı batch sayısı", f"{outlier_cnt}" if target_kwh is not None else "-")

    st.markdown("### 📈 Trend (kWh/t) + Hedef Bandı")
    # Trend chart: kWh/t
    chart_df = dfp[["timestamp_dt", "kwh_per_t"]].dropna().copy()
    if chart_df.empty:
        st.info("Trend için yeterli kWh/t yok.")
    else:
        base = alt.Chart(chart_df).mark_line().encode(
            x=alt.X("timestamp_dt:T", title="Zaman"),
            y=alt.Y("kwh_per_t:Q", title="kWh/t"),
            tooltip=["timestamp_dt:T", "kwh_per_t:Q"],
        ).properties(height=320)

        layers = [base]

        if target_kwh is not None:
            tgt_df = pd.DataFrame({"timestamp_dt": [chart_df["timestamp_dt"].min(), chart_df["timestamp_dt"].max()], "target": [target_kwh, target_kwh]})
            tgt = alt.Chart(tgt_df).mark_rule(strokeDash=[6, 4]).encode(
                x="timestamp_dt:T",
                y=alt.Y("target:Q"),
                tooltip=[alt.Tooltip("target:Q", title="Hedef kWh/t")],
            )
            layers.append(tgt)

        st.altair_chart(alt.layer(*layers).interactive(), use_container_width=True)

    st.markdown("### 🧪 Kalite/Proses Dağılımı (Demo Bandları)")
    # Tap temp dağılımı + duration
    d1, d2 = st.columns(2)
    with d1:
        if tap.dropna().empty:
            st.info("Tap T yok.")
        else:
            tap_df = dfp[["timestamp_dt", "tap_temp_c"]].dropna()
            st.altair_chart(
                alt.Chart(tap_df).mark_line().encode(x="timestamp_dt:T", y="tap_temp_c:Q", tooltip=["timestamp_dt:T", "tap_temp_c:Q"]).properties(height=260),
                use_container_width=True
            )
    with d2:
        if dur.dropna().empty:
            st.info("Süre yok.")
        else:
            dur_df = dfp[["timestamp_dt", "duration_min"]].dropna()
            st.altair_chart(
                alt.Chart(dur_df).mark_line().encode(x="timestamp_dt:T", y="duration_min:Q", tooltip=["timestamp_dt:T", "duration_min:Q"]).properties(height=260),
                use_container_width=True
            )

    st.markdown("### 💰 Finansal Özet (€/t)")
    if fin_rows:
        st.table(pd.DataFrame(fin_rows))
        st.markdown(f"**Toplam potansiyel (demo hesap – €/t):** ≈ **{total_gain_eur_per_t:,.2f} €/t**")
    else:
        st.info("Finansal hesap için yeterli veri yok (kWh/t ve/veya elektrot kg/t).")

    st.markdown("### ✅ Yönetim İçin 3 Cümlelik Özet")
    # Kısa anlatı (otomatik)
    lines = []
    if target_kwh is not None and len(kwh) > 0:
        lines.append(f"- Seçili dönemde ortalama **{kwh.mean():.1f} kWh/t**, hedef bandı **{target_kwh:.1f} kWh/t** (demo hedef).")
    if outlier_cnt:
        lines.append(f"- **{outlier_cnt} batch** hedef bandının üstünde; odak bu batch’leri stabil band içine almak.")
    if fin_rows:
        lines.append(f"- Ölçülebilir potansiyel **≈ {total_gain_eur_per_t:,.2f} €/t** (enerji+elektrot).")
    if not lines:
        lines = ["- Veri arttıkça KPI+finans özetleri daha stabil hale gelecek."]
    for l in lines:
        st.markdown(l)


# ----------------------------------------------
# OPERATIONS DASHBOARD (Mühendis/Operatör)
# ----------------------------------------------
def show_operations_dashboard(sim_mode: bool):
    st.markdown("## Operations Dashboard – İzlenen Parametreler ve Stabilite")
    if sim_mode:
        st.info("🧪 Demo/Simülasyon datası gösteriliyor.")

    data_source = get_data_source(sim_mode)
    if not data_source:
        st.info("Önce veri ekleyin (2. Canlı Veri) veya Simülasyon Modu açın.")
        return

    df = build_df(data_source)
    if df.empty:
        st.info("Veri boş.")
        return

    with st.expander("🎯 İzlenen Parametreler / Karar Değişkenleri", expanded=False):
        st.markdown(
            "- **İzlenen parametreler:** kWh/t, elektrot, Tap T, O₂ debisi, Slag foaming, Panel ΔT, süre\n"
            "- **Karar değişkenleri (demo):** güç profili, O₂ stratejisi, köpük hedefi, bekleme/hold, ark stabilitesi\n"
            "- Bu sayfa **iyi/kötü** etiketlemez; **trend + dağılım + sapma** gösterir."
        )

    # KPI stats
    kwh = df["kwh_per_t"].dropna()
    elec_pt = (df["electrode_kg_per_heat"] / df["tap_weight_t"]).replace([np.inf, -np.inf], np.nan).dropna()
    tap = df["tap_temp_c"].dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Batch", f"{len(df)}")
    c2.metric("kWh/t (ort ± std)", f"{kwh.mean():.1f} ± {kwh.std():.1f}" if len(kwh) else "-")
    c3.metric("Elektrot kg/t (ort ± std)", f"{elec_pt.mean():.3f} ± {elec_pt.std():.3f}" if len(elec_pt) else "-")
    c4.metric("Tap T (ort ± std)", f"{tap.mean():.0f} ± {tap.std():.0f}" if len(tap) else "-")

    st.markdown("### 📈 Trendler")
    # Multi-line trend: kWh/t, Tap T, electrode kg/heat (not per ton)
    trend_cols = ["kwh_per_t", "tap_temp_c", "electrode_kg_per_heat", "o2_flow_nm3h", "slag_foaming_index", "panel_delta_t_c"]
    exist = [c for c in trend_cols if c in df.columns]
    plot_df = df[["timestamp_dt"] + exist].copy()
    long = plot_df.melt("timestamp_dt", var_name="var", value_name="val").dropna()

    if long.empty:
        st.info("Trend için veri yok.")
    else:
        var_map = {
            "kwh_per_t": "kWh/t",
            "tap_temp_c": "Tap T (°C)",
            "electrode_kg_per_heat": "Elektrot (kg/şarj)",
            "o2_flow_nm3h": "O₂ (Nm³/h)",
            "slag_foaming_index": "Slag foaming",
            "panel_delta_t_c": "Panel ΔT (°C)",
        }
        long["var_name"] = long["var"].map(var_map).fillna(long["var"])

        chart = alt.Chart(long).mark_line().encode(
            x=alt.X("timestamp_dt:T", title="Zaman"),
            y=alt.Y("val:Q", title=None),
            color=alt.Color("var_name:N", title="Değişken", legend=alt.Legend(orient="top")),
            tooltip=["timestamp_dt:T", "var_name:N", "val:Q"],
        ).properties(height=380)
        st.altair_chart(chart.interactive(), use_container_width=True)

    st.markdown("### 🧾 Batch Seç – Özet")
    # Batch selection for drill-down
    ids = df["heat_id"].astype(str).tolist()
    default_idx = max(0, len(ids) - 1)
    selected = st.selectbox("Batch / Heat seç", ids, index=default_idx)

    row = df[df["heat_id"].astype(str) == str(selected)].tail(1)
    if row.empty:
        st.info("Batch bulunamadı.")
        return

    r = row.iloc[0]
    s1, s2, s3 = st.columns(3)
    s1.metric("kWh/t", f"{r.get('kwh_per_t', np.nan):.1f}" if pd.notna(r.get("kwh_per_t")) else "-")
    s2.metric("Tap T (°C)", f"{r.get('tap_temp_c', np.nan):.0f}" if pd.notna(r.get("tap_temp_c")) else "-")
    s3.metric("Elektrot (kg/şarj)", f"{r.get('electrode_kg_per_heat', np.nan):.2f}" if pd.notna(r.get("electrode_kg_per_heat")) else "-")

    st.markdown("**Operatör Notu:** " + (str(r.get("operator_note", "")) if pd.notna(r.get("operator_note")) else "-"))

    # Minimal “yorum” (etiketlemeden)
    st.markdown("### 🔎 Kısa Teknik Not (etiketsiz)")
    notes = []
    if pd.notna(r.get("panel_delta_t_c")) and float(r["panel_delta_t_c"]) > 25:
        notes.append("- Panel ΔT yüksek; soğutma devresi/hat kısıtları kontrol edilebilir.")
    if pd.notna(r.get("slag_foaming_index")) and float(r["slag_foaming_index"]) >= 9:
        notes.append("- Köpük seviyesi üst bantta; karbon/O₂ dengesi gözlenebilir.")
    if not notes:
        notes.append("- Bu batch için belirgin “üst bant” sinyali yok (demo kriter).")
    for n in notes:
        st.markdown(n)


# ----------------------------------------------
# BATCH INSIGHTS (Batch list + dağılım)
# ----------------------------------------------
def show_batch_insights(sim_mode: bool):
    st.markdown("## Batch Insights – Liste + Dağılım + Drill-down")
    if sim_mode:
        st.info("🧪 Demo/Simülasyon datası gösteriliyor.")

    data_source = get_data_source(sim_mode)
    if not data_source:
        st.info("Önce veri ekleyin (2. Canlı Veri) veya Simülasyon Modu açın.")
        return

    df = build_df(data_source)
    if df.empty:
        st.info("Veri boş.")
        return

    # Derived columns
    df = df.copy()
    df["electrode_kg_per_t"] = (df["electrode_kg_per_heat"] / df["tap_weight_t"]).replace([np.inf, -np.inf], np.nan)

    st.markdown("### 📋 Batch Listesi")
    cols = ["timestamp_dt", "heat_id", "tap_weight_t", "kwh_per_t", "electrode_kg_per_t", "tap_temp_c", "duration_min", "o2_flow_nm3h", "slag_foaming_index", "panel_delta_t_c"]
    show_cols = [c for c in cols if c in df.columns]
    view = df[show_cols].rename(columns={
        "timestamp_dt": "Zaman",
        "heat_id": "Heat ID",
        "tap_weight_t": "Tap (t)",
        "kwh_per_t": "kWh/t",
        "electrode_kg_per_t": "Elektrot (kg/t)",
        "tap_temp_c": "Tap T (°C)",
        "duration_min": "Süre (dk)",
        "o2_flow_nm3h": "O₂ (Nm³/h)",
        "slag_foaming_index": "Foaming",
        "panel_delta_t_c": "Panel ΔT (°C)",
    })

    st.dataframe(view, use_container_width=True)

    st.markdown("### 🔵 Dağılım Analizi (Scatter)")
    c1, c2 = st.columns(2)
    with c1:
        if df["kwh_per_t"].notna().sum() > 5 and df["slag_foaming_index"].notna().sum() > 5:
            sc = alt.Chart(df.dropna(subset=["kwh_per_t", "slag_foaming_index"])).mark_circle(size=60).encode(
                x=alt.X("slag_foaming_index:Q", title="Slag foaming"),
                y=alt.Y("kwh_per_t:Q", title="kWh/t"),
                tooltip=["heat_id:N", "timestamp_dt:T", "kwh_per_t:Q", "slag_foaming_index:Q"]
            ).properties(height=300)
            st.altair_chart(sc.interactive(), use_container_width=True)
        else:
            st.info("kWh/t ve foaming için yeterli veri yok.")
    with c2:
        if df["electrode_kg_per_t"].notna().sum() > 5 and df["panel_delta_t_c"].notna().sum() > 5:
            sc2 = alt.Chart(df.dropna(subset=["electrode_kg_per_t", "panel_delta_t_c"])).mark_circle(size=60).encode(
                x=alt.X("panel_delta_t_c:Q", title="Panel ΔT (°C)"),
                y=alt.Y("electrode_kg_per_t:Q", title="Elektrot (kg/t)"),
                tooltip=["heat_id:N", "timestamp_dt:T", "electrode_kg_per_t:Q", "panel_delta_t_c:Q"]
            ).properties(height=300)
            st.altair_chart(sc2.interactive(), use_container_width=True)
        else:
            st.info("Elektrot kg/t ve Panel ΔT için yeterli veri yok.")

    st.markdown("### 🧾 Batch Drill-down")
    ids = df["heat_id"].astype(str).tolist()
    default_idx = max(0, len(ids) - 1)
    selected = st.selectbox("Detay için batch seç", ids, index=default_idx, key="batch_insights_pick")
    r = df[df["heat_id"].astype(str) == str(selected)].tail(1).iloc[0]
    detail = {
        "Zaman": r.get("timestamp_dt"),
        "Heat ID": r.get("heat_id"),
        "Tap (t)": r.get("tap_weight_t"),
        "kWh/t": r.get("kwh_per_t"),
        "Enerji (kWh)": r.get("energy_kwh"),
        "Süre (dk)": r.get("duration_min"),
        "Tap T (°C)": r.get("tap_temp_c"),
        "O₂ (Nm³/h)": r.get("o2_flow_nm3h"),
        "Foaming": r.get("slag_foaming_index"),
        "Panel ΔT (°C)": r.get("panel_delta_t_c"),
        "Elektrot (kg/şarj)": r.get("electrode_kg_per_heat"),
        "Elektrot (kg/t)": r.get("electrode_kg_per_t"),
        "Not": r.get("operator_note", ""),
    }
    st.json({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in detail.items()})


# ----------------------------------------------
# LAB / SIMULATION (Adhoc + akış kontrol)
# ----------------------------------------------
def show_lab_simulation(sim_mode: bool):
    st.markdown("## Lab – Simülasyon / Adhoc Analiz (İleri Seviye)")
    st.caption("Bu sayfa demo/Ar-Ge amaçlıdır. Yönetim ekranı değildir.")

    if not sim_mode:
        st.warning("Lab için Simülasyon Modu’nu aç.")
        return

    ensure_simulation_data_initialized()

    st.markdown("### 🔄 Veri Akışı Kontrolü")
    batch = st.slider(
        "Akış hızı (şarj / adım)",
        min_value=1,
        max_value=500,
        value=int(SIM_STREAM_BATCH_DEFAULT),
        step=1,
        key="lab_batch_slider",
    )

    c1, c2, c3 = st.columns([1.3, 1.0, 1.0])
    with c1:
        st.toggle(
            "9000 şarjı zamanla oku",
            key="sim_stream_enabled",
            help="Açıkken 1000 historical sonrası kalan veriyi batch ile ekleyerek akışı simüle eder.",
        )
    with c2:
        st.toggle(
            "Otomatik ilerlet",
            key="sim_stream_autostep",
            help="Açıkken sayfa yenilendiğinde bir kez batch kadar ilerler.",
        )
    with c3:
        st.toggle(
            "Auto-refresh",
            key="sim_stream_autorefresh",
            help="Açıkken otomatik yenileme yapar; 'zamanla oku' gerçekten akar.",
        )

    st.number_input(
        "Auto-refresh (sn)",
        min_value=1,
        max_value=30,
        value=int(st.session_state.sim_stream_refresh_sec),
        step=1,
        key="sim_stream_refresh_sec",
        help="Zamanla akışın hızı. 2-3 sn demo için iyi.",
    )

    b1, b2, b3 = st.columns([1.0, 1.0, 2.0])
    with b1:
        if st.button("▶️ İlerlet (1 adım)"):
            advanced = advance_sim_stream(batch)
            if not advanced:
                st.info("Akış tamamlandı: 10.000 / 10.000")
            st.rerun()
    with b2:
        if st.button("⟲ Reset (1000’e dön)"):
            ensure_simulation_data_initialized(force_reset=True)
            st.rerun()
    with b3:
        st.caption(f"Akış ilerleme: {int(st.session_state.sim_stream_progress)} / {SIM_STREAM_TOTAL}")

    # ✅ gerçek “zamanla akış” (autorefresh)
    if st.session_state.sim_stream_enabled and st.session_state.sim_stream_autostep:
        if st.session_state.sim_stream_autorefresh:
            st.autorefresh(interval=int(st.session_state.sim_stream_refresh_sec) * 1000, key="lab_autorefresh_tick")

        cur = int(st.session_state.sim_stream_progress)
        # aynı progress’te üst üste adım atmasın
        if st.session_state.sim_stream_last_step_progress != cur:
            st.session_state.sim_stream_last_step_progress = cur
            advance_sim_stream(batch)

    st.markdown("### 🧪 Adhoc Görünüm")
    df = build_df(st.session_state.sim_data or [])
    if df.empty:
        st.info("Simülasyon datası yok.")
        return

    st.dataframe(df.tail(50), use_container_width=True)


# ----------------------------------------------
# ARC OPTIMIZER (Legacy/PoC sayfa - istersen kaldır)
# ----------------------------------------------
def show_arc_optimizer_page(sim_mode: bool):
    st.markdown("## 3. Arc Optimizer (PoC) – Trendler, KPI ve Öneriler")
    st.caption("Not: Demo yorumlarına göre artık ana ekranlar Executive/Operations/Batch/Lab olarak ayrıldı.")
    if sim_mode:
        st.info("🧪 **Simülasyon Modu Aktif.**")

    data_source = get_data_source(sim_mode)
    if not data_source:
        st.info("Önce 2. sayfadan veri ekleyin veya Simülasyon Modu açın.")
        return

    df = build_df(data_source)
    if df.empty:
        st.info("Veri boş.")
        return

    last = df.iloc[-1]
    last_n = df.tail(10)

    avg_kwh_t = last_n["kwh_per_t"].dropna().mean()
    avg_electrode = last_n["electrode_kg_per_heat"].dropna().mean()
    avg_tap_temp = last_n["tap_temp_c"].dropna().mean()

    # Üst satır: sol KPI'lar, sağ model kutusu
    kpi_col, model_col = st.columns([3, 2])

    with kpi_col:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Son Şarj kWh/t", f"{last['kwh_per_t']:.1f}" if pd.notna(last.get("kwh_per_t")) else "-")
        c2.metric("Son Şarj Elektrot", f"{last['electrode_kg_per_heat']:.2f} kg/şarj" if pd.notna(last.get("electrode_kg_per_heat")) else "-")
        c3.metric("Son Tap Sıcaklığı", f"{last['tap_temp_c']:.0f} °C" if pd.notna(last.get("tap_temp_c")) else "-")
        c4.metric("Son 10 Şarj Ort. kWh/t", f"{avg_kwh_t:.1f}" if avg_kwh_t and not pd.isna(avg_kwh_t) else "-")

        st.markdown("### 🚨 Proses Durumu (Band yaklaşımı)")
        alarms = []
        if avg_kwh_t and pd.notna(last.get("kwh_per_t")) and last["kwh_per_t"] > avg_kwh_t * 1.05:
            alarms.append("⚡ kWh/t son 10 ortalamasına göre üst bantta")
        if avg_tap_temp and pd.notna(last.get("tap_temp_c")) and abs(last["tap_temp_c"] - avg_tap_temp) > 15:
            alarms.append("🔥 Tap sıcaklığı sapması > 15°C")
        if pd.notna(last.get("panel_delta_t_c")) and last["panel_delta_t_c"] > 25:
            alarms.append("💧 Panel ΔT üst bantta (>25°C)")
        if last.get("slag_foaming_index") is not None and float(last["slag_foaming_index"]) >= 9:
            alarms.append("🌋 Slag foaming üst bantta (≥9)")

        if alarms:
            for a in alarms:
                st.warning(a)
        else:
            st.success("✅ Belirgin üst bant sinyali yok (demo)")

    with model_col:
        st.markdown("#### 🤖 AI Model / Eğitim Modu")
        train_mode = st.radio(
            "Eğitim Modu",
            ["Model Eğit", "Sürekli Eğit", "Dijital İkiz Modu"],
            index=0,
            key="train_mode_arc",
        )

        current_rows = len(df)
        progress_ratio = min(current_rows / DIGITAL_TWIN_TARGET_HEATS, 1.0) if DIGITAL_TWIN_TARGET_HEATS else 0.0

        st.caption(f"Veri ilerleme: **{current_rows} / {DIGITAL_TWIN_TARGET_HEATS}** şarj")
        st.progress(progress_ratio)

        if train_mode == "Model Eğit":
            if st.button("Modeli Eğit", key="btn_train_manual"):
                train_arc_model(df, note="(Model Eğit)", min_samples=20)

        elif train_mode == "Sürekli Eğit":
            train_arc_model(df, note="(Sürekli Eğit)", min_samples=20)

        elif train_mode == "Dijital İkiz Modu":
            if current_rows < DIGITAL_TWIN_MIN_START:
                st.warning(f"Dijital ikiz için ≥ {DIGITAL_TWIN_MIN_START} şarj gerekli; şu an {current_rows}.")
            else:
                if current_rows > int(st.session_state.model_last_trained_rows_marker):
                    train_arc_model(df, note="(Dijital İkiz Modu)", min_samples=DIGITAL_TWIN_MIN_START)

        st.write(f"**Durum:** {st.session_state.model_status}")
        if st.session_state.model_last_train_time:
            st.caption(
                f"Son eğitim: {st.session_state.model_last_train_time} · "
                f"Veri: {st.session_state.model_last_train_rows} · "
                f"Toplam eğitim: {st.session_state.model_train_count}"
            )
        else:
            st.caption("Model henüz hiç eğitilmedi.")


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
            st.caption(f"Sim ilerleme: {int(st.session_state.sim_stream_progress)} / {SIM_STREAM_TOTAL}")
        else:
            st.session_state.sim_data = None

        st.markdown("---")
        page = st.radio(
            "Sayfa Seç",
            [
                "Executive Dashboard",
                "Operations Dashboard",
                "Batch Insights",
                "Lab (Simulation)",
                "1. Setup",
                "2. Canlı Veri",
                "3. Arc Optimizer (PoC)",
            ],
            index=0,
        )

    if page == "1. Setup":
        show_setup_form()
    elif page == "2. Canlı Veri":
        show_runtime_page(sim_mode)
    elif page == "3. Arc Optimizer (PoC)":
        show_arc_optimizer_page(sim_mode)
    elif page == "Executive Dashboard":
        show_executive_dashboard(sim_mode)
    elif page == "Operations Dashboard":
        show_operations_dashboard(sim_mode)
    elif page == "Batch Insights":
        show_batch_insights(sim_mode)
    else:
        show_lab_simulation(sim_mode)


if __name__ == "__main__":
    main()
