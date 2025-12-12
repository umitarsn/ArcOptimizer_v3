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

TZ = ZoneInfo("Europe/Istanbul")

SETUP_SAVE_PATH = "data/saved_inputs.json"
RUNTIME_SAVE_PATH = "data/runtime_data.json"
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

MODEL_SAVE_PATH = "models/arc_optimizer_model.pkl"

# Dijital ikiz hedefi artık 10.000 şarj
DIGITAL_TWIN_TARGET_HEATS = 10000     # hedef toplam
DIGITAL_TWIN_MIN_START = 1000         # DT öğrenmeye başlama eşiği (historical)

# Akış varsayımları (demo)
SIM_TOTAL_HEATS = 10000
SIM_HISTORICAL_HEATS = 1000           # ilk 1000: historical
MINUTES_PER_HEAT = 60                # 1000 heat ~ 41.7 gün (≈42 gün)

# ----------------------------------------------
# GLOBAL SESSION STATE
# ----------------------------------------------
if "info_state" not in st.session_state:
    st.session_state.info_state = {}

if "profit_info_state" not in st.session_state:
    st.session_state.profit_info_state = {}

# Simülasyon verisi (full) ve görünür kısım
if "sim_full_data" not in st.session_state:
    st.session_state.sim_full_data = None

if "sim_visible_n" not in st.session_state:
    st.session_state.sim_visible_n = SIM_HISTORICAL_HEATS  # default: historical görünsün

if "sim_mode_flag" not in st.session_state:
    st.session_state.sim_mode_flag = None

# Model eğitim durumu
if "model_status" not in st.session_state:
    st.session_state.model_status = "Henüz eğitilmedi."
    st.session_state.model_last_train_time = None
    st.session_state.model_last_train_rows = 0
    st.session_state.model_train_count = 0
    st.session_state.model_last_trained_n = 0  # dijital ikizde gereksiz sık eğitimi engellemek için

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
def generate_full_simulation_data(total: int = SIM_TOTAL_HEATS):
    """
    10.000 şarjlık simülasyon verisi üretir.
    - Zaman serisi gerçekçi: her heat ~ 60 dk aralıklı
    - Deterministik: seed sabit -> demo stabil
    """
    rng = random.Random(42)
    now = datetime.now(TZ)

    # en eski zaman: total heat önce
    start_ts = now - timedelta(minutes=MINUTES_PER_HEAT * (total - 1))

    sim_list = []
    for i in range(total):
        ts = start_ts + timedelta(minutes=MINUTES_PER_HEAT * i)
        heat_id = f"SIM-{i+1:05d}"

        # Basit ama stabil fiziksel tutarlılık:
        tap_weight = 35 + rng.uniform(-3, 3)                     # ton
        duration_min = 55 + rng.uniform(-10, 10)                 # dk
        kwh_per_t = 420 + rng.uniform(-25, 25)                   # kWh/t
        energy_kwh = max(0.0, tap_weight * kwh_per_t)            # kWh (tipik 12k–18k)

        tap_temp = 1610 + rng.uniform(-15, 15)                   # °C
        o2_flow = 950 + rng.uniform(-150, 150)                   # Nm³/h
        slag_foaming = rng.randint(3, 9)                         # 0–10
        panel_delta_t = 18 + rng.uniform(-5, 8)                  # °C
        electrode_cons = 1.9 + rng.uniform(-0.3, 0.3)            # kg/şarj

        sim_list.append(
            {
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
        )
    return sim_list


def get_sim_visible_data():
    """Full sim datasından görünür kısmı döndürür."""
    if not st.session_state.sim_full_data:
        return []
    n = int(st.session_state.sim_visible_n)
    n = max(SIM_HISTORICAL_HEATS, min(SIM_TOTAL_HEATS, n))
    return st.session_state.sim_full_data[:n]

# ----------------------------------------------
# MODEL FONKSİYONLARI (VERİYE DOKUNMADAN)
# ----------------------------------------------
def get_arc_training_data(df: pd.DataFrame):
    """
    Arc Optimizer için eğitim datasını hazırlar.
    Multi-output: [kwh_per_t, tap_temp_c]
    """
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
    """
    Arc Optimizer modeli (RandomForestRegressor) eğitilir.
    Durum, session_state'e yazılır.
    """
    st.session_state.model_status = "Eğitiliyor..."

    X, y, feature_cols, target_cols = get_arc_training_data(df)
    if X is None:
        st.session_state.model_status = "Eğitim için uygun veri bulunamadı."
        st.error("Model eğitimi için gerekli kolonlar yok veya yeterli dolu kayıt yok.")
        return False

    if len(X) < min_samples:
        st.session_state.model_status = (
            f"Eğitim için veri yetersiz: {len(X)} şarj (gereken ≥ {min_samples})."
        )
        st.warning(f"Bu mod için en az {min_samples} şarj gerekli, şu anda {len(X)} kayıt var.")
        return False

    model = RandomForestRegressor(
        n_estimators=250,
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
        st.info(
            "🧪 **Simülasyon Modu Aktif.** Aşağıdaki veriler gerçek zamanlı yerine "
            "simülasyon amaçlı oluşturulmuştur."
        )
    else:
        st.markdown(
            "Bu sayfada fırın işletmesi sırasında her **şarj / heat** için toplanan "
            "operasyonel veriler girilir veya otomasyondan okunur."
        )

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
            # 5000 limiti simülasyonda yetmiyordu; sadece input limitini genişlettik
            energy_kwh = st.number_input("Toplam Enerji (kWh)", min_value=0.0, step=50.0, max_value=50000.0)
        with c5:
            tap_temp = st.number_input(
                "Tap Sıcaklığı (°C)", min_value=0.0, max_value=2000.0, step=1.0
            )
        with c6:
            o2_flow = st.number_input("Ortalama O2 Debisi (Nm³/h)", min_value=0.0, step=1.0)

        c7, c8, c9 = st.columns(3)
        with c7:
            slag_foaming = st.slider("Slag Foaming Seviyesi (0–10)", 0, 10, 5)
        with c8:
            panel_delta_t = st.number_input("Panel ΔT (°C)", min_value=0.0, step=0.1)
        with c9:
            electrode_cons = st.number_input(
                "Elektrot Tüketimi (kg/şarj)", min_value=0.0, step=0.01
            )

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

    if sim_mode:
        data_source = get_sim_visible_data()
    else:
        data_source = runtime_data

    if not data_source:
        st.info("Henüz canlı veri girilmedi.")
        return

    df = pd.DataFrame(data_source)
    try:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp_dt")
    except Exception:
        df["timestamp_dt"] = df["timestamp"]

    st.markdown("### Kayıtlı Canlı Veriler")
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
# 3) ARC OPTIMIZER
# ----------------------------------------------
def show_arc_optimizer_page(sim_mode: bool):
    st.markdown("## 3. Arc Optimizer – Trendler, KPI ve Öneriler")
    if sim_mode:
        st.info("🧪 **Simülasyon Modu Aktif.** Arc Optimizer çıktıları simüle edilen veri üzerinden hesaplanır.")

    if sim_mode:
        data_source = get_sim_visible_data()
    else:
        data_source = runtime_data

    if not data_source:
        st.info("Önce 2. sayfadan canlı veri ekleyin.")
        return

    df = pd.DataFrame(data_source)
    try:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp_dt")
    except Exception:
        df["timestamp_dt"] = df["timestamp"]

    last = df.iloc[-1]
    last_n = df.tail(10)

    avg_kwh_t = last_n["kwh_per_t"].dropna().mean()
    avg_electrode = last_n["electrode_kg_per_heat"].dropna().mean()
    avg_tap_temp = last_n["tap_temp_c"].dropna().mean()

    if len(df) >= 10 and df["kwh_per_t"].notna().sum() >= 10:
        first5 = df["kwh_per_t"].dropna().head(5).mean()
        last5 = df["kwh_per_t"].dropna().tail(5).mean()
        saving_potential = max(0.0, first5 - last5)
    else:
        saving_potential = 0.0

    kpi_col, model_col = st.columns([3, 2])

    with kpi_col:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Son Şarj kWh/t", f"{last['kwh_per_t']:.1f}" if pd.notna(last.get("kwh_per_t", None)) else "-")
        c2.metric(
            "Son Şarj Elektrot",
            f"{last['electrode_kg_per_heat']:.2f} kg/şarj" if pd.notna(last.get("electrode_kg_per_heat", None)) else "-",
        )
        c3.metric("Son Tap Sıcaklığı", f"{last['tap_temp_c']:.0f} °C" if pd.notna(last.get("tap_temp_c", None)) else "-")
        c4.metric("Son 10 Şarj Ort. kWh/t", f"{avg_kwh_t:.1f}" if avg_kwh_t and not pd.isna(avg_kwh_t) else "-")

    with model_col:
        st.markdown("#### 🤖 AI Model / Eğitim Modu")
        train_mode = st.radio("Eğitim Modu", ["Model Eğit", "Sürekli Eğit", "Dijital İkiz Modu"], index=0, key="train_mode_arc")

        current_rows = len(df)

        if train_mode == "Model Eğit":
            st.caption("Bu buton, mevcut veri setiyle modeli bir kez eğitir (demo / PoC).")
            st.caption(f"Mevcut veri sayısı: {current_rows} şarj (önerilen ≥ 20).")
            if st.button("Modeli Eğit", key="btn_train_manual"):
                ok = train_arc_model(df, note="(Model Eğit)", min_samples=20)
                if ok:
                    st.success(f"Model {st.session_state.model_last_train_rows} şarj verisiyle {st.session_state.model_last_train_time} tarihinde eğitildi.")

        elif train_mode == "Sürekli Eğit":
            st.caption("Her sayfa yenilemesinde mevcut veriyle model güncellenir (demo modu).")
            st.caption(f"Mevcut veri sayısı: {current_rows} şarj (önerilen ≥ 20).")
            ok = train_arc_model(df, note="(Sürekli Eğit)", min_samples=20)
            if ok:
                st.success(f"Model {st.session_state.model_last_train_rows} şarj verisiyle {st.session_state.model_last_train_time} tarihinde eğitildi.")

        elif train_mode == "Dijital İkiz Modu":
            st.caption(
                "Dijital ikiz modu: **1000 şarj historical** ile başlar, veri geldikçe **10.000 şarj** hedefe kadar öğrenmeye devam eder."
            )

            # İlerleme: X/10000 + %X + progress bar
            progress_pct = min(100.0, (current_rows / DIGITAL_TWIN_TARGET_HEATS) * 100.0)
            st.caption(f"Veri ilerleme durumu: **{current_rows} / {DIGITAL_TWIN_TARGET_HEATS}** şarj")
            st.progress(progress_pct / 100.0)
            st.caption(f"Eğitim ilerlemesi: **%{progress_pct:.1f}**")

            if current_rows < DIGITAL_TWIN_MIN_START:
                st.warning(
                    f"Dijital ikiz eğitimine başlamak için en az {DIGITAL_TWIN_MIN_START} şarj gerekiyor; "
                    f"şu an {current_rows} şarj var."
                )
            else:
                # Dijital ikiz: gereksiz her rerun'da eğitmesin
                # (akış hızlıyken CPU'yu yakmamak için)
                retrain_batch = 200  # her +200 şarjda bir retrain
                should_retrain = (st.session_state.model_last_trained_n == 0) or (
                    current_rows - st.session_state.model_last_trained_n >= retrain_batch
                )

                if should_retrain:
                    st.session_state.model_status = "Eğitiliyor..."
                    ok = train_arc_model(df, note="(Dijital İkiz Modu)", min_samples=DIGITAL_TWIN_MIN_START)
                    if ok:
                        st.session_state.model_last_trained_n = current_rows
                        st.success(f"Model {st.session_state.model_last_train_rows} şarj verisiyle {st.session_state.model_last_train_time} tarihinde eğitildi.")
                else:
                    # eğitim arada yapılmıyorsa durum yine de “öğreniyor” kalsın
                    pass

                if current_rows < DIGITAL_TWIN_TARGET_HEATS:
                    st.session_state.model_status = (
                        f"Dijital İkiz **öğrenme aşamasında** "
                        f"(%{progress_pct:.1f} — {current_rows}/{DIGITAL_TWIN_TARGET_HEATS} şarj)"
                    )
                else:
                    st.session_state.model_status = (
                        f"Dijital İkiz **hazır** ✅ "
                        f"(%100.0 — {current_rows} şarj ile eğitildi)"
                    )

        st.write(f"**Durum:** {st.session_state.model_status}")
        if st.session_state.model_last_train_time:
            st.caption(
                f"Son eğitim: {st.session_state.model_last_train_time} · "
                f"Veri sayısı: {st.session_state.model_last_train_rows} şarj · "
                f"Toplam eğitim: {st.session_state.model_train_count}"
            )
        else:
            st.caption("Model henüz hiç eğitilmedi.")

        model, feat_cols, target_cols = load_arc_model()

        # Dijital İkiz What-if Simülasyonu
        if (
            train_mode == "Dijital İkiz Modu"
            and model is not None
            and feat_cols is not None
            and current_rows >= DIGITAL_TWIN_MIN_START
        ):
            st.markdown("#### Dijital İkiz – What-if Simülasyonu")
            last_row_for_defaults = df.iloc[-1]

            def num_input(name, col_name, min_v, max_v, step, fmt="%.1f"):
                default = float(last_row_for_defaults.get(col_name, (min_v + max_v) / 2))
                default = max(min_v, min(max_v, default))
                return st.number_input(
                    name,
                    min_value=min_v,
                    max_value=max_v,
                    value=float(default),
                    step=step,
                    format=fmt,
                    key=f"dtwin_{col_name}",
                )

            c1, c2 = st.columns(2)
            with c1:
                tap_weight = num_input("Tap Weight (t)", "tap_weight_t", 20.0, 80.0, 0.5)
                duration = num_input("Süre (dk)", "duration_min", 20.0, 120.0, 1.0, "%.0f")
                # enerji aralığını genişlettik (sim datası 12k–18k tipik)
                energy = num_input("Enerji (kWh)", "energy_kwh", 1000.0, 50000.0, 50.0)
                o2_flow = num_input("O2 Debisi (Nm³/h)", "o2_flow_nm3h", 200.0, 3000.0, 10.0)
            with c2:
                slag = num_input("Slag Foaming (0–10)", "slag_foaming_index", 0.0, 10.0, 0.5)
                panel_dT = num_input("Panel ΔT (°C)", "panel_delta_t_c", 0.0, 60.0, 0.5)
                elec = num_input("Elektrot (kg/şarj)", "electrode_kg_per_heat", 0.5, 6.0, 0.05)

            if st.button("Simülasyonu Çalıştır", key="btn_dtwin_sim"):
                inp = {
                    "tap_weight_t": tap_weight,
                    "duration_min": duration,
                    "energy_kwh": energy,
                    "o2_flow_nm3h": o2_flow,
                    "slag_foaming_index": slag,
                    "panel_delta_t_c": panel_dT,
                    "electrode_kg_per_heat": elec,
                }

                row_df = pd.DataFrame([inp])[feat_cols]
                row_df = row_df.fillna(row_df.mean())

                try:
                    preds = model.predict(row_df)[0]
                    pred_dict = dict(zip(target_cols, preds))
                    kwh_pred = float(pred_dict.get("kwh_per_t", float("nan")))
                    tap_pred = float(pred_dict.get("tap_temp_c", float("nan")))

                    st.markdown("**AI Tahmin (Dijital İkiz):**")
                    st.write(f"- kWh/t ≈ **{kwh_pred:.1f}**")
                    st.write(f"- Tap T ≈ **{tap_pred:.0f} °C**")
                except Exception as e:
                    st.error(f"Tahmin hesaplanırken hata oluştu: {e}")

        elif model is not None and feat_cols is not None:
            missing = [c for c in feat_cols if c not in df.columns]
            if not missing:
                last_features = df.iloc[[-1]][feat_cols].fillna(df[feat_cols].mean())
                try:
                    preds = model.predict(last_features)[0]
                    pred_dict = dict(zip(target_cols, preds))
                    st.markdown("**AI Tahmin (Son Şarj için):**")
                    st.caption(
                        f"kWh/t ≈ {pred_dict.get('kwh_per_t', float('nan')):.1f}, "
                        f"Tap T ≈ {pred_dict.get('tap_temp_c', float('nan')):.0f} °C"
                    )
                except Exception:
                    pass

    # ---- ZAMAN TRENDi + TAHMİN ----
    trend_df = df.set_index("timestamp_dt")[["kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"]]
    min_time = df["timestamp_dt"].min()
    last_time = df["timestamp_dt"].max()
    real_span = last_time - min_time
    if real_span.total_seconds() <= 0:
        real_span = timedelta(hours=6)

    future_span = real_span * 0.20

    def _safe_base(val_avg, val_last, default):
        if val_avg is not None and not pd.isna(val_avg):
            return val_avg
        if val_last is not None and not pd.isna(val_last):
            return val_last
        return default

    base_tap_temp = _safe_base(avg_tap_temp, last.get("tap_temp_c"), 1600.0)
    base_kwh_t = _safe_base(avg_kwh_t, last.get("kwh_per_t"), 420.0)
    base_electrode = _safe_base(avg_electrode, last.get("electrode_kg_per_heat"), 2.0)

    predicted_tap_temp_target = base_tap_temp + 5.0
    predicted_kwh_t_target = base_kwh_t - 5.0
    predicted_electrode_target = base_electrode

    future_points = []
    last_kwh = last.get("kwh_per_t", base_kwh_t)
    last_tap_temp = last.get("tap_temp_c", base_tap_temp)
    last_electrode = last.get("electrode_kg_per_heat", base_electrode)

    for i in range(4):
        frac = i / 3.0
        t = last_time + future_span * frac
        kwh_val = last_kwh + (predicted_kwh_t_target - last_kwh) * frac
        tap_val = last_tap_temp + (predicted_tap_temp_target - last_tap_temp) * frac
        el_val = last_electrode + (predicted_electrode_target - last_electrode) * frac
        future_points.append({"timestamp_dt": t, "kwh_per_t": kwh_val, "tap_temp_c": tap_val, "electrode_kg_per_heat": el_val})

    future_df = pd.DataFrame(future_points)
    predicted_tap_time = future_points[-1]["timestamp_dt"]

    actual_long = trend_df.reset_index().melt(
        id_vars=["timestamp_dt"],
        value_vars=["kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"],
        var_name="variable",
        value_name="value",
    )
    actual_long["data_type"] = "Aktüel"

    future_long = future_df.melt(
        id_vars=["timestamp_dt"],
        value_vars=["kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"],
        var_name="variable",
        value_name="value",
    )
    future_long["data_type"] = "Potansiyel (AI)"

    combined = pd.concat([actual_long, future_long], ignore_index=True)

    var_map = {"kwh_per_t": "kWh/t", "tap_temp_c": "Tap T (°C)", "electrode_kg_per_heat": "Elektrot (kg/şarj)"}
    combined["variable_name"] = combined["variable"].map(var_map)

    domain_min = min_time
    if predicted_tap_time > domain_min:
        domain_max = domain_min + (predicted_tap_time - domain_min) / 0.9
    else:
        domain_max = domain_min + timedelta(hours=6)

    st.markdown("### Proses Gidişatı – Zaman Trendi ve Tahmini Döküm Anı (AI)")

    base_chart = (
        alt.Chart(combined)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "timestamp_dt:T",
                title="Zaman",
                scale=alt.Scale(domain=[domain_min, domain_max]),
                axis=alt.Axis(format="%H:%M", labelFontSize=12, titleFontSize=14),
            ),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
            color=alt.Color(
                "variable_name:N",
                title="Değişken",
                legend=alt.Legend(orient="top", direction="horizontal", labelFontSize=11, titleFontSize=12),
            ),
            strokeDash=alt.StrokeDash(
                "data_type:N",
                title="Veri Tipi",
                scale=alt.Scale(domain=["Aktüel", "Potansiyel (AI)"], range=[[1, 0], [6, 4]]),
            ),
        )
        .properties(height=420, width="container")
    )

    tap_point_df = future_df[future_df["timestamp_dt"] == predicted_tap_time][["timestamp_dt", "tap_temp_c"]].copy()
    tap_point_df.rename(columns={"tap_temp_c": "value"}, inplace=True)
    tap_point_df["variable_name"] = "Tap T (°C)"

    point_chart = alt.Chart(tap_point_df).mark_point(size=120, filled=True).encode(
        x="timestamp_dt:T", y="value:Q", color=alt.Color("variable_name:N", legend=None)
    )

    label_df = tap_point_df.copy()
    label_df["label_top"] = label_df["timestamp_dt"].dt.strftime("Hedef Döküm Zamanı (AI): %Y-%m-%d %H:%M")
    label_df["label_bottom"] = label_df["value"].map(lambda v: f"Sıcaklık: {v:.0f} °C")

    label_top_chart = alt.Chart(label_df).mark_text(align="left", dx=35, dy=-35, fontSize=12, fontWeight="bold").encode(
        x="timestamp_dt:T", y="value:Q", text="label_top:N"
    )
    label_bottom_chart = alt.Chart(label_df).mark_text(align="left", dx=35, dy=-10, fontSize=11).encode(
        x="timestamp_dt:T", y="value:Q", text="label_bottom:N"
    )

    now_df = pd.DataFrame({"timestamp_dt": [last_time]})
    now_rule = alt.Chart(now_df).mark_rule(strokeDash=[2, 2]).encode(x="timestamp_dt:T")

    full_chart = (base_chart + point_chart + now_rule + label_top_chart + label_bottom_chart).properties(
        padding={"right": 20, "left": 10, "top": 40, "bottom": 20}
    )

    st.altair_chart(full_chart.interactive(), use_container_width=True)

    delta_min = (predicted_tap_time - last_time).total_seconds() / 60.0
    st.markdown(
        f"**Tahmini Döküm Anı (AI):** {predicted_tap_time.strftime('%Y-%m-%d %H:%M')} (yaklaşık {delta_min:.0f} dk sonra)"
    )

    # ------------------------------------------
    # PROSES KAZANÇ ANALİZİ (TON BAŞINA)
    # ------------------------------------------
    st.markdown("### 💰 Proses Kazanç Analizi (Ton Başına)")

    ENERGY_PRICE_EUR_PER_KWH = 0.12
    ELECTRODE_PRICE_EUR_PER_KG = 3.0

    rows = []
    total_gain_per_t = 0.0

    # Enerji: Potansiyel (AI) asla aktüelden kötü görünmesin (demo algısı)
    if pd.notna(last.get("kwh_per_t", None)) and avg_kwh_t and not pd.isna(avg_kwh_t):
        real = float(last["kwh_per_t"])
        raw_target = max(avg_kwh_t - 5.0, 0.0)
        target = min(real, raw_target)  # <- potansiyel kötüleşmesin
        diff = real - target
        gain = abs(diff) * ENERGY_PRICE_EUR_PER_KWH
        total_gain_per_t += gain
        rows.append(
            {"tag": "kwh_per_t", "deg": "Enerji tüketimi", "akt": f"{real:.1f} kWh/t", "pot": f"{target:.1f} kWh/t",
             "fark": f"{diff:+.1f} kWh/t", "kazanc": f"{gain:.2f} €/t", "type": "cost"}
        )

    # Elektrot (kg/t): potansiyel kötüleşmesin
    if pd.notna(last.get("electrode_kg_per_heat", None)) and pd.notna(last.get("tap_weight_t", None)):
        tap_w = float(last["tap_weight_t"]) or 0.0
        if tap_w > 0:
            real_pt = float(last["electrode_kg_per_heat"]) / tap_w
            raw_target_pt = (float(avg_electrode) / tap_w) if pd.notna(avg_electrode) else max(real_pt - 0.05, 0.0)
            target_pt = min(real_pt, raw_target_pt)
            diff = real_pt - target_pt
            gain = abs(diff) * ELECTRODE_PRICE_EUR_PER_KG
            total_gain_per_t += gain
            rows.append(
                {"tag": "electrode", "deg": "Elektrot tüketimi", "akt": f"{real_pt:.3f} kg/t", "pot": f"{target_pt:.3f} kg/t",
                 "fark": f"{diff:+.3f} kg/t", "kazanc": f"{gain:.2f} €/t", "type": "cost"}
            )

    # Tap sıcaklığı
    if pd.notna(last.get("tap_temp_c", None)) and avg_tap_temp and not pd.isna(avg_tap_temp):
        real = float(last["tap_temp_c"])
        target = float(avg_tap_temp)
        diff = real - target
        tap_gain_range = "0.03–0.10 €/t + Kalite ↑"
        rows.append(
            {"tag": "tap_temp_c", "deg": "Tap sıcaklığı optimizasyonu", "akt": f"{real:.0f} °C", "pot": f"{target:.0f} °C",
             "fark": f"{diff:+.0f} °C", "kazanc": tap_gain_range, "type": "mixed"}
        )

    # Panel ΔT
    if pd.notna(last.get("panel_delta_t_c", None)):
        real = float(last["panel_delta_t_c"])
        target = 20.0
        diff = real - target
        rows.append(
            {"tag": "panel_delta_t", "deg": "Panel ΔT", "akt": f"{real:.1f} °C", "pot": f"{target:.1f} °C",
             "fark": f"{diff:+.1f} °C", "kazanc": "Kalite ↑", "type": "quality"}
        )

    # Slag foaming
    slag_val = None
    if last.get("slag_foaming_index", None) is not None:
        slag_val = float(last["slag_foaming_index"])
        target = 7.0
        diff = slag_val - target
        rows.append(
            {"tag": "slag_foaming", "deg": "Köpük yüksekliği / slag foaming", "akt": f"{slag_val:.1f}", "pot": f"{target:.1f}",
             "fark": f"{diff:+.1f}", "kazanc": "Enerji verimliliği ↑, elektrot ve refrakter tüketimi ↓", "type": "quality"}
        )

    # Refrakter aşınma seviyesi
    if pd.notna(last.get("tap_temp_c", None)) and pd.notna(last.get("panel_delta_t_c", None)):
        t_act = float(last["tap_temp_c"])
        dT_act = float(last["panel_delta_t_c"])

        if (avg_tap_temp is not None and not pd.isna(avg_tap_temp)):
            dt_from_avg = t_act - float(avg_tap_temp)
        else:
            dt_from_avg = 0.0

        if dt_from_avg > 20 or dT_act > 30:
            refr_level = "Yüksek risk"
        elif dt_from_avg > 10 or dT_act > 25:
            refr_level = "Orta"
        else:
            refr_level = "Düşük"

        rows.append(
            {"tag": "refractory_wear", "deg": "Refrakter aşınma seviyesi", "akt": refr_level, "pot": "AI kontrollü optimum bölge",
             "fark": "-", "kazanc": "Refrakter ömrü ↑, planlı duruşlar dışında duruş ↓", "type": "quality"}
        )

    # Karışım kalitesi
    if (
        pd.notna(last.get("kwh_per_t", None)) and avg_kwh_t and not pd.isna(avg_kwh_t)
        and pd.notna(last.get("tap_temp_c", None)) and avg_tap_temp and not pd.isna(avg_tap_temp)
    ):
        score = 0
        if slag_val is not None and slag_val >= 7.0:
            score += 1
        if abs(float(last["kwh_per_t"]) - float(avg_kwh_t)) <= 10:
            score += 1
        if abs(float(last["tap_temp_c"]) - float(avg_tap_temp)) <= 10:
            score += 1

        mix_level = "İyi" if score == 3 else ("Orta" if score == 2 else "Riskli")
        rows.append(
            {"tag": "mix_quality", "deg": "Karışım kalitesi (homojenlik)", "akt": mix_level, "pot": "AI ile stabil ve homojen bölge",
             "fark": "-", "kazanc": "Kalite ↑, iç hurda ve yeniden işleme ↓", "type": "quality"}
        )

    widths = [1.0, 2.0, 1.3, 1.3, 1.1, 1.8, 0.5]
    hcols = st.columns(widths)
    hcols[0].markdown("**Tag**")
    hcols[1].markdown("**Değişken**")
    hcols[2].markdown("**Aktüel**")
    hcols[3].markdown("**Potansiyel (AI)**")
    hcols[4].markdown("**Fark**")
    hcols[5].markdown("**Tahmini Kazanç**")
    hcols[6].markdown("")

    profit_state = st.session_state.profit_info_state
    for row in rows:
        cols = st.columns(widths)
        cols[0].markdown(row["tag"])
        cols[1].markdown(row["deg"])
        cols[2].markdown(row["akt"])
        cols[3].markdown(row["pot"])
        cols[4].markdown(row["fark"])
        cols[5].markdown(row["kazanc"])

        btn_key = f"profit_info_btn_{row['tag']}"
        if cols[6].button("ℹ️", key=btn_key):
            profit_state[row["tag"]] = not profit_state.get(row["tag"], False)

    st.markdown(
        f"**Toplam Potansiyel Kazanç (AI tahmini, ton başına – doğrudan hesaplanabilen kalemler):** "
        f"≈ **{total_gain_per_t:,.1f} €/t**"
    )

    # Basit öneriler
    st.markdown("### Model Önerileri (Örnek / Demo Mantık)")
    suggestions = []

    if pd.notna(last.get("kwh_per_t", None)) and avg_kwh_t and not pd.isna(avg_kwh_t) and last["kwh_per_t"] > avg_kwh_t * 1.05:
        suggestions.append("🔌 Son şarjın **kWh/t** değeri son 10 şarj ortalamasına göre yüksek. Oksijen debisi ve güç profilini gözden geçirin.")

    if pd.notna(last.get("electrode_kg_per_heat", None)) and avg_electrode and not pd.isna(avg_electrode) and last["electrode_kg_per_heat"] > avg_electrode * 1.10:
        suggestions.append("🧯 **Elektrot tüketimi** son şarjda yükselmiş. Ark stabilitesi ve elektrot hareketlerini kontrol edin.")

    if pd.notna(last.get("tap_temp_c", None)) and avg_tap_temp and not pd.isna(avg_tap_temp) and last["tap_temp_c"] < avg_tap_temp - 10:
        suggestions.append("🔥 Tap sıcaklığı son şarjda düşük. Enerji girişini hafif artırmak veya şarj sonu bekleme süresini optimize etmek gerekebilir.")

    if last.get("slag_foaming_index", None) is not None and last["slag_foaming_index"] >= 8:
        suggestions.append("🌋 Slag foaming seviyesi yüksek (≥8). Karbon/O₂ dengesini ve köpük kontrolünü gözden geçirin.")

    if last.get("panel_delta_t_c", None) is not None and last["panel_delta_t_c"] > 25:
        suggestions.append("💧 Panel ΔT yüksek. Soğutma devresinde dengesizlik olabilir; panel debilerini ve tıkalı hatları kontrol edin.")

    if saving_potential > 0.0:
        suggestions.append(f"📉 kWh/t trendine göre yaklaşık **{saving_potential:.1f} kWh/t** iyileştirme potansiyeli görülüyor.")

    if not suggestions:
        suggestions.append("✅ Model açısından belirgin bir anomali/iyileştirme alarmı yok. Mevcut ayarlar stabil görünüyor.")

    for s in suggestions:
        st.markdown(f"- {s}")

# ----------------------------------------------
# MAIN
# ----------------------------------------------
def main():
    with st.sidebar:
        st.markdown("### FeCr AI")

        # Simülasyon modu varsayılan olarak AÇIK
        sim_mode = st.toggle(
            "Simülasyon Modu",
            value=True,
            help="Açıkken sistem canlı veri yerine simüle edilmiş veri kullanır.",
        )

        # Simülasyon akışı kontrolleri (sayfa yapısını bozmaz; sidebar sadece)
        sim_stream = False
        stream_9000 = False
        auto_stream = False
        stream_speed = 25

        if sim_mode:
            st.divider()
            st.markdown("⏳ **Simülasyon Veri Akışı**")

            sim_stream = st.toggle("Simülasyon Veri Akışı", value=True)
            stream_9000 = st.toggle("9000 şarj zamanla oku", value=False)
            stream_speed = st.slider("Akış hızı (şarj/yenileme)", min_value=1, max_value=200, value=25)

            auto_stream = st.toggle("Otomatik akış", value=False, help="Açıkken sayfa kendini yenileyerek veriyi akar.")

        # Full sim datasını bir kez üret (st.autorefresh yok!)
        if sim_mode:
            if st.session_state.sim_full_data is None:
                st.session_state.sim_full_data = generate_full_simulation_data(SIM_TOTAL_HEATS)
                st.session_state.sim_visible_n = SIM_HISTORICAL_HEATS
            # sim_mode_flag sadece kontrol için
            st.session_state.sim_mode_flag = True
        else:
            st.session_state.sim_mode_flag = False

        page = st.radio("Sayfa Seç", ["1. Setup", "2. Canlı Veri", "3. Arc Optimizer"])

    # Akış mantığı: historical 1000 sabit; 9000 zamanla eklensin
    if sim_mode and sim_stream and stream_9000:
        current = int(st.session_state.sim_visible_n)
        if current < SIM_TOTAL_HEATS:
            st.session_state.sim_visible_n = min(SIM_TOTAL_HEATS, current + int(stream_speed))

        # otomatik akış açık ise: kendini rerun ettir
        if auto_stream and st.session_state.sim_visible_n < SIM_TOTAL_HEATS:
            # çok agresif olmasın
            time.sleep(0.6)
            st.rerun()

    if page == "1. Setup":
        show_setup_form()
    elif page == "2. Canlı Veri":
        show_runtime_page(sim_mode)
    else:
        show_arc_optimizer_page(sim_mode)


if __name__ == "__main__":
    main()
