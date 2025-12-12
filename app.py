import os
import json
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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
DIGITAL_TWIN_TARGET_HEATS = 10000
DIGITAL_TWIN_MIN_START = 200   # dijital ikiz öğrenmeye başlamak için min şarj
DIGITAL_TWIN_RETRAIN_EVERY_N = 200  # her N yeni şarjda bir yeniden eğit

# Simülasyonda başlangıçta görünen historical şarj sayısı
SIM_INITIAL_HISTORICAL = 1000

# ----------------------------------------------
# GLOBAL SESSION STATE
# ----------------------------------------------
if "info_state" not in st.session_state:
    st.session_state.info_state = {}

if "profit_info_state" not in st.session_state:
    st.session_state.profit_info_state = {}

if "sim_data" not in st.session_state:
    st.session_state.sim_data = None

if "sim_mode_flag" not in st.session_state:
    st.session_state.sim_mode_flag = None

# sim_full_data: 10k historical dataset (tamamı geçmişte)
if "sim_full_data" not in st.session_state:
    st.session_state.sim_full_data = None

# sim_loaded_count: şu an UI'da görünen heat sayısı (1000'den 10k'ya akar)
if "sim_loaded_count" not in st.session_state:
    st.session_state.sim_loaded_count = 0

# Model eğitim durumu
if "model_status" not in st.session_state:
    st.session_state.model_status = "Henüz eğitilmedi."
    st.session_state.model_last_train_time = None
    st.session_state.model_last_train_rows = 0
    st.session_state.model_train_count = 0
    st.session_state.model_last_retrain_trigger_rows = 0  # son retrain hangi row sayısında tetiklendi

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
# SİMÜLASYON (10.000 historical + zamanla okuma)
# ----------------------------------------------
def _make_one_heat(ts: datetime, heat_no: int):
    heat_id = f"SIM-{heat_no}"

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
        "tap_weight_t": tap_weight,
        "duration_min": duration_min,
        "energy_kwh": energy_kwh,
        "tap_temp_c": tap_temp,
        "o2_flow_nm3h": o2_flow,
        "slag_foaming_index": slag_foaming,
        "panel_delta_t_c": panel_delta_t,
        "electrode_kg_per_heat": electrode_cons,
        "kwh_per_t": kwh_per_t,
        "operator_note": "Historical simulation record",
    }


def generate_simulation_full_history(total_n: int = 10000):
    """
    Tamamı geçmişte olan 10.000 heat üretir:
    now - total_n saat ... now aralığı (1 saat aralıklı).
    """
    now = datetime.now(TZ).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=total_n)

    full = []
    for i in range(total_n):
        ts = start + timedelta(hours=i + 1)  # 1..total_n
        full.append(_make_one_heat(ts, i + 1))
    return full


def ensure_simulation_initialized():
    """
    - sim_full_data yoksa 10.000 historical oluşturur
    - sim_loaded_count 0 ise 1000 ile başlatır
    - sim_data = sim_full_data[:sim_loaded_count]
    """
    if st.session_state.sim_full_data is None:
        st.session_state.sim_full_data = generate_simulation_full_history(DIGITAL_TWIN_TARGET_HEATS)

    if st.session_state.sim_loaded_count <= 0:
        st.session_state.sim_loaded_count = min(SIM_INITIAL_HISTORICAL, DIGITAL_TWIN_TARGET_HEATS)

    st.session_state.sim_data = st.session_state.sim_full_data[: st.session_state.sim_loaded_count]


def simulate_stream_step(step_n: int):
    """
    Zamanla okuma: sim_loaded_count'u step_n artırır (max 10k).
    """
    if st.session_state.sim_full_data is None:
        ensure_simulation_initialized()

    current = st.session_state.sim_loaded_count
    target = DIGITAL_TWIN_TARGET_HEATS
    new_count = min(target, current + max(1, int(step_n)))
    st.session_state.sim_loaded_count = new_count
    st.session_state.sim_data = st.session_state.sim_full_data[:new_count]


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
    return X, y, feature_cols, target_cols


def train_arc_model(df: pd.DataFrame, note: str = "", min_samples: int = 20):
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
        return False

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "target_cols": target_cols,
        },
        MODEL_SAVE_PATH,
    )

    now_str = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    rows = len(X)

    st.session_state.model_status = f"Eğitildi ✅ {note}".strip()
    st.session_state.model_last_train_time = now_str
    st.session_state.model_last_train_rows = rows
    st.session_state.model_train_count += 1
    st.session_state.model_last_retrain_trigger_rows = rows

    return True


def load_arc_model():
    if not os.path.exists(MODEL_SAVE_PATH):
        return None, None, None
    try:
        data = joblib.load(MODEL_SAVE_PATH)
        return data.get("model"), data.get("feature_cols"), data.get("target_cols")
    except Exception:
        return None, None, None


def maybe_retrain_digital_twin(df: pd.DataFrame):
    """
    Dijital ikizde: veri arttıkça her DIGITAL_TWIN_RETRAIN_EVERY_N şarjda bir yeniden eğit.
    """
    current_rows = len(df)
    last_trained_rows = int(st.session_state.model_last_retrain_trigger_rows or 0)

    if current_rows < DIGITAL_TWIN_MIN_START:
        st.session_state.model_status = (
            f"Dijital ikiz için veri yetersiz: {current_rows}/{DIGITAL_TWIN_MIN_START}"
        )
        return False, False

    # İlk defa veya N kadar artış olduysa
    should_train = (last_trained_rows == 0) or (current_rows - last_trained_rows >= DIGITAL_TWIN_RETRAIN_EVERY_N)

    trained = False
    if should_train:
        trained = train_arc_model(df, note="(Dijital İkiz Modu)", min_samples=DIGITAL_TWIN_MIN_START)

    # Durum mesajı
    if current_rows < DIGITAL_TWIN_TARGET_HEATS:
        st.session_state.model_status = (
            f"Dijital İkiz **öğreniyor** ⏳ ({current_rows}/{DIGITAL_TWIN_TARGET_HEATS} şarj)"
        )
    else:
        st.session_state.model_status = (
            f"Dijital İkiz **hazır** ✅ ({current_rows} şarj ile eğitiliyor)"
        )

    return should_train, trained


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
                        st.session_state.info_state[row_key] = not st.session_state.info_state.get(
                            row_key, False
                        )

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
            energy_kwh = st.number_input("Toplam Enerji (kWh)", min_value=0.0, step=10.0)
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

    # veri kaynağı
    data_source = st.session_state.sim_data if sim_mode else runtime_data

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
def show_arc_optimizer_page(sim_mode: bool, stream_on: bool, stream_step: int):
    st.markdown("## 3. Arc Optimizer – Trendler, KPI ve Öneriler")
    if sim_mode:
        st.info("🧪 **Simülasyon Modu Aktif.** Çıktılar simüle edilen veri üzerinden hesaplanır.")

    data_source = st.session_state.sim_data if sim_mode else runtime_data
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
        c1.metric("Son Şarj kWh/t", f"{last['kwh_per_t']:.1f}" if pd.notna(last.get("kwh_per_t")) else "-")
        c2.metric("Son Şarj Elektrot", f"{last['electrode_kg_per_heat']:.2f} kg/şarj" if pd.notna(last.get("electrode_kg_per_heat")) else "-")
        c3.metric("Son Tap Sıcaklığı", f"{last['tap_temp_c']:.0f} °C" if pd.notna(last.get("tap_temp_c")) else "-")
        c4.metric("Son 10 Şarj Ort. kWh/t", f"{avg_kwh_t:.1f}" if avg_kwh_t and not pd.isna(avg_kwh_t) else "-")

    with model_col:
        st.markdown("#### 🤖 AI Model / Eğitim Modu")

        # İlerleme görünümü (10k hedef)
        current_rows = len(df)
        st.caption(f"Veri ilerleme durumu: **{current_rows} / {DIGITAL_TWIN_TARGET_HEATS}** şarj")
        st.progress(min(current_rows / DIGITAL_TWIN_TARGET_HEATS, 1.0))

        # streaming bilgisi
        if sim_mode:
            if stream_on and current_rows < DIGITAL_TWIN_TARGET_HEATS:
                st.caption(f"Veri akışı: **aktif** · Her yenilemede +{stream_step} şarj")
            elif stream_on and current_rows >= DIGITAL_TWIN_TARGET_HEATS:
                st.caption("Veri akışı: **tamamlandı** (10.000 şarja ulaşıldı)")
            else:
                st.caption("Veri akışı: **kapalı**")

        train_mode = st.radio(
            "Eğitim Modu",
            ["Model Eğit", "Sürekli Eğit", "Dijital İkiz Modu"],
            index=2,  # dijital ikiz default seçili dursun (demo mantığı)
            key="train_mode_arc",
        )

        if train_mode == "Model Eğit":
            st.caption("Mevcut veriyle modeli 1 kez eğitir.")
            if st.button("Modeli Eğit", key="btn_train_manual"):
                ok = train_arc_model(df, note="(Model Eğit)", min_samples=20)
                if ok:
                    st.success(f"Model {len(df)} şarj ile eğitildi.")

        elif train_mode == "Sürekli Eğit":
            st.caption("Her sayfa yenilemesinde modeli yeniden eğitir (demo).")
            ok = train_arc_model(df, note="(Sürekli Eğit)", min_samples=20)
            if ok:
                st.success(f"Model {len(df)} şarj ile eğitildi.")

        elif train_mode == "Dijital İkiz Modu":
            st.caption(
                "Dijital ikiz: veri geldikçe **öğrenmeye devam eder**. "
                f"Model her **{DIGITAL_TWIN_RETRAIN_EVERY_N} yeni şarjda** bir yeniden eğitilir."
            )

            if current_rows < DIGITAL_TWIN_MIN_START:
                st.warning(
                    f"Dijital ikiz için en az {DIGITAL_TWIN_MIN_START} şarj gerekir; "
                    f"şu an {current_rows} şarj var."
                )
                st.session_state.model_status = "Eğitim beklemede (veri yetersiz)."
            else:
                should_train, trained = maybe_retrain_digital_twin(df)
                if should_train and trained:
                    st.success(f"Model güncellendi: {current_rows} şarj (retrain).")
                elif should_train and not trained:
                    st.warning("Model retrain denendi ama başarısız oldu.")
                else:
                    st.info("Yeni veri birikiyor… (retrain eşiği gelince güncellenecek)")

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

        # ---- Dijital İkiz What-if ----
        if train_mode == "Dijital İkiz Modu" and model is not None and feat_cols is not None and current_rows >= DIGITAL_TWIN_MIN_START:
            st.markdown("#### Dijital İkiz – What-if Simülasyonu")
            last_row_for_defaults = df.iloc[-1]

            def num_input(name, col_name, min_v, max_v, step, fmt="%.1f"):
                default = float(last_row_for_defaults.get(col_name, (min_v + max_v) / 2))
                # default min/max dışında kalırsa kırp (Streamlit patlamasın)
                default = max(min_v, min(max_v, default))
                return st.number_input(
                    name,
                    min_value=float(min_v),
                    max_value=float(max_v),
                    value=float(default),
                    step=float(step),
                    format=fmt,
                    key=f"dtwin_{col_name}",
                )

            c1, c2 = st.columns(2)
            with c1:
                tap_weight = num_input("Tap Weight (t)", "tap_weight_t", 20.0, 60.0, 0.5)
                duration = num_input("Süre (dk)", "duration_min", 30.0, 120.0, 1.0, "%.0f")

                # ✅ Enerji sınırlarını dinamik yap (hata fix)
                _last_energy = float(last_row_for_defaults.get("energy_kwh", 15000.0))
                energy_max = max(5000.0, _last_energy * 1.6)
                energy_min = max(0.0, _last_energy * 0.4)
                energy = num_input("Enerji (kWh)", "energy_kwh", energy_min, energy_max, 50.0)

                o2_flow = num_input("O2 Debisi (Nm³/h)", "o2_flow_nm3h", 300.0, 2500.0, 10.0)
            with c2:
                slag = num_input("Slag Foaming (0–10)", "slag_foaming_index", 0.0, 10.0, 0.5)
                panel_dT = num_input("Panel ΔT (°C)", "panel_delta_t_c", 3.0, 60.0, 0.5)
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

    # ---- Trend grafiği (aynı mantık) ----
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
        future_points.append(
            {"timestamp_dt": t, "kwh_per_t": kwh_val, "tap_temp_c": tap_val, "electrode_kg_per_heat": el_val}
        )

    future_df = pd.DataFrame(future_points)
    predicted_tap_time = future_points[-1]["timestamp_dt"]

    actual_long = (
        trend_df.reset_index()
        .melt(
            id_vars=["timestamp_dt"],
            value_vars=["kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"],
            var_name="variable",
            value_name="value",
        )
    )
    actual_long["data_type"] = "Aktüel"

    future_long = (
        future_df.melt(
            id_vars=["timestamp_dt"],
            value_vars=["kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"],
            var_name="variable",
            value_name="value",
        )
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

    st.markdown("### Proses Gidişatı – Zaman Trendi ve Tahmini Dökümküm Anı (AI)")

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

    point_chart = (
        alt.Chart(tap_point_df)
        .mark_point(size=120, filled=True)
        .encode(x="timestamp_dt:T", y="value:Q", color=alt.Color("variable_name:N", legend=None))
    )

    label_df = tap_point_df.copy()
    label_df["label_top"] = label_df["timestamp_dt"].dt.strftime("Hedef Döküm Zamanı (AI): %Y-%m-%d %H:%M")
    label_df["label_bottom"] = label_df["value"].map(lambda v: f"Sıcaklık: {v:.0f} °C")

    label_top_chart = (
        alt.Chart(label_df)
        .mark_text(align="left", dx=35, dy=-35, fontSize=12, fontWeight="bold")
        .encode(x="timestamp_dt:T", y="value:Q", text="label_top:N")
    )
    label_bottom_chart = (
        alt.Chart(label_df)
        .mark_text(align="left", dx=35, dy=-10, fontSize=11)
        .encode(x="timestamp_dt:T", y="value:Q", text="label_bottom:N")
    )

    now_df = pd.DataFrame({"timestamp_dt": [last_time]})
    now_rule = alt.Chart(now_df).mark_rule(strokeDash=[2, 2]).encode(x="timestamp_dt:T")

    full_chart = (base_chart + point_chart + now_rule + label_top_chart + label_bottom_chart).properties(
        padding={"right": 20, "left": 10, "top": 40, "bottom": 20}
    )

    st.altair_chart(full_chart.interactive(), use_container_width=True)

    delta_min = (predicted_tap_time - last_time).total_seconds() / 60.0
    st.markdown(
        f"**Tahmini Döküm Anı (AI):** {predicted_tap_time.strftime('%Y-%m-%d %H:%M')} "
        f"(yaklaşık {delta_min:.0f} dk sonra)"
    )

    # Basit öneriler (aynı)
    st.markdown("### Model Önerileri (Örnek / Demo Mantık)")
    suggestions = []

    if (
        pd.notna(last.get("kwh_per_t"))
        and avg_kwh_t
        and not pd.isna(avg_kwh_t)
        and last["kwh_per_t"] > avg_kwh_t * 1.05
    ):
        suggestions.append("🔌 Son şarjın **kWh/t** değeri son 10 şarj ortalamasına göre yüksek.")

    if (
        pd.notna(last.get("electrode_kg_per_heat"))
        and avg_electrode
        and not pd.isna(avg_electrode)
        and last["electrode_kg_per_heat"] > avg_electrode * 1.10
    ):
        suggestions.append("🧯 **Elektrot tüketimi** son şarjda yükselmiş.")

    if (
        pd.notna(last.get("tap_temp_c"))
        and avg_tap_temp
        and not pd.isna(avg_tap_temp)
        and last["tap_temp_c"] < avg_tap_temp - 10
    ):
        suggestions.append("🔥 Tap sıcaklığı son şarjda düşük.")

    if last.get("slag_foaming_index") is not None and float(last["slag_foaming_index"]) >= 8:
        suggestions.append("🌋 Slag foaming seviyesi yüksek (≥8).")

    if last.get("panel_delta_t_c") is not None and float(last["panel_delta_t_c"]) > 25:
        suggestions.append("💧 Panel ΔT yüksek.")

    if saving_potential > 0.0:
        suggestions.append(f"📉 kWh/t trendine göre yaklaşık **{saving_potential:.1f} kWh/t** iyileştirme potansiyeli.")

    if not suggestions:
        suggestions.append("✅ Belirgin bir anomali alarmı yok. Stabil görünüyor.")

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

        stream_on = False
        stream_step = 25

        if sim_mode:
            # sim init
            ensure_simulation_initialized()

            # Zamanla okuma (9000 heat akışı)
            st.markdown("#### ⏳ Simülasyon Veri Akışı")
            stream_on = st.toggle("9000 şarjı zamanla oku", value=True)
            stream_step = st.slider("Akış hızı (şarj/yenileme)", 1, 200, 25)

            # Otomatik yenileme: akış açıkken ve hedefe ulaşılmadıysa
            if stream_on and st.session_state.sim_loaded_count < DIGITAL_TWIN_TARGET_HEATS:
                # 2 saniyede bir yenile
                st.autorefresh(interval=2000, key="sim_stream_refresh")
                simulate_stream_step(stream_step)

        else:
            # gerçek mod
            st.session_state.sim_mode_flag = False
            st.session_state.sim_data = None

        page = st.radio("Sayfa Seç", ["1. Setup", "2. Canlı Veri", "3. Arc Optimizer"])

    if page == "1. Setup":
        show_setup_form()
    elif page == "2. Canlı Veri":
        show_runtime_page(sim_mode)
    else:
        show_arc_optimizer_page(sim_mode, stream_on, stream_step)


if __name__ == "__main__":
    main()
