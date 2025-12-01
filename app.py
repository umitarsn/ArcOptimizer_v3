import os
import json
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import altair as alt

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

# ----------------------------------------------
# SETUP VERİLERİ
# ----------------------------------------------
if os.path.exists(SETUP_SAVE_PATH):
    with open(SETUP_SAVE_PATH, "r", encoding="utf-8") as f:
        saved_inputs = json.load(f)
else:
    saved_inputs = {}

if "info_state" not in st.session_state:
    st.session_state.info_state = {}

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
# SİMÜLASYON VERİ ÜRETİCİSİ
# ----------------------------------------------
def generate_simulation_runtime_data(n: int = 15):
    sim_list = []
    now = datetime.now(TZ)
    for i in range(n):
        ts = now - timedelta(hours=(n - 1 - i))
        heat_id = f"SIM-{i+1}"
        tap_weight = 35 + random.uniform(-3, 3)
        kwh_per_t = 420 + random.uniform(-25, 25)
        energy_kwh = tap_weight * kwh_per_t
        duration_min = 55 + random.uniform(-10, 10)
        tap_temp = 1610 + random.uniform(-15, 15)
        o2_flow = 950 + random.uniform(-150, 150)
        slag_foaming = random.randint(3, 9)
        panel_delta_t = 18 + random.uniform(-5, 8)
        electrode_cons = 1.9 + random.uniform(-0.3, 0.3)
        sim_list.append(
            {
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
                "operator_note": "Simülasyon kaydı",
            }
        )
    return sim_list

# ----------------------------------------------
# EXCEL'DEN SORU TABLOLARI
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
        "3. Detaylı bilgi ve açıklama için ℹ️ simgesine tıklayınız."
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
                    birim_str = str(raw_birim).strip()
                    if birim_str.lower() not in ("", "none", "nan"):
                        birim = birim_str

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
                        else:
                            st.markdown("")

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
                    if pd.notna(onem_raw):
                        detaylar.append("🔵 **Önem:** " + str(onem))
                    if detaylar:
                        st.info("\n".join(detaylar))

                total_fields += 1
                kayit_degeri = str(saved_inputs.get(val_key, "")).strip()
                if kayit_degeri:
                    total_filled += 1
                    if onem == 1:
                        required_filled += 1
                if onem == 1:
                    required_fields += 1

    st.sidebar.subheader("📊 Setup Veri Giriş Durumu")
    pct_all = round(100 * total_filled / total_fields, 1) if total_fields else 0.0
    pct_req = round(100 * required_filled / required_fields, 1) if required_fields else 0.0

    st.sidebar.metric("Toplam Giriş Oranı", f"{pct_all}%")
    st.sidebar.progress(min(pct_all / 100, 1.0))
    st.sidebar.metric("Zorunlu Veri Girişi", f"{pct_req}%")
    st.sidebar.progress(min(pct_req / 100, 1.0))
    eksik = required_fields - required_filled
    if eksik > 0:
        st.sidebar.warning(f"❗ Eksik Zorunlu Değerler: {eksik}")

# ----------------------------------------------
# 2) CANLI VERİ SAYFASI
# ----------------------------------------------
def show_runtime_page(sim_mode: bool):
    st.markdown("## 2. Canlı Veri – Şarj Bazlı Anlık Veriler")
    if sim_mode:
        st.info("🧪 **Simülasyon Modu Aktif.** Veriler simülasyon amaçlı oluşturulur ve kaydedilmez.")
    else:
        st.markdown(
            "Bu sayfada fırın işletmesi sırasında her **şarj / heat** için toplanan "
            "operasyonel veriler girilir veya otomasyon sisteminden okunur."
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
            st.error("Heat ID / Şarj No girilmesi zorunludur.")
        else:
            if sim_mode:
                st.warning("Simülasyon Modu açıkken yeni veri kalıcı olarak kaydedilmez.")
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

    data_source = generate_simulation_runtime_data() if sim_mode else runtime_data
    if not data_source:
        st.info("Henüz canlı veri girilmemiş.")
   
