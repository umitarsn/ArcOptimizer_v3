import os
import json
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo   # Türkiye saati için
import pandas as pd
import streamlit as st
import altair as alt

# ----------------------------------------------
# GENEL AYARLAR
# ----------------------------------------------
st.set_page_config(
    page_title="FeCr AI",               # Sekme / iOS varsayılan adı
    page_icon="apple-touch-icon.png",   # Proje root'taki logo dosyası
    layout="wide",
)

# Türkiye saati
TZ = ZoneInfo("Europe/Istanbul")

# Sabit inputların kaydedileceği dosya
SETUP_SAVE_PATH = "data/saved_inputs.json"
# Runtime (şarj bazlı) verilerin kaydedileceği dosya
RUNTIME_SAVE_PATH = "data/runtime_data.json"

os.makedirs("data", exist_ok=True)

# ----------------------------------------------
# KAYITLI SETUP VERİLERİNİ YÜKLE
# ----------------------------------------------
if os.path.exists(SETUP_SAVE_PATH):
    with open(SETUP_SAVE_PATH, "r", encoding="utf-8") as f:
        saved_inputs = json.load(f)
else:
    saved_inputs = {}

if "info_state" not in st.session_state:
    st.session_state.info_state = {}

# ----------------------------------------------
# RUNTIME VERİLERİ YÜKLE / KAYDET
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
# SİMÜLASYON VERİ ÜRETİCİSİ
# ----------------------------------------------
def generate_simulation_runtime_data(n=15):
    """Simülasyon Modu için örnek şarj datası üretir."""
    sim_list = []
    # Türkiye saatiyle şu an
    now = datetime.now(TZ)

    # İlk nokta: now - (n-1) saat, son nokta: tam "şu an"
    for i in range(n):
        ts = now - timedelta(hours=(n - 1 - i))
        heat_id = f"SIM-{i+1}"

        tap_weight = 35 + random.uniform(-3, 3)          # ton
        kwh_per_t = 420 + random.uniform(-25, 25)        # kWh/t
        energy_kwh = tap_weight * kwh_per_t
        duration_min = 55 + random.uniform(-10, 10)      # dk
        tap_temp = 1610 + random.uniform(-15, 15)        # °C
        o2_flow = 950 + random.uniform(-150, 150)        # Nm³/h
        slag_foaming = random.randint(3, 9)              # 0–10
        panel_delta_t = 18 + random.uniform(-5, 8)       # °C
        electrode_cons = 1.9 + random.uniform(-0.3, 0.3) # kg/şarj

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
