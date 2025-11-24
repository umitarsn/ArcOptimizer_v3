import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import time 

# Diğer importlar
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import math 

# --- Durum takibi için global değişkenler ---
LOGO_PROCESS_SUCCESS = False
LOGO_ERROR_MESSAGE = ""
icon_preview_obj = None

# ------------------------------------------------------------
# 1. LOGO VE İKON İŞLEME (SADECE BASE64)
# ------------------------------------------------------------

def process_logo_for_ios(image_path):
    """
    Logoyu işler, 192x192 kare boyuta getirir ve PURE Base64 string olarak döndürür.
    Renk modunu ve kesme mantığını sağlamlaştırdık.
    """
    global LOGO_PROCESS_SUCCESS, LOGO_ERROR_MESSAGE, icon_preview_obj
    try:
        # Kodun ARADIĞI KAYNAK dosya: logo.jpg
        # RGB'ye çevirerek görsel hataları minimize ediyoruz
        img = Image.open(image_path).convert("RGB") 
        
        # 1. Şeffaf (PNG) ise beyaz zemin ekle
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        
        # 2. Mutlak Sol Kare Kesim 
        width, height = img.size
        side = min(width, height) 
        # Soldan kare kesim her zaman doğru çalışır: (0, 0, kısa kenar, kısa kenar)
        img_square_cropped = img.crop((0, 0, side, side)) 
        
        # 3. İkon boyutuna küçült/büyüt 
        # st.image önizlemesi için 120x120
        icon_preview_obj = img_square_cropped.resize((120, 120)) 
        
        # iOS ve Favicon için 192x192 (Base64 enjeksiyonunu daha sağlam yapar)
        img_final_base64 = img_square_cropped.resize((192, 192)) 

        # 4. KRİTİK: Base64 stringini oluştur
        buffered = io.BytesIO()
        img_final_base64.save(buffered, format="PNG") 
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        LOGO_PROCESS_SUCCESS = True
        
        # Base64 stringini ve orijinal logo objesini döndür.
        return f"data:image/png;base64,{img_str}", img 

    except FileNotFoundError:
        LOGO_ERROR_MESSAGE = f"❌ Hata: '{image_path}' dosyası bulunamadı. Lütfen dosya adını ve GitHub'daki büyük/küçük harfleri kontrol edin."
        return None, None
    except Exception as e:
        LOGO_ERROR_MESSAGE = f"⚠️ Logo işleme hatası: {e}"
        return None, None

# logo.jpg'yi girdi olarak kullan
icon_href, original_logo_obj = process_logo_for_ios("logo.jpg")

# ------------------------------------------------------------
# 2. SAYFA AYARLARI VE HTML ENJEKSİYONU
# ------------------------------------------------------------
st.set_page_config(
    page_title="Ferrokrom AI",
    layout="wide",
    page_icon=icon_preview_obj if icon_preview_obj else "⚒️", 
    initial_sidebar_state="expanded"
)

# iOS Ana Ekran İkonu Enjeksiyonu
if icon_href:
    # KRİTİK: Zaman damgası cache buster olarak kullanılıyor.
    cache_buster_time = int(time.time()) 
    
    st.markdown(
        f"""
        <head>
            <link rel="icon" type="image/png" sizes="192x192" href="{icon_href}">
            
            <link rel="apple-touch-icon" sizes="120x120" href="{icon_href}">
            <link rel="apple-touch-icon" sizes="180x180" href="{icon_href}">
            
            <meta name="apple-mobile-web-app-title" content="Ferrokrom AI - {cache_buster_time}">
            <meta name="apple-mobile-web-app-capable" content="yes">
            <meta name="apple-mobile-web-app-status-bar-style" content="black">
        </head>
        """,
        unsafe_allow_html=True
    )

# Streamlit Üst Bar Logosu
try:
    if original_logo_obj:
        # st.logo burada logo.jpg dosyasını kullanır
        st.logo("logo.jpg", icon_image="logo.jpg")
except:
    pass

# ------------------------------------------------------------
# 3. UYGULAMA ANA AKIŞI FONKSİYONLARI
# ------------------------------------------------------------

@st.cache_data
def generate_dummy_trend_data(n_points=50):
    dates = pd.date_range(start="2023-01-01", periods=n_points, freq="D")
    panel_temps = np.random.normal(35, 5, n_points) + np.linspace(0, 10, n_points)
    stability = np.random.normal(85, 5, n_points) + np.sin(np.linspace(0, 4*np.pi, n_points)) * 5
    return pd.DataFrame({
        "Tarih": dates,
        "Panel_Temp_Avg": panel_temps,
        "Arc_Stability_KPI": stability.clip(0, 100)
    })

@st.cache_data
def generate_dummy_scrap_data():
    data = {
        'Scrap_Type': ['Heavy Melt Steel', 'Shredded Scrap', 'Busheling Scrap', 'Heavy Melt Steel', 'Plate/Structural'],
        'Supplier': ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier A'],
        'Price_USD_t': [420, 380, 450, 410, 480],
        'Quality_Index': [90, 75, 95, 85, 98],
        'Lot_tonnage': [1500, 1000, 800, 2000, 1200],
        'kWh_per_t': [380, 450, 350, 400, 320]
    }
    return pd.DataFrame(data)

@st.cache_data
def feature_engineering(df):
    if 'panel_T_out_C' in df.columns and 'panel_T_in_C' in df.columns:
        df['Panel_Temp_Delta_C'] = df['panel_T_out_C'] - df['panel_T_in_C']
    if 'power_kWh' in df.columns and 'tap_time_min' in df.columns:
        df['Energy_Rate'] = df['power_kWh'] / df['tap_time_min']
    
    # Simülasyon için eksik sütunları ortalama ile doldur
    if 'Scrap_Quality_Index' not in df.columns:
        df['Scrap_Quality_Index'] = 80
    if 'Thermal_Stress_Index' not in df.columns:
        df['Thermal_Stress_Index'] = 10
        
    return df

def create_gauge_chart(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Tahmini Döküm Sıcaklığı (°C)"},
