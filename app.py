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
import math # Bazı matematiksel işlemler için

# --- Durum takibi için global değişkenler ---
LOGO_PROCESS_SUCCESS = False
LOGO_ERROR_MESSAGE = ""
icon_preview_obj = None

# ------------------------------------------------------------
# 1. LOGO VE İKON İŞLEME (SADECE BASE64)
# ------------------------------------------------------------

def process_logo_for_ios(image_path):
    """
    Logoyu işler, 120x120 kare boyuta getirir ve PURE Base64 string olarak döndürür.
    Disk kaydetme denemesi kaldırılmıştır.
    """
    global LOGO_PROCESS_SUCCESS, LOGO_ERROR_MESSAGE, icon_preview_obj
    try:
        # Kodun ARADIĞI KAYNAK dosya: logo.jpg (veya GitHub'daki tam ad neyse)
        img = Image.open(image_path)
        
        # 1. Şeffaf (PNG) ise beyaz zemin ekle (JPG olduğundan genellikle gerekmez, ama kontrol iyi)
        if img.mode in ('RGBA', 'LA'):
            background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
            background.paste(img, img.split()[-1])
            img = background
        
        # 2. Mutlak Sol Kare Kesim (Logo solda olduğu için soldan kare kesim)
        width, height = img.size
        side = min(width, height) # Kısa kenarı al
        left = 0
        top = 0
        right = side
        bottom = side
        
        # Eğer resim yatay ise (width > height), kareyi soldan kes.
        if width > height:
             img_square_cropped = img.crop((left, top, height, bottom))
        # Eğer resim dikey veya kare ise
        else:
             img_square_cropped = img.crop((left, top, right, bottom))
        
        # 3. İkon boyutuna (120x120) küçült/büyüt
        img_final_icon = img_square_cropped.resize((120, 120))
        icon_preview_obj = img_final_icon

        # 4. KRİTİK: Base64 stringini oluştur
        buffered = io.BytesIO()
        # İkon olarak kullanılacağı için PNG formatında kaydedilir
        img_final_icon.save(buffered, format="PNG") 
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
            <link rel="apple-touch-icon" href
