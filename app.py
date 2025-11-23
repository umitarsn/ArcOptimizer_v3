import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# --- Durum takibi için global değişkenler ---
LOGO_PROCESS_SUCCESS = False
LOGO_ERROR_MESSAGE = ""
icon_preview_obj = None
ICON_FILE_NAME = "apple-icon.png" # Streamlit klasörüne kaydedilecek ikon adı

# ------------------------------------------------------------
# 1. LOGO VE İKON İŞLEME (DOSYAYA KAYDETME İLE BASE64 BYPASS)
# ------------------------------------------------------------

def process_logo_for_ios(image_path):
    """
    Logoyu işler, 120x120 kare boyuta getirir ve Base64 sorununu aşmak için 
    doğrudan ICON_FILE_NAME olarak diske kaydeder.
    """
    global LOGO_PROCESS_SUCCESS, LOGO_ERROR_MESSAGE, icon_preview_obj
    is_file_linked = False 
    try:
        # 1. Logoyu aç
        img = Image.open(image_path)
        
        # 2. Şeffaf (PNG) ise beyaz zemin ekle
        if img.mode in ('RGBA', 'LA'):
            background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
            background.paste(img, img.split()[-1])
            img = background
        
        # 3. Mutlak Sol Kare Kesim (Şekil + BG kısmı)
        width, height = img.size
        left, top, right, bottom = 0, 0, height, height
        img_square_cropped = img.crop((left, top, right, bottom))
        
        # 4. İkon boyutuna (120x120) küçült/büyüt
        img_final_icon = img_square_cropped.resize((120, 120))
        icon_preview_obj = img_final_icon

        # 5. KRİTİK: Base64 yerine diske kaydet
        try:
            img_final_icon.save(ICON_FILE_NAME, format="PNG")
            LOGO_PROCESS_SUCCESS = True
            is_file_linked = True
            return ICON_FILE_NAME, img, is_file_linked # Dosya yolunu döndür

        except Exception as save_e:
            LOGO_ERROR_MESSAGE = f"⚠️ Dosya kaydetme hatası: {save_e}. Lütfen yetkileri kontrol edin."
            # Kaydetme başarısız olursa yine de Base64 döndür (uyumsuz olsa bile)
            buffered = io.BytesIO()
            img_final_icon.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            LOGO_PROCESS_SUCCESS = True
            return f"data:image/png;base64,{img_str}", img, is_file_linked
        
    except FileNotFoundError:
        LOGO_ERROR_MESSAGE = f"❌ Hata: '{image_path}' dosyası bulunamadı. Lütfen dosya adını kontrol edin."
        return None, None, False
    except Exception as e:
        LOGO_ERROR_MESSAGE = f"⚠️ Logo işleme hatası: {e}"
        return None, None, False

# logo.jpg kullanılıyor. Eğer logo.png kullanıyorsanız burayı değiştirin.
icon_href, original_logo_obj, is_file_linked = process_logo_for_ios("logo.jpg")

# ------------------------------------------------------------
# 2. SAYFA AYARLARI
# ------------------------------------------------------------
st.set_page_config(
    page_title="Ferrokrom AI",
    layout="wide",
    page_icon=icon_preview_obj if icon_preview_obj else "⚒️", 
    initial_sidebar_state="expanded"
)

# iOS Ana Ekran İkonu Enjeksiyonu
if icon_href:
    # KRİTİK DÜZELTME: Sürüm parametresi eklenerek (cache buster) iOS'un ikonu yenilemeye zorlanması
    # icon_href burada ya "apple-icon.png" ya da Base64 verisidir.
    cache_buster = "?v=1.0" if is_file_linked else ""
    icon_link = f"{icon_href}{cache_buster}" 
    
    st.markdown(
        f"""
        <head>
            <link rel="apple-touch-icon" href="{icon_link}">
            <link rel="apple-touch-icon" sizes="120x120" href="{icon_link}">
            <meta name="apple-mobile-web-app-title" content="Ferrokrom AI">
            <meta name="apple-mobile-web-app-capable" content="yes">
            <meta name="apple-mobile-web-app-status-bar-style" content="black">
        </head>
        """,
        unsafe_allow_html=True
    )

# Streamlit Üst Bar Logosu
try:
    if original_logo_obj:
        st.logo("logo.jpg", icon_image="logo.jpg")
except:
    pass


# ------------------------------------------------------------
# 3. VERİ VE SİMÜLASYON FONKSİYONLARI (Değişmedi)
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

def generate_dummy_scrap_data(n_suppliers=4, n_lots=40):
    np.random.seed(42)
    suppliers = [f"Tedarikçi {chr(65 + i)}" for i in range(n_suppliers)]
    scrap_types = ["Krom İçi (High C)", "Paslanmaz Hurda", "Düşük Tenörlü Cevher", "Şarj Kromu"]
    rows = []
    for i in range(n_lots):
        rows.append({
            "Supplier": np.random.choice(suppliers),
            "Scrap_Type": np.random.choice(scrap_types),
            "Price_USD_t": round(np.random.uniform(200, 500), 1),
            "Quality_Index": round(np.random.uniform(60, 95), 1),
            "Lot_tonnage": round(np.random.uniform(30, 90), 1),
            "Yield_pct": round(np.random.uniform(85, 98), 1),
            "kWh_per_t": round(np.random.uniform(350, 450), 1),
            "Electrode_kg_per_t": round(np.random.uniform(1.5, 2.5), 2),
            "O2_Nm3_per_t": round(np.random.uniform(200, 250), 1)
        })
    return pd.DataFrame(rows)

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_thermal_cols = ["panel_T_in_C", "panel_T_out_C", "panel_flow_kg_s", "power_kWh"]
    if all(col in df.columns for col in required_thermal_cols):
        cp_kJ = 4.18  
        df['Q_Panel_kW'] = df['panel_flow_kg_s'] * (df['panel_T_out_C'] - df['panel_T_in_C']) * cp_kJ 
        df['Thermal_Stress_Index'] = (df['Q_Panel_kW'] * 0.1) + (df['power_kWh'] * 0.005) 
        max_val = df['Thermal_Stress_Index'].max()
        df['Thermal_Stress_Index'] = (df['Thermal_Stress_Index'] / max_val * 100) if max_val > 0 else 50.0
        df = df.drop(columns=['Q_Panel_kW'])

    required_scrap_cols = ["scrap_HMS80_20_pct", "scrap_HBI_pct", "scrap_Shredded_pct"]
    if all(col in df.columns for col in required_scrap_cols):
        df['Scrap_Quality_Index'] = (
            df['scrap_HBI_pct'] * 1.0 + 
            df['scrap_Shredded_pct'] * 0.7 + 
            df['scrap_HMS80_20_pct'] * 0.4
        )
        df = df.drop(columns=required_scrap_cols, errors='ignore') 
    
    return df.rename(columns={'Thermal_Imbalance_Index': 'Thermal_Stress_Index'})

def create_gauge_chart(value, title="Sıcaklık", min_v=1500, max_v=1750, target=1620):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        title = {'text': title},
        delta = {'reference': target, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [min_v, max_v], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'steps': [
                {'range': [min_v, 1600], 'color': '#4dabf5'},
                {'range': [1600, 1640], 'color': '#66ff66'},
                {'range': [1640, max_v], 'color': '#ff6666'}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1700}}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    return fig

def generate_cfd_fields(power, arc_deviation_pct):
    nx, ny = 50, 50
    x = np.linspace(0, 10, nx); y = np.linspace(0, 10, ny)
    X, Y = np.meshgrid(x, y)
    
    deviation_amount = (arc_deviation_pct / 100.0) * 5.0
    center_x = 5.0 + deviation_amount * np.cos(np.pi/4) 
    center_y = 5.0 + deviation_amount * np.sin(np.pi/4)
    dist_sq = (X - center_x)**2 + (Y - center_y)**2
    
    diffusion_factor = 8.0 + (power / 400.0) 
    max_arc_temp = 1600 + (power * 0.06) 
    temp_field = max_arc_temp * np.exp(-dist_sq / diffusion_factor)
    temp_field = np.maximum(temp_field, 1500) 
    
    angle = np.arctan2(Y - center_y, X - center_x)
    radius = np.sqrt(dist_sq)
    vel_mag = (power / 5000.0) * np.exp(-radius/3.0)
    V_x = -vel_mag * np.sin(angle) + (vel_mag * 0.3 * np.cos(angle))
    V_y = vel_mag * np.cos(angle) + (vel_mag * 0.3 * np.sin(angle))
    
    return X, Y, temp_field, V_x, V_y


# ------------------------------------------------------------
# 4. UYGULAMA ANA AKIŞI (Değişmedi)
# ------------------------------------------------------------
def main():
    # --- LOGO DEBUG VE MENÜ BAŞLIĞI ---
    if original_logo_obj:
        st.sidebar.image(original_logo_obj, use_container_width=True)
    else:
        st.sidebar.header("Ferrokrom AI")
        
    if LOGO_ERROR_MESSAGE:
        st.sidebar.error(LOGO_ERROR_MESSAGE)
    
    if LOGO_PROCESS_SUCCESS and icon_preview_obj:
        st.sidebar.markdown("---")
        st.sidebar.caption("✅ iOS İkon Önizlemesi:")
        st.sidebar.image(icon_preview_obj, width=80)
        
        if is_file_linked:
             st.sidebar.success(f"✅ Başarılı: İkon Base64 yerine **{ICON_FILE_NAME}** dosyasına bağlandı.")
        else:
            st.sidebar.warning("⚠️ Base64 Yedekleme Kullanılıyor.")
    st.sidebar.markdown("---")
    
    selected_module = st.sidebar.radio(
        "📑 Modül Seçimi:",
        [
            "1️⃣ AI Bakım ve Duruş Engelleme",
            "2️⃣ AI Girdi Maliyetleri Düşürme",
            "3️⃣ Karar Destek Modülü (Process)",
            "4️⃣ Alarm, Tavsiye ve KPI'lar",
            "5️⃣ AI Enterprise Level (EBITDA)",
            "6️⃣ Scrap & Purchase Intelligence"
        ]
    )
    st.sidebar.markdown("---")

    # --- VERİ YÜKLEME ---
    try:
        df = pd.read_csv("data/BG_EAF_panelcooling_demo.csv")
    except FileNotFoundError:
        st.error("❌ Veri dosyası bulunamadı! data/BG_EAF_panelcooling_demo.csv'yi kontrol edin.")
        st.stop()

    df = feature_engineering(df)
    
    target_col = "tap_temperature_C"
    drop_cols = ["heat_id", "tap_temperature_C", "melt_temperature_C", "panel_T_in_C", "panel_T_out_C", "panel_flow_kg_s"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target_col], errors='ignore')
    y = df[target_col]
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    trend_df = generate_dummy_trend_data()
    tonnage = 10.0 

    # ------------------------------------------------------------------
    # ORTAK GİRDİLER (SIDEBAR)
    # ------------------------------------------------------------------
    st.sidebar.header("🎛️ Simülasyon Kontrol Paneli")
    
    input_data = {}

    arc_stability_factor = st.sidebar.slider("⚡ Ark Stabilizasyon Faktörü (0-1)", 0.0, 1.0, 0.90, 0.01)
    calculated_stress = (1.0 - arc_stability_factor) * 100
    input_data['Thermal_Stress_Index'] = calculated_stress
    
    for col in X.columns:
        if col == 'power_kWh':
            input_data[col] = st.sidebar.slider("Güç (kWh)", 3000.0, 5000.0, 4000.0)
        elif col == 'oxygen_Nm3':
            input_data[col] = st.sidebar.slider("Oksijen (Nm3)", 100.0, 300.0, 200.0)
        elif col == 'Scrap_Quality_Index':
            input_data[col] = st.sidebar.slider("Hurda Kalitesi (0-100)", 0.0, 100.0, 70.0)
        elif col == 'tap_time_min':
            input_data[col] = st.sidebar.slider("Döküm Süresi (dk)", 40.0, 70.0, 55.0)
        elif col != 'Thermal_Stress_Index':
            input_data[col] = df[col].mean()

    if selected_module in ["2️⃣ AI Girdi Maliyetleri Düşürme", "5️⃣ AI Enterprise Level (EBITDA)", "6️⃣ Scrap & Purchase Intelligence"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("💰 Piyasa Fiyatları")
        price_scrap = st.sidebar.number_input("Hurda Fiyatı ($/t)", 200., 600., 400.)
        price_elec = st.sidebar.number_input("Elektrik Fiyatı ($/MWh)", 30, 200, 90)
        price_oxy = st.sidebar.number_input("Oksijen Fiyatı ($/Nm³)", 0.02, 1.00, 0.08, step=0.01)
        price_electrode = st.sidebar.number_input("Elektrot Fiyatı ($/kg)", 2.0, 15.0, 4.0, step=0.5)
    else:
        price_scrap, price_elec, price_oxy, price_electrode = 400, 90, 0.08, 4.0

    input_df = pd.DataFrame([input_data])[X.columns]
    prediction = model.predict(input_df)[0]
    panel_health_index = 100 - calculated_stress
    arc_deviation_pct = (1.0 - arc_stability_factor) * 40.0 

    # --- MODÜL İÇERİKLERİ (Değişmedi) ---
    if selected_module == "1️⃣ AI Bakım ve Duruş Engelleme":
        st.title("🛡️ Modül 1: AI Bakım & Duruş Engelleme")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Panel Sıcaklık Trendi")
            fig_trend = px.line(trend_df, x="Tarih", y="Panel_Temp_Avg", title="Panel Çıkış Suyu Sıcaklığı")
            fig_trend.add_hline(y=45, line_dash="dot", annotation_text="Limit", line_color="red")
            st.plotly_chart(fig_trend, use_container_width=True)
        with col2:
            st.subheader("Panel Sağlık Skoru")
            fig_health = go.Figure(go.Indicator(mode="gauge+number", value=panel_health_index, title={'text': "Sağlık"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green" if panel_health_index > 50 else "red"}}))
            fig_health.update_layout(height=250)
            st.plotly_chart(fig_health, use_container_width=True)
            if panel_health_index < 40: st.error("🚨 **KRİTİK:** Yüksek termal stres!")
            else: st.success("✅ Sistem Stabil.")

    elif selected_module == "2️⃣ AI Girdi Maliyetleri Düşürme":
        st.title("💰 Modül 2: Girdi Maliyetleri Optimizasyonu")
        cost_elec = (input_data['power_kWh'] * (price_elec / 1000.0))
        cost_oxy = input_data['oxygen_Nm3'] * price_oxy
        cost_scrap = tonnage * price_scrap
        cost_electrode = tonnage * 1.8 * price_electrode
        total_cost = cost_scrap + cost_elec + cost_oxy + cost_electrode
        unit_cost = total_cost / tonnage
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Toplam Döküm Maliyeti", f"${total_cost:,.2f}")
            st.metric("Birim Maliyet ($/t)", f"${unit_cost:.2f}")
        with c2:
            df_cost = pd.DataFrame({"Kalem": ["Hurda", "Elektrik", "Oksijen", "Elektrot"], "Maliyet": [cost_scrap, cost_elec, cost_oxy, cost_electrode]})
            fig_pie = px.pie(df_cost, values='Maliyet', names='Kalem', title="Maliyet Kırılımı", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

    elif selected_module == "3️⃣ Karar Destek Modülü (Process)":
        st.title("📈 Modül 3: Karar Destek ve Dijital İkiz")
        c_left, c_right = st.columns([1, 2])
        with c_left:
            st.subheader("Sıcaklık Tahmini")
            st.plotly_chart(create_gauge_chart(prediction), use_container_width=True)
            st.metric("Ark Sapma Yüzdesi", f"%{arc_deviation_pct:.1f}", delta_color="inverse" if arc_deviation_pct > 20 else "normal")
        with c_right:
            st.subheader("Fırın İçi Akışkan Dinamiği (CFD)")
            X, Y, T, Vx, Vy = generate_cfd_fields(input_data['power_kWh'], arc_deviation_pct)
            fig_cfd, ax = plt.subplots(figsize=(8, 5))
            c = ax.contourf(
