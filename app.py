import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------
# 1. SAYFA VE LOGO AYARLARI
# ------------------------------------------------------------

# Logoyu yüklemeye çalışalım (Hata almamak için try-except bloğu)
try:
    logo_img = Image.open("logo.png")
    page_icon_img = logo_img
except FileNotFoundError:
    logo_img = None
    page_icon_img = "⚒️" # Logo yoksa maden emojisi

st.set_page_config(
    page_title="BG Maden AI",
    layout="wide",
    page_icon=page_icon_img,
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# 2. YARDIMCI FONKSİYONLAR & SİMÜLASYON MOTORU
# ------------------------------------------------------------

@st.cache_data
def generate_dummy_trend_data(n_points=50):
    """Trend grafikleri için yapay zaman serisi verisi üretir."""
    dates = pd.date_range(start="2023-01-01", periods=n_points, freq="D")
    
    # Panel Sıcaklık Trendi (Artan trend = Aşınma simülasyonu)
    panel_temps = np.random.normal(35, 5, n_points) + np.linspace(0, 10, n_points)
    
    # Ark Stabilite Trendi (Dalgalı yapı)
    stability = np.random.normal(85, 5, n_points) + np.sin(np.linspace(0, 4*np.pi, n_points)) * 5
    
    return pd.DataFrame({
        "Tarih": dates,
        "Panel_Temp_Avg": panel_temps,
        "Arc_Stability_KPI": stability.clip(0, 100)
    })

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Ham veriden Termal Stres ve Kalite İndekslerini türetir."""
    df = df.copy()
    
    # Termal Stres İndeksi Hesaplama
    required_thermal_cols = ["panel_T_in_C", "panel_T_out_C", "panel_flow_kg_s", "power_kWh"]
    if all(col in df.columns for col in required_thermal_cols):
        cp_kJ = 4.18  
        df['Q_Panel_kW'] = df['panel_flow_kg_s'] * (df['panel_T_out_C'] - df['panel_T_in_C']) * cp_kJ 
        df['Thermal_Stress_Index'] = (df['Q_Panel_kW'] * 0.1) + (df['power_kWh'] * 0.005) 
        
        # Normalize et (0-100)
        max_val = df['Thermal_Stress_Index'].max()
        df['Thermal_Stress_Index'] = (df['Thermal_Stress_Index'] / max_val * 100) if max_val > 0 else 50.0
        df = df.drop(columns=['Q_Panel_kW'])

    # Hurda Kalite İndeksi Hesaplama
    required_scrap_cols = ["scrap_HMS80_20_pct", "scrap_HBI_pct", "scrap_Shredded_pct"]
    if all(col in df.columns for col in required_scrap_cols):
        df['Scrap_Quality_Index'] = (
            df['scrap_HBI_pct'] * 1.0 + 
            df['scrap_Shredded_pct'] * 0.7 + 
            df['scrap_HMS80_20_pct'] * 0.4
        )
        df = df.drop(columns=required_scrap_cols, errors='ignore') 
    
    # Eski isim uyumluluğu için rename (varsa)
    if 'Thermal_Imbalance_Index' in df.columns:
        df = df.rename(columns={'Thermal_Imbalance_Index': 'Thermal_Stress_Index'})
        
    return df

def create_gauge_chart(value, title="Sıcaklık", min_v=1500, max_v=1750, target=1620):
    """Profesyonel Gauge (İbreli) Gösterge."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        title = {'text': title},
        delta = {'reference': target, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [min_v, max_v], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'steps': [
                {'range': [min_v, 1600], 'color': '#4dabf5'}, # Soğuk
                {'range': [1600, 1640], 'color': '#66ff66'}, # İdeal
                {'range': [1640, max_v], 'color': '#ff6666'}], # Sıcak
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1700}}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    return fig

def generate_cfd_fields(power, arc_deviation_pct):
    """
    Bilimsel CFD Simülasyonu: 
    Ark gücüne göre havuz hacmi ve sapma yüzdesine göre merkez kayması.
    Ayrıca sıvı metal hareketini göstermek için vektörler (quiver) üretir.
    """
    nx, ny = 50, 50
    x = np.linspace(0, 10, nx); y = np.linspace(0, 10, ny)
    X, Y = np.meshgrid(x, y)
    
    # Sapma Miktarı: %40 sapma -> ~2 metre kayma
    deviation_amount = (arc_deviation_pct / 100.0) * 5.0
    center_x = 5.0 + deviation_amount * np.cos(np.pi/4) 
    center_y = 5.0 + deviation_amount * np.sin(np.pi/4)
    
    dist_sq = (X - center_x)**2 + (Y - center_y)**2
    
    # Dağılım: Güç arttıkça havuz genişler
    diffusion_factor = 8.0 + (power / 400.0) 
    max_arc_temp = 1600 + (power * 0.06) 
    
    # Sıcaklık Alanı
    temp_field = max_arc_temp * np.exp(-dist_sq / diffusion_factor)
    temp_field = np.maximum(temp_field, 1500) # Min banyo sıcaklığı
    
    # Akış Vektörleri (Sıvı Metal Hareketi)
    # Merkezden dışa doğru termal konveksiyon ve manyetik dönme etkisi
    angle = np.arctan2(Y - center_y, X - center_x)
    radius = np.sqrt(dist_sq)
    
    # Hız büyüklüğü güce bağlı
    vel_mag = (power / 5000.0) * np.exp(-radius/3.0)
    
    # Dönme (Vortex) + Radyal Genişleme
    V_x = -vel_mag * np.sin(angle) + (vel_mag * 0.3 * np.cos(angle))
    V_y = vel_mag * np.cos(angle) + (vel_mag * 0.3 * np.sin(angle))
    
    return X, Y, temp_field, V_x, V_y

# ------------------------------------------------------------
# 3. ANA UYGULAMA AKIŞI
# ------------------------------------------------------------
def main():
    # --- SOL MENÜ: LOGO VE NAVİGASYON ---
    if logo_img:
        st.sidebar.image(logo_img, use_container_width=True)
    
    st.sidebar.title("BG Maden AI")
    st.sidebar.markdown("**Akıllı Karar Destek Sistemi**")
    st.sidebar.markdown("---")
    
    selected_module = st.sidebar.radio(
        "📑 Modül Seçimi:",
        [
            "1️⃣ AI Bakım ve Duruş Engelleme",
            "2️⃣ AI Girdi Maliyetleri Düşürme",
            "3️⃣ Karar Destek Modülü (Process)",
            "4️⃣ Alarm, Tavsiye ve KPI'lar",
            "5️⃣ AI Enterprise Level (EBITDA)"
        ]
    )
    
    st.sidebar.markdown("---")

    # --- VERİ YÜKLEME VE MODEL EĞİTİMİ ---
    try:
        df = pd.read_csv("data/BG_EAF_panelcooling_demo.csv")
    except FileNotFoundError:
        st.error("❌ Veri dosyası bulunamadı! Lütfen 'data/BG_EAF_panelcooling_demo.csv' yolunu kontrol edin.")
        st.stop()

    # Feature Engineering
    df = feature_engineering(df)
    
    # Model Eğitimi (Random Forest)
    target_col = "tap_temperature_C"
    drop_cols = ["heat_id", "tap_temperature_C", "melt_temperature_C", "panel_T_in_C", "panel_T_out_C", "panel_flow_kg_s"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target_col], errors='ignore')
    y = df[target_col]
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Trend Verisi
    trend_df = generate_dummy_trend_data()
    tonnage = 10.0 # Varsayılan tonaj

    # ------------------------------------------------------------------
    # ORTAK GİRDİLER (SIDEBAR - KONTROL PANELİ)
    # ------------------------------------------------------------------
    st.sidebar.header("🎛️ Simülasyon Kontrol Paneli")
    
    input_data = {}

    # 1. Ark Stabilizasyonu (En Kritik Girdi)
    arc_stability_factor = st.sidebar.slider(
        "⚡ Ark Stabilizasyon Faktörü (0-1)", 
        0.0, 1.0, 0.90, 0.01,
        help="1.0 = Tam Merkezde/Stabil. Düşük değer = Yüksek Sapma/Risk."
    )
    
    # Ark stabilitesine göre Termal Stres ve Sapma Yüzdesi türetilir
    calculated_stress = (1.0 - arc_stability_factor) * 100
    input_data['Thermal_Stress_Index'] = calculated_stress
    
    # 2. Proses Girdileri
    for col in X.columns:
        if col == 'power_kWh':
            input_data[col] = st.sidebar.slider("Güç (kWh)", 3000.0, 5000.0, 4000.0)
        elif col == 'oxygen_Nm3':
            input_data[col] = st.sidebar.slider("Oksijen (Nm3)", 100.0, 300.0, 200.0)
        elif col == 'Scrap_Quality_Index':
            input_data[col] = st.sidebar.slider("Hurda Kalitesi (0-100)", 0.0, 100.0, 70.0)
        elif col == 'tap_time_min':
            input_data[col] = st.sidebar.slider("Döküm Süresi (dk)", 40.0, 70.0, 55.0)
        elif col != 'Thermal_Stress_Index': # Bunu zaten hesapladık
            input_data[col] = df[col].mean()

    # 3. Fiyat Girdileri (Sadece ilgili modüllerde gösterilebilir ama kolaylık için burada)
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Piyasa Fiyatları")
    price_scrap = st.sidebar.number_input("Hurda ($/ton)", 200., 600., 400.)
    price_elec = st.sidebar.number_input("Elektrik ($/kWh)", 0.05, 0.3, 0.10)
    price_oxy = st.sidebar.number_input("Oksijen ($/Nm3)", 0.05, 0.5, 0.15)
    price_electrode = st.sidebar.number_input("Elektrot ($/kg)", 2.0, 8.0, 4.5)

    # --- ORTAK HESAPLAMALAR ---
    input_df = pd.DataFrame([input_data])[X.columns]
    prediction = model.predict(input_df)[0] # Sıcaklık Tahmini
    
    # KPI Hesaplamaları
    arc_deviation_pct = (1.0 - arc_stability_factor) * 40.0 # %0-40 arası sapma
    
    # ------------------------------------------------------------------
    # MODÜL 1: AI BAKIM VE DURUŞ ENGELLEME
    # ------------------------------------------------------------------
    if selected_module == "1️⃣ AI Bakım ve Duruş Engelleme":
        st.title("🛡️ Modül 1: AI Bakım ve Duruş Engelleme")
        st.markdown("Fırın refrakter sağlığı ve panel soğutma sistemi risk analizi.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Panel Sıcaklık Trendi (Aşınma İzleme)")
            fig_trend = px.line(trend_df, x="Tarih", y="Panel_Temp_Avg", title="Günlük Ortalama Panel Çıkış Suyu Sıcaklığı")
            fig_trend.add_hline(y=45, line_dash="dot", annotation_text="Risk Limiti", line_color="red")
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with col2:
            st.subheader("Anlık Panel Sağlığı")
            # Sağlık skoru Stres ile ters orantılı
            health_score = 100 - calculated_stress
            
            fig_health = go.Figure(go.Indicator(
                mode = "gauge+number", value = health_score,
                title = {'text': "Sağlık Skoru"},
                gauge = {
                    'axis': {'range': [0, 100]}, 
                    'bar': {'color': "green" if health_score > 50 else "red"},
                    'steps': [{'range': [0, 30], 'color': '#ffcccc'}, {'range': [70, 100], 'color': '#ccffcc'}]
                }
            ))
            fig_health.update_layout(height=300)
            st.plotly_chart(fig_health, use_container_width=True)
            
            if health_score < 40:
                st.error("🚨 **KRİTİK:** Panel delinme riski yüksek! Ark stabilizasyonu bozuk.")
            else:
                st.success("✅ Panel durumu stabil.")

    # ------------------------------------------------------------------
    # MODÜL 2: AI GİRDİ MALİYETLERİ DÜŞÜRME
    # ------------------------------------------------------------------
    elif selected_module == "2️⃣ AI Girdi Maliyetleri Düşürme":
        st.title("💰 Modül 2: Girdi Maliyetleri Optimizasyonu")
        
        # Maliyet Hesabı
        c_scrap = tonnage * price_scrap
        c_elec = input_data['power_kWh'] * price_elec
        c_oxy = input_data['oxygen_Nm3'] * price_oxy
        c_elec_rod = tonnage * 1.8 * price_electrode
        total = c_scrap + c_elec + c_oxy + c_elec_rod
        unit_cost = total / tonnage

        col_pie, col_metric = st.columns([1, 1])
        
        with col_pie:
            st.subheader("Maliyet Kırılımı")
            df_cost = pd.DataFrame({
                "Kalem": ["Hurda", "Elektrik", "Oksijen", "Elektrot"],
                "Tutar": [c_scrap, c_elec, c_oxy, c_elec_rod]
            })
            fig_pie = px.pie(df_cost, values='Tutar', names='Kalem', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_metric:
            st.subheader("Birim Maliyet Analizi")
            st.metric("Mevcut Maliyet", f"${unit_cost:.2f} / ton")
            st.metric("Hedef Maliyet", "$450.00 / ton", delta=f"${unit_cost-450:.2f}")
            
            st.info(f"ℹ️ Hurda kalitesi şu an **{input_data['Scrap_Quality_Index']:.0f}**. Daha yüksek kaliteli hurda (HBI vb.), elektrik tüketimini düşürerek toplam maliyeti dengeleyebilir.")

    # ------------------------------------------------------------------
    # MODÜL 3: KARAR DESTEK MODÜLÜ (PROSES & CFD)
    # ------------------------------------------------------------------
    elif selected_module == "3️⃣ Karar Destek Modülü (Process)":
        st.title("🎯 Modül 3: Karar Destek ve Dijital İkiz")
        
        col_temp, col_cfd = st.columns([1, 2])
        
        with col_temp:
            st.subheader("Sıcaklık Tahmini")
            st.plotly_chart(create_gauge_chart(prediction), use_container_width=True)
            
            st.markdown("### ⚡ Ark Durumu")
            st.metric("Ark Sapma Yüzdesi", f"%{arc_deviation_pct:.1f}", delta_color="inverse" if arc_deviation_pct > 20 else "normal")
            if arc_deviation_pct > 20:
                st.warning("⚠️ Ark merkezden kaymış durumda.")
            else:
                st.success("✅ Ark merkezde ve stabil.")
            
        with col_cfd:
            st.subheader("Fırın İçi Akışkan ve Isı Dinamiği (CFD)")
            st.info("Bu simülasyon, ark gücü ve sapmasına bağlı olarak **sıvı metal havuzunun şeklini** ve **hareket yönünü (oklar)** gösterir.")
            
            # CFD Hesabı
            pwr = input_data['power_kWh']
            X, Y, T, Vx, Vy = generate_cfd_fields(pwr, arc_deviation_pct)
            
            fig_cfd, ax = plt.subplots(figsize=(8, 5))
            # Isı haritası
            c = ax.contourf(X, Y, T, levels=25, cmap='inferno')
            # Akış Vektörleri (Movement)
            ax.quiver(X[::4, ::4], Y[::4, ::4], Vx[::4, ::4], Vy[::4, ::4], color='white', alpha=0.6)
            
            fig_cfd.colorbar(c, label='Sıcaklık (°C)')
            ax.set_title(f"Sıvı Metal Havuzu (Güç: {pwr} kWh, Sapma: %{arc_deviation_pct:.1f})")
            ax.set_xlabel("Fırın Genişliği (m)")
            ax.set_ylabel("Fırın Derinliği (m)")
            st.pyplot(fig_cfd)

    # ------------------------------------------------------------------
    # MODÜL 4: ALARM, TAVSİYE VE KPI
    # ------------------------------------------------------------------
    elif selected_module == "4️⃣ Alarm, Tavsiye ve KPI'lar":
        st.title("📢 Modül 4: Alarm Merkezi ve KPI Takibi")
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Ark Stabilite KPI", f"{arc_stability_factor*100:.1f}", delta="Hedef > 85")
        k2.metric("Enerji Tüketimi", f"{(input_data['power_kWh']/tonnage):.1f} kWh/t")
        k3.metric("Döküm Süresi", f"{input_data['tap_time_min']:.0f} dk")
        alarm_count = 1 if arc_deviation_pct > 20 else 0
        k4.metric("Aktif Alarm", f"{alarm_count}", delta_color="inverse")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Stabilite Geçmişi")
            fig_stab = px.area(trend_df, x="Tarih", y="Arc_Stability_KPI", title="Ark Stabilizasyon Performansı")
            st.plotly_chart(fig_stab, use_container_width=True)
            
        with c2:
            st.subheader("📋 Operatör Aksiyon Paneli")
            if alarm_count > 0:
                st.error("🛑 **ACİL AKSİYON:** Ark sapması sınır değerin üzerinde! DC akım dengesini kontrol edin.")
            elif prediction > 1650:
                st.warning("⚠️ **UYARI:** Aşırı ısınma. Güç kademesini düşürün.")
            else:
                st.success("✅ **DURUM:** Proses optimum aralıkta çalışıyor.")

    # ------------------------------------------------------------------
    # MODÜL 5: AI ENTERPRISE LEVEL (EBITDA)
    # ------------------------------------------------------------------
    elif selected_module == "5️⃣ AI Enterprise Level (EBITDA)":
        st.title("🏢 Modül 5: Kurumsal İş Zekası (EBITDA)")
        
        with st.expander("📊 Finansal Hedef Ayarları", expanded=True):
            c_e1, c_e2 = st.columns(2)
            sales_price = c_e1.number_input("Hedef Satış Fiyatı ($/ton)", 500, 2000, 900)
            monthly_target = c_e2.number_input("Aylık Hedef Tonaj", 1000, 50000, 10000)
            fixed_cost = st.number_input("Aylık Sabit Giderler ($)", 100000, 2000000, 500000)

        # EBITDA Hesabı
        var_cost_total = unit_cost * monthly_target
        revenue = sales_price * monthly_target
        gross = revenue - var_cost_total
        ebitda = gross - fixed_cost
        
        # Waterfall Grafiği
        fig_water = go.Figure(go.Waterfall(
            measure = ["relative", "relative", "total", "relative", "total"],
            x = ["Satış Geliri", "Değişken Mal.", "Brüt Kar", "Sabit Gider", "EBITDA"],
            text = [f"{revenue/1e6:.1f}M", f"-{var_cost_total/1e6:.1f}M", f"{gross/1e6:.1f}M", f"-{fixed_cost/1e6:.1f}M", f"{ebitda/1e6:.1f}M"],
            y = [revenue, -var_cost_total, 0, -fixed_cost, 0],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_water.update_layout(title="Aylık Karlılık Şelalesi ($)", showlegend=False)
        st.plotly_chart(fig_water, use_container_width=True)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Aylık Ciro", f"${revenue:,.0f}")
        m2.metric("EBITDA", f"${ebitda:,.0f}", delta_color="normal" if ebitda>0 else "inverse")
        m3.metric("EBITDA Marjı", f"%{(ebitda/revenue)*100:.1f}")

if __name__ == "__main__":
    main()
