import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------
# 1. SAYFA AYARLARI
# ------------------------------------------------------------
st.set_page_config(
    page_title="BG-ArcOptimizer Enterprise",
    layout="wide",
    page_icon="🏭",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# 2. YARDIMCI FONKSİYONLAR & VERİ HAZIRLIĞI
# ------------------------------------------------------------

@st.cache_data
def generate_dummy_trend_data(n_points=50):
    """Trend grafikleri için sahte zaman serisi verisi üretir."""
    dates = pd.date_range(start="2023-01-01", periods=n_points, freq="D")
    
    # Panel Sıcaklık Trendi (Modül 1 için)
    panel_temps = np.random.normal(35, 5, n_points) + np.linspace(0, 10, n_points) # Artan trend (aşınma simülasyonu)
    
    # Maliyet Trendi (Modül 5 için)
    costs = np.random.normal(450, 20, n_points)
    
    return pd.DataFrame({
        "Tarih": dates,
        "Panel_Temp_Avg": panel_temps,
        "Cost_Per_Ton": costs
    })

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

def generate_cfd_fields(power):
    nx, ny = 50, 50
    x = np.linspace(0, 10, nx); y = np.linspace(0, 10, ny)
    X, Y = np.meshgrid(x, y)
    center_x, center_y = 5.0, 5.0
    dist_sq = (X - center_x)**2 + (Y - center_y)**2
    diffusion_factor = 10.0 + (power / 500.0) 
    max_arc_temp = 1600 + (power * 0.05) 
    temp_field = max_arc_temp * np.exp(-dist_sq / diffusion_factor)
    return X, Y, np.maximum(temp_field, 1500)

# ------------------------------------------------------------
# 3. ANA UYGULAMA
# ------------------------------------------------------------
def main():
    # --- SOL MENÜ NAVİGASYON ---
    st.sidebar.title("📑 Modül Seçimi")
    
    selected_module = st.sidebar.radio(
        "Görüntülemek istediğiniz modülü seçin:",
        [
            "1️⃣ AI Bakım ve Duruş Engelleme",
            "2️⃣ AI Girdi Maliyetleri Düşürme",
            "3️⃣ Karar Destek Modülü (Process)",
            "4️⃣ Alarm, Tavsiye ve KPI'lar",
            "5️⃣ AI Enterprise Level (EBITDA)"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"Şu an aktif modül: **{selected_module}**")

    # --- VERİ YÜKLEME ---
    # Demo veri yükleme (Arka planda)
    try:
        df = pd.read_csv("data/BG_EAF_panelcooling_demo.csv")
    except:
        st.error("Demo veri dosyası bulunamadı! Lütfen data klasörünü kontrol edin.")
        st.stop()

    df = feature_engineering(df)
    
    # Model Eğitimi (Her sayfada ihtiyaç duyulabilir)
    target_col = "tap_temperature_C"
    drop_cols = ["heat_id", "tap_temperature_C", "melt_temperature_C", "panel_T_in_C", "panel_T_out_C", "panel_flow_kg_s"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target_col], errors='ignore')
    y = df[target_col]
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Trend verisi
    trend_df = generate_dummy_trend_data()

    # ------------------------------------------------------------------
    # ORTAK GİRDİLER (SIDEBAR - Her zaman görünür olsun veya modüle özel)
    # ------------------------------------------------------------------
    st.sidebar.markdown("### 🎛️ Aktif Simülasyon Girdileri")
    
    # Bu girdiler model tahmini için tüm modüllerde gerekli
    input_data = {}
    
    # Modül 2 ve 5 için Fiyat Girdileri
    if selected_module in ["2️⃣ AI Girdi Maliyetleri Düşürme", "5️⃣ AI Enterprise Level (EBITDA)"]:
        st.sidebar.subheader("💰 Piyasa Fiyatları")
        price_scrap = st.sidebar.number_input("Hurda ($/ton)", 200., 600., 400.)
        price_elec = st.sidebar.number_input("Elektrik ($/kWh)", 0.05, 0.3, 0.10)
        price_oxy = st.sidebar.number_input("Oksijen ($/Nm3)", 0.05, 0.5, 0.15)
        price_electrode = st.sidebar.number_input("Elektrot ($/kg)", 2.0, 8.0, 4.5)
    else:
        # Varsayılan değerler
        price_scrap=400; price_elec=0.10; price_oxy=0.15; price_electrode=4.5

    # Proses Girdileri (Sliderlar)
    for col in X.columns:
        if col == 'power_kWh':
            input_data[col] = st.sidebar.slider("Güç (kWh)", 3000.0, 5000.0, 4000.0)
        elif col == 'oxygen_Nm3':
            input_data[col] = st.sidebar.slider("Oksijen (Nm3)", 100.0, 300.0, 200.0)
        elif col == 'Thermal_Stress_Index':
            input_data[col] = st.sidebar.slider("Termal Stres İndeksi (0-100)", 0.0, 100.0, 50.0)
        elif col == 'Scrap_Quality_Index':
            input_data[col] = st.sidebar.slider("Hurda Kalitesi (0-100)", 0.0, 100.0, 70.0)
        elif col == 'tap_time_min':
            input_data[col] = st.sidebar.slider("Süre (dk)", 40.0, 70.0, 55.0)
        else:
            input_data[col] = df[col].mean()

    # TAHMİN YAP
    input_df = pd.DataFrame([input_data])[X.columns]
    prediction = model.predict(input_df)[0]
    
    # Tonaj (Sabit varsayım veya girdi)
    tonnage = 10.0 # 10 tonluk ocak

    # ------------------------------------------------------------------
    # MODÜL 1: AI BAKIM VE DURUŞ ENGELLEME
    # ------------------------------------------------------------------
    if selected_module == "1️⃣ AI Bakım ve Duruş Engelleme":
        st.title("🛡️ Modül 1: AI Bakım ve Duruş Engelleme")
        st.markdown("Fırın refrakter sağlığı, panel soğutma performansı ve ekipman anormalliklerinin takibi.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Panel Sıcaklık Trendi (Refrakter Aşınma Göstergesi)")
            # Plotly Line Chart
            fig_trend = px.line(trend_df, x="Tarih", y="Panel_Temp_Avg", title="Panel Çıkış Suyu Sıcaklığı (Günlük Ort.)")
            fig_trend.add_hline(y=45, line_dash="dot", annotation_text="Risk Limiti", line_color="red")
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with col2:
            st.subheader("Anlık Sağlık Skorları")
            stress_val = input_data['Thermal_Stress_Index']
            
            # Sağlık Gauge
            fig_health = go.Figure(go.Indicator(
                mode = "gauge+number", value = 100 - stress_val,
                title = {'text': "Panel Sağlık Skoru"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "green" if stress_val < 50 else "red"}}
            ))
            fig_health.update_layout(height=250)
            st.plotly_chart(fig_health, use_container_width=True)
            
            if stress_val > 70:
                st.error("⚠️ DİKKAT: Yüksek Termal Stres! Panel delinme riski.")
                st.info("👉 Öneri: Soğutma suyu debisini %10 artırın.")
            else:
                st.success("✅ Sistem Normal: Bakım ihtiyacı öngörülmüyor.")

    # ------------------------------------------------------------------
    # MODÜL 2: AI GİRDİ MALİYETLERİ DÜŞÜRME
    # ------------------------------------------------------------------
    elif selected_module == "2️⃣ AI Girdi Maliyetleri Düşürme":
        st.title("💰 Modül 2: AI Girdi Maliyetleri Optimizasyonu")
        st.markdown("Hurda reçetesi ve enerji birim maliyetlerinin detaylı analizi.")
        
        # Maliyet Hesabı
        c_scrap = tonnage * price_scrap
        c_elec = input_data['power_kWh'] * price_elec
        c_oxy = input_data['oxygen_Nm3'] * price_oxy
        c_elec_rod = tonnage * 1.8 * price_electrode # 1.8 kg/ton varsayım
        total = c_scrap + c_elec + c_oxy + c_elec_rod
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Maliyet Dağılımı")
            cost_data = pd.DataFrame({
                "Kalem": ["Hurda", "Elektrik", "Oksijen", "Elektrot"],
                "Tutar": [c_scrap, c_elec, c_oxy, c_elec_rod]
            })
            fig_pie = px.pie(cost_data, values='Tutar', names='Kalem', title='Döküm Başına Maliyet Kırılımı', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            st.subheader("Birim Maliyet Analizi ($/ton)")
            unit_cost = total / tonnage
            st.metric("Mevcut Birim Maliyet", f"${unit_cost:.2f} / ton")
            st.metric("Hedef Birim Maliyet", "$450.00 / ton", delta=f"${unit_cost-450:.2f}")
            
            st.markdown("### 💡 Tasarruf Önerisi")
            if input_data['power_kWh'] > 4200:
                st.warning("Enerji tüketimi yüksek. Oksijen miktarını artırarak elektrikten tasarruf edebilirsiniz.")
            else:
                st.success("Enerji tüketimi optimum seviyede.")

    # ------------------------------------------------------------------
    # MODÜL 3: KARAR DESTEK MODÜLÜ
    # ------------------------------------------------------------------
    elif selected_module == "3️⃣ Karar Destek Modülü (Process)":
        st.title("🎯 Modül 3: Karar Destek ve Dijital İkiz")
        
        col_main, col_cfd = st.columns([1, 2])
        
        with col_main:
            st.subheader("Sıcaklık Tahmini")
            st.plotly_chart(create_gauge_chart(prediction), use_container_width=True)
            
            st.info(f"Girdiğiniz parametrelerle beklenen döküm sıcaklığı: **{prediction:.1f} °C**")
            
        with col_cfd:
            st.subheader("Fırın İçi Termal Dağılım (CFD Simülasyonu)")
            X_grid, Y_grid, T_field = generate_cfd_fields(input_data['power_kWh'])
            fig_cfd, ax = plt.subplots(figsize=(6,4))
            c = ax.contourf(X_grid, Y_grid, T_field, levels=20, cmap='inferno')
            fig_cfd.colorbar(c, label='Sıcaklık (°C)')
            ax.set_title(f"Taban Sıcaklık Haritası (Güç: {input_data['power_kWh']} kWh)")
            st.pyplot(fig_cfd)

        st.markdown("### 🔥 Isınma Eğrisi Simülasyonu")
        # Basit bir ısınma eğrisi
        time_steps = np.linspace(0, input_data['tap_time_min'], 20)
        temp_curve = 1500 + (prediction - 1500) * (time_steps / input_data['tap_time_min'])**0.5
        fig_curve = px.line(x=time_steps, y=temp_curve, labels={'x':'Zaman (dk)', 'y':'Sıcaklık (°C)'}, title="Tahmini Isınma Yolu")
        st.plotly_chart(fig_curve, use_container_width=True)

    # ------------------------------------------------------------------
    # MODÜL 4: ALARM, TAVSİYE VE KPI
    # ------------------------------------------------------------------
    elif selected_module == "4️⃣ Alarm, Tavsiye ve KPI'lar":
        st.title("📢 Modül 4: Alarm Merkezi ve KPI Takibi")
        
        # KPI Kartları
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Günlük Döküm Sayısı", "12", "+2")
        k2.metric("Ort. Döküm Süresi", "52 dk", "-3 dk")
        k3.metric("Enerji (kWh/ton)", "410", "+5")
        k4.metric("Aktif Alarm Sayısı", "1", "Normal")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🔔 Canlı Alarm Listesi")
            alarms = pd.DataFrame({
                "Zaman": ["10:45", "11:20", "14:05"],
                "Ekipman": ["Panel 4", "Elektrot 2", "Oksijen Valfi"],
                "Durum": ["Yüksek Sıcaklık", "Kırılma Riski", "Basınç Düşük"],
                "Öncelik": ["Yüksek", "Orta", "Düşük"]
            })
            # Renklendirme fonksiyonu eklenebilir, basit tablo:
            st.dataframe(alarms, use_container_width=True, hide_index=True)
            
        with c2:
            st.subheader("📋 Operatör Tavsiyeleri")
            if prediction > 1650:
                st.warning("⚠️ Sıcaklık çok yüksek. Gücü kesin.")
            elif prediction < 1600:
                st.error("⚠️ Sıcaklık düşük. Enerji verin.")
            else:
                st.success("✅ Proses stabil ilerliyor.")
            
            st.info("💡 İpucu: Bir sonraki şarjda hurda yoğunluğunu artırmak enerji verimini %2 iyileştirebilir.")

    # ------------------------------------------------------------------
    # MODÜL 5: AI ENTERPRISE LEVEL (EBITDA)
    # ------------------------------------------------------------------
    elif selected_module == "5️⃣ AI Enterprise Level (EBITDA)":
        st.title("🏢 Modül 5: Kurumsal İş Zekası (EBITDA)")
        
        # Kurumsal Girdiler (Bu sayfaya özel)
        with st.expander("Kurumsal Hedef Ayarları", expanded=True):
            col_ent1, col_ent2 = st.columns(2)
            sales_price = col_ent1.number_input("Hedef Satış Fiyatı ($/ton)", 500, 2000, 900)
            monthly_tonnage = col_ent2.number_input("Aylık Hedef Tonaj", 1000, 50000, 10000)
            fixed_costs = st.number_input("Aylık Sabit Giderler (Personel+Kira) $", 100000, 2000000, 500000)

        # Hesaplamalar
        unit_var_cost = (tonnage * price_scrap + input_data['power_kWh']*price_elec + input_data['oxygen_Nm3']*price_oxy + tonnage*1.8*price_electrode) / tonnage
        
        total_revenue = sales_price * monthly_tonnage
        total_var_cost = unit_var_cost * monthly_tonnage
        gross_profit = total_revenue - total_var_cost
        ebitda = gross_profit - fixed_costs
        
        # Şelale Grafiği Verisi
        fig_waterfall = go.Figure(go.Waterfall(
            name = "EBITDA", orientation = "v",
            measure = ["relative", "relative", "total", "relative", "total"],
            x = ["Satış Geliri", "Değişken Maliyetler", "Brüt Kar", "Sabit Giderler", "EBITDA"],
            textposition = "outside",
            text = [f"${total_revenue/1000:.0f}k", f"-${total_var_cost/1000:.0f}k", f"${gross_profit/1000:.0f}k", f"-${fixed_costs/1000:.0f}k", f"${ebitda/1000:.0f}k"],
            y = [total_revenue, -total_var_cost, 0, -fixed_costs, 0],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_waterfall.update_layout(title = "Aylık Karlılık Analizi (Waterfall)", showlegend = False)
        
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # Sonuç Kartları
        m1, m2, m3 = st.columns(3)
        m1.metric("Aylık Ciro", f"${total_revenue:,.0f}")
        m2.metric("EBITDA", f"${ebitda:,.0f}", delta_color="normal" if ebitda>0 else "inverse")
        m3.metric("EBITDA Marjı", f"%{(ebitda/total_revenue)*100:.1f}")

        if ebitda < 0:
            st.error("🚨 Şirket Zarar Ediyor! Sabit giderleri düşürün veya satış fiyatını artırın.")

if __name__ == "__main__":
    main()
