import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ------------------------------------------------------------
# 1. SAYFA AYARLARI ve YARDIMCI FONKSİYONLAR
# ------------------------------------------------------------
st.set_page_config(
    page_title="BG-ArcOptimizer v2",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded" 
)

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Termal Stres ve Hurda Kalite İndeksini hesaplar, 
    ML modelinin kullanacağı yeni feature'ları oluşturur.
    """
    df = df.copy()
    
    # --- 1. Termal Stres İndeksi (Modül 1 & 4) - Yüksek değerler Manyetik Dengesizlik Riskini Temsil Eder ---
    required_thermal_cols = ["panel_T_in_C", "panel_T_out_C", "panel_flow_kg_s", "power_kWh"]
    if all(col in df.columns for col in required_thermal_cols):
        cp_kJ = 4.18  
        df['Q_Panel_kW'] = df['panel_flow_kg_s'] * (df['panel_T_out_C'] - df['panel_T_in_C']) * cp_kJ 
        
        # Termal Stres Simülasyonu: Yüksek Q_Panel ve Güç, termal stresi artırır. 
        # (Bu, DC manyetik alanının neden olduğu refrakter aşınmasının sonuçlarını simüle edebilir)
        df['Thermal_Stress_Index'] = (df['Q_Panel_kW'] * 0.1) + (df['power_kWh'] * 0.005) 
        
        # 0-100 aralığına normalize et
        max_val = df['Thermal_Stress_Index'].max()
        if max_val > 0:
            df['Thermal_Stress_Index'] = (df['Thermal_Stress_Index'] / max_val) * 100
        else:
             df['Thermal_Stress_Index'] = 50.0 
        
        df = df.drop(columns=['Q_Panel_kW'])

    # --- 2. Hurda Kalite İndeksi (Modül 2) ---
    required_scrap_cols = ["scrap_HMS80_20_pct", "scrap_HBI_pct", "scrap_Shredded_pct"]
    if all(col in df.columns for col in required_scrap_cols):
        # Varsayım: HBI yüksek (1.0), Shredded orta (0.7), HMS düşük (0.4) kalite katsayısı
        df['Scrap_Quality_Index'] = (
            df['scrap_HBI_pct'] * 1.0 + 
            df['scrap_Shredded_pct'] * 0.7 + 
            df['scrap_HMS80_20_pct'] * 0.4
        )
        # Hesaplanan değeri 0-100 arasında tutarız.
        
        # Orijinal hurda yüzdesi kolonlarını modelden kaldırıp, sadece indeksi kullanıyoruz
        df = df.drop(columns=required_scrap_cols, errors='ignore') 
        
    return df

def create_gauge_chart(value, target=1620, min_range=1500, max_range=1750):
    """Sıcaklık için ibreli gösterge (Gauge) oluşturur (Modül 4)."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Tahmini Döküm Sıcaklığı (°C)", 'font': {'size': 20}},
        delta = {'reference': target, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [min_range, max_range], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_range, 1600], 'color': '#4dabf5'},
                {'range': [1600, 1640], 'color': '#66ff66'},
                {'range': [1640, max_range], 'color': '#ff6666'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 1700}}))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    return fig

def generate_cfd_fields(power, magnetic_deviation_factor):
    """
    Arc Ocağı Taban Sıcaklık Alanını Simüle Eder (DC Ark Akımının Manyetik Sapma Etkisi).
    DC akımının yarattığı elektromanyetik dengesizlik, sıvı metal havuzunun ısı merkezini kaydırır.
    """
    nx, ny = 50, 50
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, ny)
    X, Y = np.meshgrid(x, y)
    
    # 1. Ark Merkezini Sapma Faktörüne Göre Kaydırma (Manyetik Etki)
    # 5.0, 5.0 fırının merkezi
    # Sapma Faktörü, 0 (Merkez) ile 3 (Maksimum Köşe Kayması) arasında bir değer alır.
    deviation_amount = magnetic_deviation_factor * 0.8
    center_x = 5.0 + deviation_amount * np.cos(np.pi/4) 
    center_y = 5.0 + deviation_amount * np.sin(np.pi/4)
    dist_sq = (X - center_x)**2 + (Y - center_y)**2
    
    # 2. Dağılım Sabiti (Havuz Hacmi): Güç arttıkça ark daha yaygınlaşır (daha geniş sıvı havuzu).
    diffusion_factor = 10.0 + (power / 500.0) 
    
    # 3. Ark Bölgesi Tepe Sıcaklığı (Güçle orantılı)
    max_arc_temp = 1600 + (power * 0.05) 
    
    # Gauss dağılımı kullanarak sıcaklık alanı oluşturma
    temp_field = max_arc_temp * np.exp(-dist_sq / diffusion_factor)
    # En düşük sıcaklık 1500 C'nin altına düşmesin
    temp_field = np.maximum(temp_field, 1500)
    
    return X, Y, temp_field

# ------------------------------------------------------------
# 2. ANA UYGULAMA AKIŞI
# ------------------------------------------------------------
def main():
    st.title("⚡ DC Ark Ocağı - Akıllı Karar Destek Paneli (Modül 3)")
    
    # --- VERİ YÜKLEME SEÇENEĞİ ---
    st.sidebar.header("📂 Veri Kaynağı")
    data_mode = st.sidebar.radio(
        "Çalışma Modu Seçiniz:",
        options=("Demo Verileri (Otomatik)", "Kendi Dosyamı Yükle (CSV)"),
        index=0 
    )

    df = None
    
    if data_mode == "Demo Verileri (Otomatik)":
        try:
            # NOT: Bu path'in projenizin kök dizinine göre doğru olduğundan emin olun
            df = pd.read_csv("data/BG_EAF_panelcooling_demo.csv") 
            st.info(f"ℹ️ **Demo Modu:** {len(df)} satırlık simülasyon verisi kullanılıyor.")
        except FileNotFoundError:
            st.error("⚠️ Demo veri dosyası bulunamadı. Lütfen kontrol edin.")
            st.stop()
            
    else:
        uploaded_file = st.sidebar.file_uploader("CSV Dosyanızı Sürükleyin", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dosya Yüklendi: {len(df)} satır.")
        else:
            st.warning("👈 Lütfen sol menüden bir CSV dosyası yükleyin veya Demo moduna geçin.")
            st.stop()

    # --- VERİ ÖN İŞLEME ve FEATURE ENGINEERING ---
    df = feature_engineering(df) 
    
    # --- MODEL EĞİTİMİ ---
    target_col = "tap_temperature_C"
    
    if target_col not in df.columns:
        st.error(f"Hata: CSV dosyasında '{target_col}' sütunu bulunamadı.")
        st.stop()

    drop_cols = ["heat_id", "tap_temperature_C", "melt_temperature_C", 
                 "panel_T_in_C", "panel_T_out_C", "panel_flow_kg_s"]
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target_col], errors='ignore')
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Başarım Metrikleri
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # --------------------------------------------------------------------------------
    # 3. KULLANICI GİRDİLERİ (SİMÜLASYON) - Sidebar (Modül 3 & 2)
    # --------------------------------------------------------------------------------
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Proses Simülasyon Parametreleri")
    
    default_tonnage = 10.0 
    tonnage = st.sidebar.number_input(
        "Tahmini Ergitme Tonajı (ton)", 
        min_value=1.0, 
        max_value=100.0, 
        value=default_tonnage, 
        step=1.0
    )
    
    # --- Hurda Kalite Girişi ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("♻️ Hurda Kalite Girdisi")
    quality_input_mode = st.sidebar.radio(
        "Kalite Girdi Şekli:",
        options=("⭐ Toplu Kalite İndeksi Gir", "📊 Hurda Karışımını Gir (Hesapla)"),
        index=0
    )
    
    input_data = {}
    
    if quality_input_mode == "⭐ Toplu Kalite İndeksi Gir":
        input_data['Scrap_Quality_Index'] = st.sidebar.slider(
            "Hurda Kalite İndeksi (0-100)", 
            0.0, 100.0, 70.0, 0.1
        )
    else:
        # Hesaplama mantığı
        pct_hbi = st.sidebar.slider("HBI Yüzdesi (%)", 0.0, 100.0, 10.0, 0.1)
        pct_shredded = st.sidebar.slider("Shredded Yüzdesi (%)", 0.0, 100.0, 40.0, 0.1)
        pct_hms = st.sidebar.slider("HMS Yüzdesi (%)", 0.0, 100.0, 50.0, 0.1)
        
        qual_hbi = 1.0; qual_shredded = 0.7; qual_hms = 0.4 
        raw_index = (pct_hbi * qual_hbi) + (pct_shredded * qual_shredded) + (pct_hms * qual_hms)
        
        input_data['Scrap_Quality_Index'] = min(raw_index, 100.0)
        st.sidebar.metric("Hesaplanan Kalite İndeksi", f"{input_data['Scrap_Quality_Index']:.1f}")
        
    st.sidebar.markdown("---")
    
    # --- Kalan Proses Parametre Girdileri ---
    for col in X.columns:
        if col not in input_data:
            min_v = float(df[col].min())
            max_v = float(df[col].max())
            mean_v = float(df[col].mean())
            
            if col == 'power_kWh':
                input_data[col] = st.sidebar.slider("Güç (power_kWh)", min_v, max_v, mean_v)
            elif col == 'oxygen_Nm3':
                input_data[col] = st.sidebar.slider("Oksijen (oxygen_Nm3)", min_v, max_v, mean_v)
            elif col == 'tap_time_min':
                input_data[col] = st.sidebar.slider("Döküm Süresi (tap_time_min)", min_v, max_v, mean_v)
            elif col == 'Thermal_Stress_Index': 
                input_data[col] = st.sidebar.slider("🔥 Panel Termal Stres İndeksi (0-100) - Manyetik Dengesizlik Riski", 0.0, 100.0, float(df['Thermal_Stress_Index'].median()))
            else:
                input_data[col] = st.sidebar.slider(f"{col}", min_v, max_v, mean_v)
            
    # Maliyet Girdileri (Modül 2)
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Anlık Birim Fiyatlar ($)")
    
    price_scrap_ton = st.sidebar.number_input("Hurda ($/ton)", 100.0, 800.0, 450.0, step=10.0)
    price_electrode = st.sidebar.number_input("Elektrot ($/kg)", 1.0, 10.0, 4.5, step=0.1)
    electrode_rate = st.sidebar.number_input("Elektrot Sarfiyatı (kg/ton)", 0.5, 5.0, 1.8, step=0.1)
    
    price_elec = st.sidebar.number_input("Elektrik ($/kWh)", 0.01, 0.50, 0.10, step=0.01)
    price_oxy = st.sidebar.number_input("Oksijen ($/Nm³)", 0.01, 1.00, 0.15, step=0.01)
    
    # --------------------------------------------------------------------------------
    # 4. KURUMSAL (ENTERPRISE) GİRDİLERİ - Sidebar (Modül 5)
    # --------------------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("🏢 AI Enterprise Level Girdileri")
    st.sidebar.caption("SAP, Satış ve Tahmin Verileri Simülasyonu")
    
    # Satış/Hedef Girdileri (SAP Simülasyonu)
    sales_price_ton = st.sidebar.number_input(
        "Hedef Satış Fiyatı ($/ton)", 
        min_value=500.0, 
        max_value=3000.0, 
        value=1500.0, 
        step=10.0
    )
    monthly_tonnage_target = st.sidebar.number_input(
        "Aylık Üretim Hedefi (ton)", 
        min_value=100.0, 
        max_value=20000.0, 
        value=10000.0, 
        step=100.0
    )

    # Global/Lokal Talep ve Maliyet Tahmini Girdileri
    forecast_elec_price = st.sidebar.number_input(
        "Tahmini Gelecek Elektrik Fiyatı ($/kWh)", 
        0.05, 0.30, 0.12, 0.01
    )
    global_demand_index = st.sidebar.slider(
        "Global Talep İndeksi (0=Düşük, 10=Yüksek)", 
        0.0, 10.0, 7.5, 0.1
    )
    
    # EBITDA için Sabit Maliyet Girdileri (Modül 5)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧮 Aylık Sabit Maliyetler ($)")
    price_labor_monthly = st.sidebar.number_input(
        "Personel/İşçilik Gideri (Aylık $)", 
        min_value=10000.0, 
        max_value=5000000.0, 
        value=500000.0, 
        step=10000.0
    )
    price_sg_a_monthly = st.sidebar.number_input(
        "Genel Yönetim/SG&A (Aylık $)", 
        min_value=10000.0, 
        max_value=2000000.0, 
        value=250000.0, 
        step=5000.0
    )
    
    # --- TAHMİN VE ANALİZ ---
    
    input_df = pd.DataFrame([input_data])[X.columns]
    prediction = model.predict(input_df)[0]
    
    # Proses Maliyeti Hesaplaması (Değişken Maliyetler - Tek Ergitme)
    pwr = input_data.get('power_kWh', 0)
    oxy = input_data.get('oxygen_Nm3', 0)
    
    cost_scrap = tonnage * price_scrap_ton 
    cost_e = pwr * price_elec
    cost_o = oxy * price_oxy
    cost_el = tonnage * electrode_rate * price_electrode 
    
    total_variable_cost_per_heat = cost_scrap + cost_e + cost_o + cost_el 
    
    cost_per_ton = total_variable_cost_per_heat / tonnage
    kwh_per_ton = pwr / tonnage
    
    # Aylık Finansal Hesaplamalar (Modül 5)
    
    # 1. Gelir
    total_sales_revenue = sales_price_ton * monthly_tonnage_target
    
    # 2. Maliyetler
    total_variable_cost_per_month = cost_per_ton * monthly_tonnage_target
    total_fixed_cost_per_month = price_labor_monthly + price_sg_a_monthly
    total_operating_cost = total_variable_cost_per_month + total_fixed_cost_per_month 
    
    # 3. Karlılık
    ebitda = total_sales_revenue - total_operating_cost
    

    # --- TABLAR (Modül 3, 4, 5) ---
    tab_main, tab_cfd, tab_enterprise = st.tabs([
        "📊 Karar Destek Paneli (Modül 3)", 
        "🔥 CFD Simülasyonu (Modül 3)",
        "🏢 AI Enterprise Level (Modül 5)"
    ])


    # --- TAB 1: KARAR DESTEK & MALİYET (Modül 3 & 2) ---
    with tab_main:
        with st.expander("📈 Model Doğruluk Oranlarını Göster"):
            c1, c2 = st.columns(2)
            c1.metric("Hata Payı (MAE)", f"±{mae:.1f} °C")
            c2.metric("Model Güveni (R²)", f"%{r2*100:.1f}")

        st.markdown("---")

        # 1. Üst Kısım: Gösterge ve Tavsiye (Modül 4)
        col_gauge, col_advice = st.columns([2, 2])
        
        with col_gauge:
            st.plotly_chart(create_gauge_chart(prediction), use_container_width=True)
        
        with col_advice:
            st.subheader("🤖 Operatör Asistanı (Modül 4)")
            thermal_index = input_data.get('Thermal_Stress_Index', 50.0) 
            quality_index = input_data.get('Scrap_Quality_Index', 70.0) 

            
            # Ana Sıcaklık Tavsiyesi
            if prediction < 1600:
                st.error(f"⚠️ **Düşük Sıcaklık ({prediction:.1f}°C)**: Enerji girişini artırın.")
                advice_temp = "Enerjiyi artırın."
            elif 1600 <= prediction <= 1640:
                st.success(f"✅ **İdeal Döküm Aralığı ({prediction:.1f}°C)**: Mevcut parametreler optimum.")
                advice_temp = "Müdahale gerekmez."
            else:
                st.warning(f"🔥 **Aşırı Isınma ({prediction:.1f}°C)**: Enerji israfını önlemek için gücü azaltın.")
                advice_temp = "Gücü azaltın."

            # Termal Stres Tavsiyesi (Modül 1 & 4) - Manyetik Denge/Stres Tavsiyesi
            if thermal_index > 75:
                st.error(f"🚨 **Yüksek Termal Stres/Manyetik Dengesizlik RİSKİ ({thermal_index:.1f} İndeks)**")
                advice_thermal = "DC akım kontrolü ve panel soğutma sistemi/refrakter kontrolü. **Bakım Uyarısı!**" 
            elif thermal_index > 55:
                st.warning(f"🔔 **Termal Stres/Dengesizlik UYARISI ({thermal_index:.1f} İndeks)**")
                advice_thermal = "DC akımını ayarlayarak manyetik dengeyi sağlamaya çalışın veya soğutma debisi kontrol edin."
            else:
                st.info(f"✨ Termal Denge Stabil ({thermal_index:.1f} İndeks)")
                advice_thermal = "Denge stabil."
                
            # Kalite Tavsiyesi (Modül 2 & 4)
            if quality_index < 40:
                st.warning(f"📉 **Düşük Kalite ({quality_index:.1f} İndeks)**")
                advice_quality = "Ergitme süresi uzayabilir, oksijen/güç artırımı gerekebilir."
            else:
                advice_quality = "Kalite yeterli."


            st.markdown("---")
            st.write(f"**Özet Tavsiye:** Sıcaklık: *{advice_temp}* | Stres/Denge: *{advice_thermal}* | Kalite: *{advice_quality}*")
            
        st.divider()

        # 2. Alt Kısım: Maliyet ve Açıklama
        col_cost, col_feat = st.columns(2)

        with col_cost:
            st.subheader("💵 Maliyet ve Performans Analizi (Modül 2)")
            
            st.dataframe(pd.DataFrame({
                "Kalem": ["Hurda ($)", "Elektrik ($)", "Oksijen ($)", "Elektrot ($)", "TOPLAM DEĞİŞKEN MALİYET ($)"],
                "Değer": [f"{cost_scrap:.2f}", f"{cost_e:.2f}", f"{cost_o:.2f}", f"{cost_el:.2f}", f"{total_variable_cost_per_heat:.2f}"]
            }), hide_index=True, use_container_width=True)
            
            st.markdown("---")
            target_cost_per_ton = 100.0 
            target_kwh_per_ton = 400.0 
            
            st.metric(
                label="Toplam Birim Maliyet ($/ton)", 
                value=f"{cost_per_ton:.2f} $",
                delta=f"{(cost_per_ton - target_cost_per_ton):.2f} $ (Hedef: {target_cost_per_ton} $)"
            )
            st.metric(
                label="Birim Enerji Tüketimi (kWh/ton)", 
                value=f"{kwh_per_ton:.1f} kWh",
                delta=f"{(kwh_per_ton - target_kwh_per_ton):.1f} kWh (Hedef: {target_kwh_per_ton} kWh)"
            )
            
        with col_feat:
            st.subheader("🔍 Model Karar Açıklaması (Modül 4)")
            
            importances = pd.DataFrame({
                'Faktör': X.columns,
                'Etki': model.feature_importances_
            }).sort_values(by='Etki', ascending=False)
            
            st.bar_chart(importances.set_index('Faktör'), color="#0056b3")
            st.caption("Modelin sıcaklık tahmininde en çok dikkate aldığı parametreler. **Scrap_Quality_Index** ve **Thermal_Stress_Index** yeni eklenen faktörlerdir.")
            
            st.markdown("---")
            st.write("**Çıkarım:**")
            st.write(f"1. En önemli faktör **{importances.iloc[0]['Faktör']}**'dir. Bunun ayarlanması tahmini en çok etkiler.")
            st.write("2. Yeni eklenen indeksler, hurda kalitesi ve fırın stabilitesinin sıcaklık tahminindeki önemini gösterir.")


    # --- TAB 2: CFD GÖRÜNÜMÜ (Modül 3 - Dijital İkiz) ---
    with tab_cfd:
        st.subheader("Sanal CFD Isı Dağılımı (DC Ark Akımı Manyetik Sapma Simülasyonu)")
        st.info("Bu sekme, Dijital İkiz konseptinin bir parçasıdır. DC akımından kaynaklanan **elektromanyetik kuvvetlerin neden olduğu sapmanın** (termal dengesizlik), sıvı metal havuzunun ısı dağılımı üzerindeki etkisini simüle eder. Soldaki **'Panel Termal Stres İndeksi'** ayarını değiştirerek bu sapmanın eriyik havuzunun **şeklini ve yerini** nasıl değiştirdiğini gözlemleyin.")
        
        # Manyetik Sapma Ayarı (Termal Stres İndeksi ile ilişkilendirildi)
        thermal_index_for_cfd = input_data.get('Thermal_Stress_Index', 50.0) 
        # 0-100 Termal İndeks -> 0-3 Sapma Faktörü (Sapma yüksek stresle doğru orantılıdır)
        magnetic_deviation_factor = thermal_index_for_cfd / 33.3 

        st.write(f"**Simüle Edilen DC Manyetik Sapma Etkisi Faktörü:** {magnetic_deviation_factor:.2f} (Merkezden Kayma Oranı)")

        pwr_cfd = input_data.get('power_kWh', 4000)
        
        X_grid, Y_grid, T_field = generate_cfd_fields(pwr_cfd, magnetic_deviation_factor) 
        
        fig, ax = plt.subplots(figsize=(8, 6))
        c = ax.contourf(X_grid, Y_grid, T_field, levels=20, cmap='inferno')
        fig.colorbar(c, label='Sıcaklık (°C)')
        ax.set_title(f"EAF Taban Sıcaklık Dağılımı (Güç: {pwr_cfd:.0f} kWh)")
        ax.set_xlabel("Fırın Genişliği (m)")
        ax.set_ylabel("Fırın Derinliği (m)")
        
        st.pyplot(fig)
        
    # --- TAB 3: AI ENTERPRISE LEVEL (Modül 5) ---
    with tab_enterprise:
        st.subheader("🏢 Kurumsal İş Zekası ve Stratejik Görünüm (Modül 5)")
        
        st.markdown("### 📈 İş Performansı ve Karlılık Metrikleri")
        
        col_m5_1, col_m5_2, col_m5_3 = st.columns(3)
        
        with col_m5_1:
            st.metric("Aylık Brüt Gelir Hedefi (Simüle)", f"{total_sales_revenue:,.0f} $", "Hedef Satış Fiyatı Bazlı")
        with col_m5_2:
            st.metric("Tahmini Aylık Değişken Maliyet", f"{total_variable_cost_per_month:,.0f} $", "AI Proses Maliyet Bazlı")
        with col_m5_3:
            st.metric("Aylık Sabit Operasyonel Maliyetler", f"{total_fixed_cost_per_month:,.0f} $", "Personel & SG&A")
            
        st.markdown("---")
        st.markdown("### 📊 Karlılık Analizi (EBITDA)")
        
        col_m5_4, col_m5_5 = st.columns(2)
        
        with col_m5_4:
            # EBITDA Metriği
            delta_value = f"{ebitda/total_sales_revenue * 100:.1f} % EBITDA Marjı" if total_sales_revenue > 0 else "N/A"
            st.metric(
                label="EBITDA (Faiz, Vergi ve Amortisman Öncesi Kar)",
                value=f"{ebitda:,.0f} $",
                delta=delta_value
            )
            
            if ebitda < 0:
                st.error("🚨 EBITDA NEGATİF: Mevcut proses verimi ve maliyet yapısıyla hedeflere ulaşılamaz.")
            elif ebitda < total_sales_revenue * 0.10: 
                 st.warning("🔔 EBITDA DÜŞÜK: Karlılık marjı artırılmalı. Proses iyileştirmesi gerekiyor.")
            else:
                st.success("✅ EBITDA YETERLİ: Proses, satış hedeflerini destekliyor.")
                
        with col_m5_5:
            st.info("💡 **AI-Destekli Karlılık Açıklaması**")
            st.write(f"EBITDA marjı **%{ebitda/total_sales_revenue * 100:.1f}** olarak hesaplanmıştır.")
            st.write("Bu değer, mevcut **AI tarafından simüle edilen proses verimliliğinin** (kWh/ton) kurumsal hedeflerle uyumunu gösterir.")
            
            # AI Analizi ve Tavsiye (Proses İyileştirmesi)
            if ebitda < 0:
                st.markdown(f"**Tavsiye:** Negatif EBITDA'nın ana sebebi **{monthly_tonnage_target:,.0f} ton** hedefinin, toplam **{total_fixed_cost_per_month:,.0f} $** sabit maliyeti absorbe edememesidir. Ya üretimi artırın, ya sabit giderleri düşürün ya da satış fiyatını yükseltin.")
            elif kwh_per_ton > 420: # Yüksek enerji tüketimi varsayımı
                 st.markdown(f"**Tavsiye:** Marj yeterli olsa da, **Birim Enerji Tüketimi ({kwh_per_ton:.1f} kWh/ton)** yüksektir. Modül 3'te **Güç/Oksijen** ayarlarını optimize ederek değişken maliyetleri düşürün, EBITDA marjını artırın.")
            else:
                st.markdown("**Tavsiye:** Proses ve hedefler uyumlu görünüyor. Pazarlama ve satış stratejilerini desteklemek için **Global Talep İndeksi**'ni düzenli olarak takip edin.")


        st.markdown("---")
        st.markdown("### 📊 Stratejik Girdi Tahmin Raporu")
        
        col_m5_6, col_m5_7 = st.columns(2)
        
        with col_m5_6:
            st.info("💡 **Girdi Tahminleri:** Proses maliyetini etkileyecek gelecekteki fiyat tahminleri.")
            st.dataframe(pd.DataFrame({
                "Kalem": ["Hurda Fiyatı ($/ton)", "Tahmini Gelecek Elektrik Fiyatı ($/kWh)", "Elektrot Fiyatı ($/kg)"],
                "Değer": [f"{price_scrap_ton:.0f} $", f"{forecast_elec_price:.3f} $", f"{price_electrode:.2f} $"]
            }), hide_index=True, use_container_width=True)

        with col_m5_7:
            st.info("🌎 **Pazar ve Talep Analizi:** Üretim planlama ve satış stratejisine etki eden makro faktörler.")
            st.metric("Global Talep İndeksi", f"{global_demand_index:.1f}/10", delta=f"Talep gücü: {'Yüksek' if global_demand_index > 7 else ('Orta' if global_demand_index > 4 else 'Düşük')}")
            st.metric("Hedeflenen Tonaj", f"{monthly_tonnage_target:,.0f} ton", delta="Aylık SAP Hedefi")


if __name__ == "__main__":
    main()
