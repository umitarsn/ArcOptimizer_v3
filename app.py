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
    Termal Dengesizlik ve Hurda Kalite İndeksini hesaplar, 
    ML modelinin kullanacağı yeni feature'ları oluşturur.
    """
    df = df.copy()
    
    # --- 1. Termal Dengesizlik İndeksi (Mevcut) ---
    required_thermal_cols = ["panel_T_in_C", "panel_T_out_C", "panel_flow_kg_s", "power_kWh"]
    if all(col in df.columns for col in required_thermal_cols):
        cp_kJ = 4.18  
        df['Q_Panel_kW'] = df['panel_flow_kg_s'] * (df['panel_T_out_C'] - df['panel_T_in_C']) * cp_kJ 
        
        # Dengesizlik Simülasyonu: Yüksek Q_Panel ve Güç termal stresi artırır
        df['Thermal_Imbalance_Index'] = (df['Q_Panel_kW'] * 0.1) + (df['power_kWh'] * 0.005) 
        
        # 0-100 aralığına normalize et
        max_val = df['Thermal_Imbalance_Index'].max()
        if max_val > 0:
            df['Thermal_Imbalance_Index'] = (df['Thermal_Imbalance_Index'] / max_val) * 100
        else:
             df['Thermal_Imbalance_Index'] = 50.0 
        
        df = df.drop(columns=['Q_Panel_kW'])

    # --- 2. Hurda Kalite İndeksi (YENİ) ---
    required_scrap_cols = ["scrap_HMS80_20_pct", "scrap_HBI_pct", "scrap_Shredded_pct"]
    if all(col in df.columns for col in required_scrap_cols):
        # Varsayım: HBI yüksek (1.0), Shredded orta (0.7), HMS düşük (0.4) kalite katsayısı
        df['Scrap_Quality_Index'] = (
            df['scrap_HBI_pct'] * 1.0 + 
            df['scrap_Shredded_pct'] * 0.7 + 
            df['scrap_HMS80_20_pct'] * 0.4
        )
        # Hurda yüzdeleri toplamı %100'ü geçmeyeceği için max index de 100*1.0 = 100'dür.
        # Hesaplanan değeri 0-100 arasında tutarız.
        
        # Orijinal hurda yüzdesi kolonlarını modelden kaldırıp, sadece indeksi kullanıyoruz
        df = df.drop(columns=required_scrap_cols, errors='ignore') 
        
    return df

def create_gauge_chart(value, target=1620, min_range=1500, max_range=1750):
    """Sıcaklık için ibreli gösterge (Gauge) oluşturur."""
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

def generate_cfd_fields(power, oxygen, time, foam_level, magnetic_deviation):
    """Manyetik sapmayı simüle eden basit CFD fonksiyonu."""
    nx, ny = 50, 50
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, ny)
    X, Y = np.meshgrid(x, y)
    
    # Sapma merkezini Manyetik Sapma faktörüne göre kaydır
    center_x = 5.0 + (magnetic_deviation * 0.3) 
    center_y = 5.0 - (magnetic_deviation * 0.1) 
    dist_sq = (X - center_x)**2 + (Y - center_y)**2
    
    base_temp = 1500 + (power / 100)
    temp_field = base_temp * np.exp(-dist_sq / (10.0 + magnetic_deviation * 0.5))
    
    return X, Y, temp_field

# ------------------------------------------------------------
# 2. ANA UYGULAMA AKIŞI
# ------------------------------------------------------------
def main():
    st.title("⚡ DC Ark Ocağı - Akıllı Karar Destek Paneli")
    
    # --- VERİ YÜKLEME SEÇENEĞİ (DEMO vs GERÇEK) ---
    st.sidebar.header("📂 Veri Kaynağı")
    data_mode = st.sidebar.radio(
        "Çalışma Modu Seçiniz:",
        options=("Demo Verileri (Otomatik)", "Kendi Dosyamı Yükle (CSV)"),
        index=0 
    )

    df = None
    
    if data_mode == "Demo Verileri (Otomatik)":
        try:
            df = pd.read_csv("data/BG_EAF_panelcooling_demo.csv")
            st.info(f"ℹ️ **Demo Modu:** {len(df)} satırlık simülasyon verisi kullanılıyor.")
        except FileNotFoundError:
            st.error("⚠️ Demo veri dosyası ('data/BG_EAF_panelcooling_demo.csv') bulunamadı! Lütfen önce veri üretim kodunu çalıştırın.")
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

    # ML Modelinde kullanılacak son feature listesi
    # Burası otomatik olarak df'de kalan tüm kolonları (power_kWh, oxygen_Nm3, tap_time_min, Thermal_Imbalance_Index, Scrap_Quality_Index) alır.
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
    # 3. KULLANICI GİRDİLERİ (SİMÜLASYON) - Sidebar
    # --------------------------------------------------------------------------------
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Simülasyon Parametreleri")
    
    # Tonaj Girdisi
    default_tonnage = 10.0 
    tonnage = st.sidebar.number_input(
        "Tahmini Ergitme Tonajı (ton)", 
        min_value=1.0, 
        max_value=100.0, 
        value=default_tonnage, 
        step=1.0
    )
    
    # --- Hurda Kalite Girişi (YENİ) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("♻️ Hurda Kalite Girdisi")
    quality_input_mode = st.sidebar.radio(
        "Kalite Girdi Şekli:",
        options=("⭐ Toplu Kalite İndeksi Gir", "📊 Hurda Karışımını Gir (Hesapla)"),
        index=0
    )
    
    input_data = {}
    
    if quality_input_mode == "⭐ Toplu Kalite İndeksi Gir":
        # Kullanıcı doğrudan indeksi girer
        input_data['Scrap_Quality_Index'] = st.sidebar.slider(
            "Hurda Kalite İndeksi (0-100)", 
            0.0, 100.0, 70.0, 0.1
        )
    else:
        # Kullanıcı hurda yüzdeleri ve kalite faktörlerini girer
        st.sidebar.caption("Her hurda tipi için yüzdesini girin:")
        
        # Hurda Yüzdeleri (Toplam %100 olmalı - kullanıcıya bırakıldı)
        pct_hbi = st.sidebar.slider("HBI Yüzdesi (%)", 0.0, 100.0, 10.0, 0.1)
        pct_shredded = st.sidebar.slider("Shredded Yüzdesi (%)", 0.0, 100.0, 40.0, 0.1)
        pct_hms = st.sidebar.slider("HMS Yüzdesi (%)", 0.0, 100.0, 50.0, 0.1)
        
        # Her hurda tipinin Kalite Faktörü (0-1) - Bu, sistemin varsayımıdır.
        # Kullanıcının görmesi için sabit bir değer verdik, asıl çarpanlar feature_engineering'de tanımlıdır.
        qual_hbi = 1.0; qual_shredded = 0.7; qual_hms = 0.4 
        
        # Kalite İndeksi Hesaplama
        raw_index = (pct_hbi * qual_hbi) + (pct_shredded * qual_shredded) + (pct_hms * qual_hms)
        
        # Sonuç, 0-100 arasında olmalıdır (max %100 HBI = 100*1.0 = 100)
        input_data['Scrap_Quality_Index'] = min(raw_index, 100.0)
        
        st.sidebar.metric("Hesaplanan Kalite İndeksi", f"{input_data['Scrap_Quality_Index']:.1f}")
        
    st.sidebar.markdown("---")
    
    # --- Kalan Parametre Girdileri ---
    for col in X.columns:
        if col not in input_data: # Kalite indeksi zaten girildi/hesaplandı
            min_v = float(df[col].min())
            max_v = float(df[col].max())
            mean_v = float(df[col].mean())
            
            if col == 'power_kWh':
                input_data[col] = st.sidebar.slider("Güç (power_kWh)", min_v, max_v, mean_v)
            elif col == 'oxygen_Nm3':
                input_data[col] = st.sidebar.slider("Oksijen (oxygen_Nm3)", min_v, max_v, mean_v)
            elif col == 'tap_time_min':
                input_data[col] = st.sidebar.slider("Döküm Süresi (tap_time_min)", min_v, max_v, mean_v)
            elif col == 'Thermal_Imbalance_Index':
                input_data[col] = st.sidebar.slider("🔥 Termal Dengesizlik İndeksi (0-100)", 0.0, 100.0, float(df['Thermal_Imbalance_Index'].median()))
            else:
                input_data[col] = st.sidebar.slider(f"{col}", min_v, max_v, mean_v)
            
    # Maliyet Girdileri
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Birim Fiyatlar ($)")
    price_elec = st.sidebar.number_input("Elektrik ($/kWh)", 0.01, 0.50, 0.10)
    price_oxy = st.sidebar.number_input("Oksijen ($/Nm³)", 0.01, 1.00, 0.15)
    price_electrode = st.sidebar.number_input("Elektrot ($/kg)", 1.0, 10.0, 4.5)
    electrode_rate = st.sidebar.number_input("Elektrot Sarfiyatı (kg/ton)", 0.5, 5.0, 1.8)
    
    # --- TAHMİN VE ANALİZ ---
    
    # Giriş data frame'ini oluştururken kolon sırasını ML eğitimindeki X'e göre ayarlamak kritik
    # input_df = pd.DataFrame([input_data])
    input_df = pd.DataFrame([input_data])[X.columns]
    
    prediction = model.predict(input_df)[0]
    
    # --- TABLAR ---
    tab_main, tab_cfd = st.tabs(["📊 Karar Destek Paneli", "🔥 CFD Simülasyonu (Demo)"])

    # --- TAB 1: KARAR DESTEK & MALİYET ---
    with tab_main:
        with st.expander("📈 Model Doğruluk Oranlarını Göster"):
            c1, c2 = st.columns(2)
            c1.metric("Hata Payı (MAE)", f"±{mae:.1f} °C")
            c2.metric("Model Güveni (R²)", f"%{r2*100:.1f}")

        st.markdown("---")

        # 1. Üst Kısım: Gösterge ve Tavsiye
        col_gauge, col_advice = st.columns([2, 2])
        
        with col_gauge:
            st.plotly_chart(create_gauge_chart(prediction), use_container_width=True)
        
        with col_advice:
            st.subheader("🤖 Operatör Asistanı")
            thermal_index = input_data.get('Thermal_Imbalance_Index', 50.0)
            quality_index = input_data.get('Scrap_Quality_Index', 70.0) # Yeni

            
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

            # Termal Dengesizlik Tavsiyesi
            if thermal_index > 75:
                st.error(f"🚨 **Termal Dengesizlik RİSKİ ({thermal_index:.1f} İndeks)**")
                advice_thermal = "AC/DC Akımı düşürülmeli."
            elif thermal_index > 55:
                st.warning(f"🔔 **Termal Dengesizlik UYARISI ({thermal_index:.1f} İndeks)**")
                advice_thermal = "Manyetik karıştırma kontrolü."
            else:
                st.info(f"✨ Termal Denge Stabil ({thermal_index:.1f} İndeks)")
                advice_thermal = "Denge stabil."
                
            # Kalite Tavsiyesi (YENİ)
            if quality_index < 40:
                st.warning(f"📉 **Düşük Kalite ({quality_index:.1f} İndeks)**")
                advice_quality = "Ergitme süresi uzayabilir, oksijen/güç artırımı gerekebilir."
            else:
                advice_quality = "Kalite yeterli."


            st.markdown("---")
            st.write(f"**Özet Tavsiye:** Sıcaklık: *{advice_temp}* | Denge: *{advice_thermal}* | Kalite: *{advice_quality}*")
            
        st.divider()

        # 2. Alt Kısım: Maliyet ve Açıklama
        col_cost, col_feat = st.columns(2)

        with col_cost:
            st.subheader("💵 Maliyet ve Performans Analizi (Tonaj Bazlı)")
            
            pwr = input_data.get('power_kWh', 0)
            oxy = input_data.get('oxygen_Nm3', 0)
            
            cost_e = pwr * price_elec
            cost_o = oxy * price_oxy
            cost_el = tonnage * electrode_rate * price_electrode 
            total_cost = cost_e + cost_o + cost_el

            cost_per_ton = total_cost / tonnage
            kwh_per_ton = pwr / tonnage
            
            target_cost_per_ton = 100.0 
            target_kwh_per_ton = 400.0

            st.dataframe(pd.DataFrame({
                "Kalem": ["Elektrik ($)", "Oksijen ($)", "Elektrot ($)", "TOPLAM MALİYET ($)"],
                "Değer": [f"{cost_e:.2f}", f"{cost_o:.2f}", f"{cost_el:.2f}", f"{total_cost:.2f}"]
            }), hide_index=True, use_container_width=True)
            
            st.markdown("---")
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
            st.subheader("🔍 Model Karar Açıklaması (Feature Importance)")
            
            importances = pd.DataFrame({
                'Faktör': X.columns,
                'Etki': model.feature_importances_
            }).sort_values(by='Etki', ascending=False)
            
            st.bar_chart(importances.set_index('Faktör'), color="#0056b3")
            st.caption("Modelin sıcaklık tahmininde en çok dikkate aldığı parametreler. **Scrap_Quality_Index** hurda kalitesinin, **Thermal_Imbalance_Index** ise stabilite verisidir.")
            
            st.markdown("---")
            st.write("**Çıkarım:**")
            st.write(f"1. En önemli faktör **{importances.iloc[0]['Faktör']}**'dir. Bunun ayarlanması tahmini en çok etkiler.")
            st.write("2. Yeni eklenen indeksler, hurda kalitesi ve fırın stabilitesinin sıcaklık tahminindeki önemini gösterir.")


    # --- TAB 2: CFD GÖRÜNÜMÜ ---
    with tab_cfd:
        st.subheader("Sanal CFD Isı Dağılımı (Manyetik Sapma Simülasyonu)")
        st.info("Soldaki 'Thermal Dengesizlik İndeksi' ayarını değiştirerek sıcaklık dağılımının merkezin dışına kaymasını gözlemleyin.")
        
        # Manyetik Sapma Ayarı (Termal İndeks ile ilişkilendirildi)
        thermal_index_for_cfd = input_data.get('Thermal_Imbalance_Index', 50.0)
        magnetic_deviation = thermal_index_for_cfd / 10.0 # 0-100 Termal İndeks -> 0-10 Sapma
        
        st.write(f"**Manyetik Sapma Faktörü:** {magnetic_deviation:.1f} (Termal Dengesizlik İndeksine göre otomatik ayarlandı.)")

        pwr_cfd = input_data.get('power_kWh', 4000)
        oxy_cfd = input_data.get('oxygen_Nm3', 200)
        
        X_grid, Y_grid, T_field = generate_cfd_fields(pwr_cfd, oxy_cfd, 50, 0.5, magnetic_deviation)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        c = ax.contourf(X_grid, Y_grid, T_field, levels=20, cmap='inferno')
        fig.colorbar(c, label='Sıcaklık (°C)')
        ax.set_title(f"EAF Taban Sıcaklık Dağılımı (Sapma Faktörü: {magnetic_deviation:.1f})")
        ax.set_xlabel("Fırın Genişliği (m)")
        ax.set_ylabel("Fırın Derinliği (m)")
        
        st.pyplot(fig)


if __name__ == "__main__":
    main()
