import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
# Diğer gerekli importlar

# ------------------------------------------------------------
# 1. LOGO VE İKON İŞLEME (BASE64 + MANIFEST İLE GARANTİLİ IOS ÇÖZÜMÜ)
# ------------------------------------------------------------

base64_icon = None
favicon_image = None

try:
    # 1. logo.jpg dosyasını aç
    im = Image.open("logo.jpg")
    favicon_image = im

    # 2. iOS için kare (180x180) versiyonunu oluştur
    img_ios = im.copy()
    width, height = img_ios.size
    
    # Sol kenardan, resmin yüksekliği kadar kare kesme
    img_ios = img_ios.crop((0, 0, height, height)) 
    img_ios = img_ios.resize((180, 180)) 
    
    # 3. Görüntüyü Base64'e dönüştür
    buff = io.BytesIO()
    img_ios.save(buff, format="PNG")
    base64_icon = base64.b64encode(buff.getvalue()).decode("utf-8")
    
    # 4. Streamlit'in HTML'ine ZORLA ENJEKSİYON (st.components.v1.html kullanılır)
    if base64_icon:
        st.components.v1.html(
            f"""
            <link rel="apple-touch-icon" 
            sizes="180x180" 
            href="data:image/png;base64,{base64_icon}">

            <link rel="manifest" href="data:application/json,{{
                %22name%22:%22Ferrokrom%20Optimizer%22,
                %22short_name%22:%22Fero%22,
                %22icons%22:[{{
                    %22src%22:%22data:image/png;base64,{base64_icon}%22,
                    %22sizes%22:%22180x180%22,
                    %22type%22:%22image/png%22
                }}],
                %22start_url%22:%22.%22,
                %22display%22:%22standalone%22
            }}">
            """,
            height=0, 
            width=0
        )

except FileNotFoundError:
    st.warning("⚠️ logo.jpg dosyası bulunamadı. Logo/ikon kullanılamıyor.")
except Exception as e:
    st.warning(f"İkon oluşturma hatası oluştu.")
    base64_icon = None
    favicon_image = None
    
# ------------------------------------------------------------
# 2. SAYFA AYARLARI
# ------------------------------------------------------------

st.set_page_config(
    page_title="Ferrokrom AI Optimizasyon",
    layout="wide",
    page_icon=favicon_image, # PIL Image objesi (favicon için)
    initial_sidebar_state="expanded"
)

# Streamlit Üst Bar Logosu (Yan panelin üstü)
try:
    if favicon_image:
        st.logo(favicon_image, icon_image=favicon_image)
except:
    pass


# ------------------------------------------------------------
# 3. VERİ VE SİMÜLASYON FONKSİYONLARI (Cache'li)
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
def generate_dummy_scrap_data(n_suppliers=4, n_lots=40):
    np.random.seed(42)
    suppliers = [f"Tedarikçi {chr(65 + i)}" for i in range(n_suppliers)]
    scrap_types = ["Krom İçi (High C)", "Paslanmaz Hurda", "Düşük Tenörlü cevher", "Şarj Kromu"]
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
# 4. UYGULAMA ANA AKIŞI
# ------------------------------------------------------------
def main():
    # --- SOL MENÜ BAŞLIĞI VE LOGO ---
    if favicon_image:
        st.sidebar.image(favicon_image, use_container_width=True)
    else:
        st.sidebar.header("Ferrokrom AI")
    
    st.sidebar.markdown("**Akıllı Karar Destek Sistemi**")
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
        if selected_module != "4️⃣ Alarm, Tavsiye ve KPI'lar": 
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

    # --- MODÜL İÇERİKLERİ ---
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
            c = ax.contourf(X, Y, T, levels=25, cmap='inferno')
            ax.quiver(X[::4, ::4], Y[::4, ::4], Vx[::4, ::4], Vy[::4, ::4], color='white', alpha=0.6)
            fig_cfd.colorbar(c, label='Sıcaklık (°C)')
            ax.set_title(f"Havuz ve Akış (Güç: {input_data['power_kWh']} kWh)")
            st.pyplot(fig_cfd)

    elif selected_module == "4️⃣ Alarm, Tavsiye ve KPI'lar":
        st.title("🚨 Modül 4: Alarm Merkezi ve KPI")
        k1, k2, k3 = st.columns(3)
        k1.metric("Ark Stabilite Skoru", f"{arc_stability_factor*100:.1f}")
        k2.metric("Döküm Süresi", f"{input_data.get('tap_time_min', 0):.1f} dk")
        alarm = "YOK" if arc_deviation_pct < 20 else "VAR"
        k3.metric("Aktif Alarm", alarm, delta_color="inverse" if alarm=="VAR" else "normal")
        st.markdown("---")
        st.subheader("Stabilite Geçmişi")
        fig_stab = px.area(trend_df, x="Tarih", y="Arc_Stability_KPI", title="Ark Stabilizasyon Performansı")
        st.plotly_chart(fig_stab, use_container_width=True)

    elif selected_module == "5️⃣ AI Enterprise Level (EBITDA)":
        st.title("🏢 Modül 5: Kurumsal İş Zekası (EBITDA)")
        with st.expander("Finansal Hedef Ayarları", expanded=True):
            col_e1, col_e2 = st.columns(2)
            sales_price = col_e1.number_input("Hedef Satış Fiyatı ($/ton)", 500, 3000, 1500)
            monthly_target = col_e2.number_input("Aylık Hedef Tonaj", 1000, 50000, 10000)
            fixed_cost = st.number_input("Aylık Sabit Giderler ($)", 100000, 2000000, 500000)
        
        cost_elec = (input_data['power_kWh'] * (price_elec / 1000.0))
        cost_oxy = input_data['oxygen_Nm3'] * price_oxy
        cost_scrap = tonnage * price_scrap
        cost_electrode = tonnage * 1.8 * price_electrode
        unit_var_cost = (cost_scrap + cost_elec + cost_oxy + cost_electrode) / tonnage
        
        revenue = sales_price * monthly_target
        var_cost_total = unit_var_cost * monthly_target
        gross = revenue - var_cost_total
        ebitda = gross - fixed_cost
        
        fig_water = go.Figure(go.Waterfall(
            name="EBITDA", orientation="v",
            measure=["relative", "relative", "total", "relative", "total"],
            x=["Ciro", "Değişken Mal.", "Brüt Kar", "Sabit Gider", "EBITDA"],
            y=[revenue, -var_cost_total, 0, -fixed_cost, 0],
            text=[f"${revenue/1e6:.1f}M", f"-${var_cost_total/1e6:.1f}M", f"${gross/1e6:.1f}M", f"-${fixed_cost/1e6:.1f}M", f"${ebitda/1e6:.1f}M"],
            connector={"line":{"color":"rgb(63, 63, 63)"}},
        ))
        st.plotly_chart(fig_water, use_container_width=True)
        st.metric("EBITDA Marjı", f"%{(ebitda/revenue)*100:.1f}")

    elif selected_module == "6️⃣ Scrap & Purchase Intelligence":
        st.title("🧠 Modül 6: Hurda ve Satınalma Zekası")
        uploaded_scrap = st.file_uploader("Hurda Verisi (CSV)", type=["csv"])
        scrap_df = pd.read_csv(uploaded_scrap) if uploaded_scrap else generate_dummy_scrap_data()
        with st.expander("Veri Önizleme"): st.dataframe(scrap_df.head(), use_container_width=True)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_scatter = px.scatter(scrap_df, x="Price_USD_t", y="Quality_Index", color="Supplier", size="Lot_tonnage", title="Tedarikçi Fiyat/Kalite Matrisi", hover_data=["Scrap_Type"])
            st.plotly_chart(fig_scatter, use_container_width=True)
        with col_s2:
            scrap_df["Energy_Cost"] = scrap_df["kWh_per_t"] * (price_elec / 1000.0)
            scrap_df["True_Cost"] = scrap_df["Price_USD_t"] + scrap_df["Energy_Cost"]
            fig_bar = px.bar(scrap_df.groupby("Supplier")[["Price_USD_t", "True_Cost"]].mean().reset_index(), x="Supplier", y=["Price_USD_t", "True_Cost"], barmode="group", title="Nominal Fiyat vs Gerçek Maliyet")
            st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__":
    main()
