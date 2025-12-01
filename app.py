import os
import json
import random
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import altair as alt

# ------------------------------------------------
# GENEL AYARLAR
# ------------------------------------------------
st.set_page_config(
    page_title="FeCr AI",
    page_icon="apple-touch-icon.png",
    layout="wide",
)

SETUP_SAVE_PATH = "data/saved_inputs.json"
RUNTIME_SAVE_PATH = "data/runtime_data.json"

os.makedirs("data", exist_ok=True)

# ------------------------------------------------
# SESSION STATE
# ------------------------------------------------
if "info_state" not in st.session_state:
    st.session_state.info_state = {}  # Setup form info

if "profit_info_state" not in st.session_state:
    st.session_state.profit_info_state = {}  # Kazanç analizi info

# ------------------------------------------------
# KAYITLI SETUP VERİLERİ
# ------------------------------------------------
if os.path.exists(SETUP_SAVE_PATH):
    with open(SETUP_SAVE_PATH, "r", encoding="utf-8") as f:
        saved_inputs = json.load(f)
else:
    saved_inputs = {}


# ------------------------------------------------
# RUNTIME VERİLERİ YÜKLE / KAYDET
# ------------------------------------------------
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

# ------------------------------------------------
# SİMÜLASYON VERİ ÜRETİCİSİ
# ------------------------------------------------
def generate_simulation_runtime_data(n=15):
    """Simülasyon Modu için örnek şarj datası üretir."""
    sim_list = []
    now = datetime.now()

    for i in range(n):
        ts = now - timedelta(hours=(n - i))
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
            }
        )

    return sim_list


# ------------------------------------------------
# EXCEL OKUMA (SETUP SAYFASI İÇİN)
# ------------------------------------------------
@st.cache_data
def load_sheets():
    file_name = "dc_saf_soru_tablosu.xlsx"
    try:
        xls = pd.read_excel(file_name, sheet_name=None)
        return {k: v.dropna(how="all") for k, v in xls.items() if not v.empty}
    except Exception as e:
        st.error(f"Excel dosyası yüklenemedi: {e}")
        return {}


# ------------------------------------------------
# 1) SETUP SAYFASI – SABİT GİRDİLER
# ------------------------------------------------
def show_setup_form():
    st.markdown("## 1. Setup – Sabit Proses / Tasarım Verileri")
    st.markdown(
        "Bu form **dc_saf_soru_tablosu.xlsx** dosyasına göre hazırlanmıştır.\n\n"
        "1. Girişi sadece **Set Değeri** alanına yapınız.\n"
        "2. 🔴 Zorunlu (Önem: 1), 🟡 Faydalı (Önem: 2), ⚪ Opsiyonel (Önem: 3) olarak belirtilmiştir.\n"
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
            # Kolon isimlerini temizle ve "set" geçen kolonu bul
            df.columns = [str(c).strip() for c in df.columns]
            unit_cols = [c for c in df.columns if "set" in str(c).lower()]
            unit_col_name = unit_cols[0] if unit_cols else None

            for idx, row in df.iterrows():
                row_key = f"{sheet_idx}_{idx}"

                önem_deger = row.get("Önem", 3)
                try:
                    önem = int(önem_deger)
                except Exception:
                    önem = 3

                renk = {1: "🔴", 2: "🟡", 3: "⚪"}.get(önem, "⚪")

                # Dinamik birim kolonu
                raw_birim = row.get(unit_col_name, "") if unit_col_name else ""
                try:
                    birim = str(raw_birim).strip()
                    if birim.lower() in ["", "none", "nan"]:
                        birim = ""
                except Exception:
                    birim = ""

                tag = row.get("Tag", "")
                val_key = f"{sheet_name}|{tag}"

                cols = st.columns([2.2, 2.5, 4.0, 2.5, 0.7])
                cols[0].markdown(f"**{tag}**")
                cols[1].markdown(f"{renk} {row.get('Değişken', '')}")
                cols[2].markdown(str(row.get("Açıklama", "")))

                current_val = saved_inputs.get(val_key, "")

                with cols[3]:
                    input_col, unit_col = st.columns([5, 2])
                    with input_col:
                        new_val = st.text_input(
                            label="",
                            value=current_val,
                            key=val_key,
                            label_visibility="collapsed",
                            placeholder=""
                        )
                        if new_val != current_val:
                            saved_inputs[val_key] = new_val
                            with open(SETUP_SAVE_PATH, "w", encoding="utf-8") as f:
                                json.dump(saved_inputs, f, ensure_ascii=False, indent=2)

                    with unit_col:
                        unit_text = f"**{birim}**" if birim else ""
                        st.markdown(unit_text)

                with cols[4]:
                    btn_key = f"info_btn_{row_key}"
                    state_key = f"info_state_{row_key}"
                    if st.button("ℹ️", key=btn_key):
                        st.session_state.info_state[state_key] = not st.session_state.info_state.get(
                            state_key, False
                        )

                if st.session_state.info_state.get(f"info_state_{row_key}", False):
                    detaylar = []

                    detay_aciklama = row.get("Detaylı Açıklama")
                    if isinstance(detay_aciklama, str) and detay_aciklama.strip():
                        detaylar.append("🔷 **Detaylı Açıklama:** " + detay_aciklama)

                    veri_kaynagi = row.get("Veri Kaynağı")
                    if isinstance(veri_kaynagi, str) and veri_kaynagi.strip():
                        detaylar.append("📌 **Kaynak:** " + veri_kaynagi)

                    kayit_araligi = row.get("Kayıt Aralığı")
                    if isinstance(kayit_araligi, str) and kayit_araligi.strip():
                        detaylar.append("⏱️ **Kayıt Aralığı:** " + kayit_araligi)

                    onem_text = row.get("Önem")
                    if pd.notna(onem_text):
                        try:
                            onem_int = int(onem_text)
                            detaylar.append("🔵 **Önem:** " + str(onem_int))
                        except Exception:
                            pass

                    if detaylar:
                        st.info("\n".join(detaylar))

                total_fields += 1
                kayit_degeri = str(saved_inputs.get(val_key, "")).strip()
                if kayit_degeri:
                    total_filled += 1
                    if önem == 1:
                        required_filled += 1
                if önem == 1:
                    required_fields += 1

    # Sidebar özet (setup için)
    st.sidebar.subheader("📊 Setup Veri Giriş Durumu")

    if total_fields > 0:
        pct_all = round(100 * total_filled / total_fields, 1)
    else:
        pct_all = 0.0

    if required_fields > 0:
        pct_required = round(100 * required_filled / required_fields, 1)
    else:
        pct_required = 0.0

    st.sidebar.metric("Toplam Giriş Oranı", f"{pct_all}%")
    st.sidebar.progress(min(pct_all / 100, 1.0))

    st.sidebar.metric("Zorunlu Veri Girişi", f"{pct_required}%")
    st.sidebar.progress(min(pct_required / 100, 1.0))

    eksik_zorunlu = required_fields - required_filled
    if eksik_zorunlu > 0:
        st.sidebar.warning(f"❗ Eksik Zorunlu Değerler: {eksik_zorunlu}")


# ------------------------------------------------
# 2) CANLI VERİ SAYFASI – ŞARJ BAZLI ANLIK VERİ
# ------------------------------------------------
def show_runtime_page(sim_mode: bool):
    st.markdown("## 2. Canlı Veri – Şarj Bazlı Anlık Veriler")
    if sim_mode:
        st.info(
            "🧪 **Simülasyon Modu Aktif.** Aşağıda gösterilen veriler gerçek zamanlı veri yerine "
            "simülasyon amaçlı oluşturulmuştur. Bu modda girilen yeni veriler dosyaya kaydedilmez."
        )

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
            tap_temp = st.number_input(
                "Tap Sıcaklığı (°C)", min_value=0.0, max_value=2000.0, step=1.0
            )
        with c6:
            o2_flow = st.number_input("Ortalama O2 Debisi (Nm³/h)", min_value=0.0, step=1.0)

        c7, c8, c9 = st.columns(3)
        with c7:
            slag_foaming = st.slider("Slag Foaming Seviyesi (0–10)", 0, 10, 5)
        with c8:
            panel_delta_t = st.number_input("Panel ΔT (°C)", min_value=0.0, step=0.1)
        with c9:
            electrode_cons = st.number_input(
                "Elektrot Tüketimi (kg/şarj)", min_value=0.0, step=0.01
            )

        note = st.text_input("Operatör Notu (opsiyonel)", "")

        submitted = st.form_submit_button("Kaydet")

    if submitted:
        if not heat_id:
            st.error("Heat ID / Şarj No girilmesi zorunludur.")
        else:
            if sim_mode:
                st.warning(
                    "Simülasyon Modu aktifken yeni veri kalıcı olarak kaydedilmez. "
                    "Bu giriş sadece test içindir."
                )
            else:
                now = datetime.now().isoformat()
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

    # Gösterilecek veri kaynağı
    if sim_mode:
        data_source = generate_simulation_runtime_data()
    else:
        data_source = runtime_data

    if not data_source:
        st.info("Henüz canlı veri girilmemiş.")
        return

    df = pd.DataFrame(data_source)
    try:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp_dt")
    except Exception:
        df["timestamp_dt"] = df["timestamp"]

    st.markdown("### Kayıtlı Canlı Veriler")
    st.dataframe(
        df[
            [
                "timestamp_dt",
                "heat_id",
                "tap_weight_t",
                "duration_min",
                "energy_kwh",
                "kwh_per_t",
                "tap_temp_c",
                "electrode_kg_per_heat",
                "slag_foaming_index",
                "panel_delta_t_c",
            ]
        ].rename(
            columns={
                "timestamp_dt": "Zaman",
                "heat_id": "Heat ID",
                "tap_weight_t": "Tap Weight (t)",
                "duration_min": "Süre (dk)",
                "energy_kwh": "Enerji (kWh)",
                "kwh_per_t": "kWh/t",
                "tap_temp_c": "Tap T (°C)",
                "electrode_kg_per_heat": "Elektrot (kg/şarj)",
                "slag_foaming_index": "Slag Foaming",
                "panel_delta_t_c": "Panel ΔT (°C)",
            }
        ),
        use_container_width=True,
    )

    st.markdown("### Basit Trendler (Canlı Veri)")
    chart_df = df.set_index("timestamp_dt")[["kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"]]
    st.line_chart(chart_df)


# ------------------------------------------------
# 3) ARC OPTIMIZER SAYFASI – MODEL OUTPUT & INSIGHTS
# ------------------------------------------------
def show_arc_optimizer_page(sim_mode: bool):
    st.markdown("## 3. Arc Optimizer – Trendler, KPI ve Öneriler")
    if sim_mode:
        st.info(
            "🧪 **Simülasyon Modu Aktif.** Arc Optimizer çıktıları simüle edilen veri üzerinden hesaplanmaktadır."
        )
    else:
        st.markdown(
            "Bu sayfa, canlı veriler üzerinden **enerji verimliliği**, "
            "**elektrot tüketimi** ve **proses stabilitesi** ile ilgili özet KPI ve "
            "modelin önerilerini gösterir."
        )

    # Veri kaynağı seçimi
    if sim_mode:
        data_source = generate_simulation_runtime_data()
    else:
        data_source = runtime_data

    if not data_source:
        st.info("Arc Optimizer çıktıları için henüz canlı veri yok. Önce 2. sayfadan veri ekleyin.")
        return

    df = pd.DataFrame(data_source)
    try:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp_dt")
    except Exception:
        df["timestamp_dt"] = df["timestamp"]

    # Son şarj ve bazı istatistikler
    last = df.iloc[-1]
    last_n = df.tail(10)

    avg_kwh_t = last_n["kwh_per_t"].dropna().mean()
    avg_electrode = last_n["electrode_kg_per_heat"].dropna().mean()
    avg_tap_temp = last_n["tap_temp_c"].dropna().mean()
    avg_panel_dt = last_n["panel_delta_t_c"].dropna().mean()
    avg_slag = last_n["slag_foaming_index"].dropna().mean()

    # Basit iyileşme potansiyeli (placeholder)
    if len(df) >= 10 and df["kwh_per_t"].notna().sum() >= 10:
        first5 = df["kwh_per_t"].dropna().head(5).mean()
        last5 = df["kwh_per_t"].dropna().tail(5).mean()
        saving_potential = max(0.0, first5 - last5)
    else:
        saving_potential = 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Son Şarj kWh/t",
        f"{last['kwh_per_t']:.1f}" if pd.notna(last.get("kwh_per_t", None)) else "-",
    )
    col2.metric(
        "Son Şarj Elektrot",
        f"{last['electrode_kg_per_heat']:.2f} kg/şarj"
        if pd.notna(last.get("electrode_kg_per_heat", None))
        else "-",
    )
    col3.metric(
        "Son Tap Sıcaklığı",
        f"{last['tap_temp_c']:.0f} °C" if pd.notna(last.get("tap_temp_c", None)) else "-",
    )
    col4.metric(
        "Son 10 Şarj Ort. kWh/t",
        f"{avg_kwh_t:.1f}" if avg_kwh_t and not pd.isna(avg_kwh_t) else "-",
    )

    # ---- Trend + Tahmini Döküm Anı (AI) ----
    trend_df = df.set_index("timestamp_dt")[["kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"]]

    min_time = df["timestamp_dt"].min()
    last_time = df["timestamp_dt"].max()
    real_span = last_time - min_time
    if real_span.total_seconds() <= 0:
        real_span = timedelta(minutes=60)

    # Gelecek kısmı: last_time'dan sonra 40% ek pencere
    future_span = real_span * 0.4
    predicted_tap_time = last_time + future_span * 0.7  # pencerenin ~%90'ına denk gelsin
    future_end = last_time + future_span

    # Ortalama/son veriye göre güvenli temel değerler
    def _safe_base(val_avg, val_last, default):
        if val_avg is not None and not pd.isna(val_avg):
            return float(val_avg)
        if val_last is not None and not pd.isna(val_last):
            return float(val_last)
        return default

    base_tap_temp = _safe_base(avg_tap_temp, last.get("tap_temp_c", None), 1610.0)
    base_kwh_t = _safe_base(avg_kwh_t, last.get("kwh_per_t", None), 420.0)
    base_electrode = _safe_base(
        avg_electrode, last.get("electrode_kg_per_heat", None), 2.0
    )

    # AI yaklaşımı: asla Aktüel’den kötü değil – teorik optimuma doğru adım
    # Tap sıcaklığı için ~3 °C düşüş (enerji tasarrufu + kalite korunumu)
    real_tap = float(last.get("tap_temp_c", base_tap_temp))
    ai_tap_target = max(real_tap - 3.0, 1580.0)  # gereksiz düşük olmasın

    # Enerji için: yaklaşık 0.4 kWh/t iyileşme hedefi
    real_kwh_t = float(last.get("kwh_per_t", base_kwh_t))
    ai_kwh_target = max(real_kwh_t - 0.4, 0.0)

    # Elektrot için: hafif iyileşme (0.003 kg/t) – ama asla daha kötü değil
    tap_w = float(last.get("tap_weight_t", 0.0) or 0.0)
    if tap_w > 0:
        real_elec_pt = float(last.get("electrode_kg_per_heat", base_electrode)) / tap_w
        ai_elec_target_pt = max(real_elec_pt - 0.003, 0.0)
        ai_elec_target_heat = ai_elec_target_pt * tap_w
    else:
        real_elec_pt = None
        ai_elec_target_pt = None
        ai_elec_target_heat = base_electrode

    # Gelecekte 3 nokta oluşturalım (son nokta = tahmini döküm anı)
    future_points = []
    last_kwh = real_kwh_t
    last_tap_temp = real_tap
    last_electrode = float(last.get("electrode_kg_per_heat", base_electrode))

    for i in range(1, 4):
        frac = i / 3.0
        t = last_time + future_span * frac

        kwh_val = last_kwh + (ai_kwh_target - last_kwh) * frac
        tap_val = last_tap_temp + (ai_tap_target - last_tap_temp) * frac
        el_heat_val = last_electrode + (ai_elec_target_heat - last_electrode) * frac

        future_points.append(
            {
                "timestamp_dt": t,
                "kwh_per_t": kwh_val,
                "tap_temp_c": tap_val,
                "electrode_kg_per_heat": el_heat_val,
            }
        )

    future_df = pd.DataFrame(future_points)

    # Long form – gerçek data
    actual_long = (
        trend_df.reset_index()
        .melt(
            id_vars=["timestamp_dt"],
            value_vars=["kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"],
            var_name="variable",
            value_name="value",
        )
    )
    actual_long["data_type"] = "Aktüel"

    # Long form – tahmin data
    future_long = (
        future_df.melt(
            id_vars=["timestamp_dt"],
            value_vars=["kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"],
            var_name="variable",
            value_name="value",
        )
    )
    future_long["data_type"] = "Potansiyel (AI)"

    combined = pd.concat([actual_long, future_long], ignore_index=True)

    variable_name_map = {
        "kwh_per_t": "kWh/t",
        "tap_temp_c": "Tap T (°C)",
        "electrode_kg_per_heat": "Elektrot (kg/şarj)",
    }
    combined["variable_name"] = combined["variable"].map(variable_name_map)

    st.markdown("### Proses Gidişatı – Zaman Trendi ve Tahmini Döküm Anı (AI)")

    base_chart = (
        alt.Chart(combined)
        .mark_line()
        .encode(
            x=alt.X(
                "timestamp_dt:T",
                title="Zaman",
                scale=alt.Scale(domain=[min_time, future_end]),
            ),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("variable_name:N", title="Değişken"),
            strokeDash=alt.StrokeDash(
                "data_type:N",
                title="Veri Tipi",
                scale=alt.Scale(
                    domain=["Aktüel", "Potansiyel (AI)"],
                    range=[[1, 0], [6, 4]],
                ),
            ),
            tooltip=[
                alt.Tooltip("timestamp_dt:T", title="Zaman"),
                alt.Tooltip("variable_name:N", title="Değişken"),
                alt.Tooltip("value:Q", title="Değer", format=".2f"),
                alt.Tooltip("data_type:N", title="Tip"),
            ],
        )
        .properties(height=320)
    )

    # Tahmini döküm sıcaklığı noktası (Tap T, potansiyel son nokta)
    tap_point_df = future_long[
        (future_long["variable"] == "tap_temp_c")
        & (future_long["timestamp_dt"] == future_df["timestamp_dt"].iloc[-1])
    ].copy()
    tap_point_df["variable_name"] = "Tap T (°C)"

    point_chart = (
        alt.Chart(tap_point_df)
        .mark_point(size=90, filled=True)
        .encode(
            x="timestamp_dt:T",
            y="value:Q",
            color=alt.Color("variable_name:N", legend=None),
            tooltip=[
                alt.Tooltip("timestamp_dt:T", title="Hedef Döküm Zamanı (AI)"),
                alt.Tooltip("value:Q", title="Hedef Tap T (°C)", format=".1f"),
            ],
        )
    )

    # "Şimdi" dikey çizgisi: son ölçüm zamanı
    now_df = pd.DataFrame({"timestamp_dt": [last_time]})
    now_rule = (
        alt.Chart(now_df)
        .mark_rule(strokeDash=[2, 2])
        .encode(
            x="timestamp_dt:T",
            tooltip=[alt.Tooltip("timestamp_dt:T", title="Son Ölçüm / Şimdi")],
        )
    )

    st.altair_chart((base_chart + point_chart + now_rule).interactive(), use_container_width=True)

    delta_min = (future_df["timestamp_dt"].iloc[-1] - last_time).total_seconds() / 60.0
    st.markdown(
        f"**Tahmini Döküm Anı (AI):** "
        f"{future_df['timestamp_dt'].iloc[-1].strftime('%Y-%m-%d %H:%M')} "
        f"(yaklaşık {delta_min:.0f} dk sonra)"
    )

    # ------------------------------------------------
    # Proses Kazanç Analizi (Ton Başına)
    # ------------------------------------------------
    st.markdown("### 💰 Proses Kazanç Analizi (Ton Başına)")

    ELECTRICITY_PRICE_EUR_PER_MWH = 50.0  # 50 €/MWh  => 0.05 €/kWh
    ELECTRODE_PRICE_EUR_PER_KG = 3.0      # örnek elektrot fiyatı

    profit_rows = []

    # 1) Enerji tüketimi (kWh/t)
    if pd.notna(last.get("kwh_per_t", None)):
        real_kwh_t = float(last["kwh_per_t"])
        target_kwh_t = ai_kwh_target  # yukarıda hesaplanan AI hedefi
        diff_kwh_t = real_kwh_t - target_kwh_t  # pozitif = iyileştirme alanı

        if diff_kwh_t > 0:
            gain_eur_t = diff_kwh_t * (ELECTRICITY_PRICE_EUR_PER_MWH / 1000.0)
        else:
            diff_kwh_t = 0.0
            gain_eur_t = 0.0
            target_kwh_t = real_kwh_t

        profit_rows.append(
            {
                "tag": "kwh_per_t",
                "deg": "Enerji tüketimi",
                "akt": f"{real_kwh_t:.1f} kWh/t",
                "pot": f"{target_kwh_t:.1f} kWh/t",
                "fark": f"{diff_kwh_t:+.1f} kWh/t",
                "kazanc_text": f"{gain_eur_t:.2f} €/t",
                "kazanc_num": gain_eur_t,
                "type": "cost",
            }
        )

    # 2) Elektrot tüketimi (kg/t) – AI asla Aktüel’den kötü değil
    if tap_w > 0 and real_elec_pt is not None:
        # Eğer gerçek zaten hedeften iyi ise kazanç = 0
        if real_elec_pt > ai_elec_target_pt:
            diff_elec_pt = real_elec_pt - ai_elec_target_pt
            gain_elec_eur_t = diff_elec_pt * ELECTRODE_PRICE_EUR_PER_KG
            target_elec_pt = ai_elec_target_pt
        else:
            diff_elec_pt = 0.0
            gain_elec_eur_t = 0.0
            target_elec_pt = real_elec_pt

        profit_rows.append(
            {
                "tag": "electrode",
                "deg": "Elektrot tüketimi",
                "akt": f"{real_elec_pt:.3f} kg/t",
                "pot": f"{target_elec_pt:.3f} kg/t",
                "fark": f"{diff_elec_pt:+.3f} kg/t",
                "kazanc_text": f"{gain_elec_eur_t:.2f} €/t",
                "kazanc_num": gain_elec_eur_t,
                "type": "cost",
            }
        )

    # 3) Tap sıcaklığı – doğrudan €/t hesaplamak yerine enerji + kalite etkisi
    if pd.notna(real_tap):
        diff_tap = real_tap - ai_tap_target  # pozitif => hedefe göre daha sıcak
        # 3 °C düşüş ≈ 0.5–1.0 kWh/t tasarruf; burada aralık veriyoruz
        if diff_tap > 0.0:
            approx_min = 0.03
            approx_max = 0.10
            kazanc_text = f"{approx_min:.2f}–{approx_max:.2f} €/t + Kalite ↑"
        else:
            diff_tap = 0.0
            kazanc_text = "Kalite korunur"

        profit_rows.append(
            {
                "tag": "tap_temp_c",
                "deg": "Tap sıcaklığı optimizasyonu",
                "akt": f"{real_tap:.0f} °C",
                "pot": f"{ai_tap_target:.0f} °C",
                "fark": f"{diff_tap:+.0f} °C",
                "kazanc_text": kazanc_text,
                "kazanc_num": None,
                "type": "quality",
            }
        )

    # 4) Panel ΔT – kalite & stabilite göstergesi (maliyet dolaylı)
    if pd.notna(last.get("panel_delta_t_c", None)):
        real_panel_dt = float(last["panel_delta_t_c"])
        # Hedef aralık 18–22 °C civarı, ortalama 20
        if real_panel_dt < 18.0:
            target_panel_dt = 18.0
        elif real_panel_dt > 22.0:
            target_panel_dt = 22.0
        else:
            target_panel_dt = real_panel_dt

        diff_panel = target_panel_dt - real_panel_dt
        if abs(diff_panel) < 0.1:
            fark_text = "≈0.0 °C"
        else:
            fark_text = f"{diff_panel:+.1f} °C"

        profit_rows.append(
            {
                "tag": "panel_delta_t",
                "deg": "Panel ΔT",
                "akt": f"{real_panel_dt:.1f} °C",
                "pot": f"{target_panel_dt:.1f} °C",
                "fark": fark_text,
                "kazanc_text": "Kalite ↑",
                "kazanc_num": None,
                "type": "quality",
            }
        )

    # 5) Slag foaming – enerji + refrakter + kalite (dolaylı)
    if pd.notna(last.get("slag_foaming_index", None)):
        real_slag = float(last["slag_foaming_index"])
        # Hedef 7–8 bandı
        if real_slag < 7.0:
            target_slag = 7.0
        elif real_slag > 8.5:
            target_slag = 8.0
        else:
            target_slag = real_slag

        diff_slag = target_slag - real_slag
        if abs(diff_slag) < 0.1:
            fark_slag = "≈0.0"
        else:
            fark_slag = f"{diff_slag:+.1f}"

        profit_rows.append(
            {
                "tag": "slag_foaming",
                "deg": "Köpük seviyesi (slag foaming)",
                "akt": f"{real_slag:.1f}",
                "pot": f"{target_slag:.1f}",
                "fark": fark_slag,
                "kazanc_text": "Kalite ↑ + Verim ↑",
                "kazanc_num": None,
                "type": "quality",
            }
        )

    # DataFrame görünümü
    if profit_rows:
        df_profit = pd.DataFrame(
            [
                {
                    "Tag": r["tag"],
                    "Değişken": r["deg"],
                    "Aktüel": r["akt"],
                    "Potansiyel (AI)": r["pot"],
                    "Fark": r["fark"],
                    "Tahmini Kazanç (€/t)": r["kazanc_text"],
                }
                for r in profit_rows
            ]
        )

        st.dataframe(df_profit, use_container_width=True, hide_index=True)

        total_gain = sum(
            r["kazanc_num"] for r in profit_rows if r["kazanc_num"] is not None
        )
        st.markdown(
            f"Toplam Potansiyel Kazanç (AI tahmini, ton başına – doğrudan hesaplanabilen kalemler): "
            f"≈ **{total_gain:.1f} €/t**"
        )

        # Info butonları – her satır için
        st.markdown("#### Hesaplama Mantığı (Özet)")

        for r in profit_rows:
            tag = r["tag"]
            btn_key = f"profit_btn_{tag}"
            state_key = f"profit_info_{tag}"

            cols = st.columns([0.1, 1.9])
            with cols[0]:
                if st.button("ℹ️", key=btn_key):
                    st.session_state.profit_info_state[state_key] = (
                        not st.session_state.profit_info_state.get(state_key, False)
                    )

            with cols[1]:
                st.markdown(f"**{r['deg']}**")

            if st.session_state.profit_info_state.get(state_key, False):
                if tag == "kwh_per_t":
                    st.info(
                        "Enerji tüketimi için ton başına kWh farkı hesaplanır. "
                        "Pozitif fark (Aktüel > AI) elektrik maliyetini artırır. "
                        "Kazanım = Fark (kWh/t) × 50 €/MWh ≈ 0.05 €/kWh üzerinden "
                        "yaklaşık elektrik gideri tasarrufu olarak değerlendirilir."
                    )
                elif tag == "electrode":
                    st.info(
                        "Elektrot tüketiminde hedef, Aktüel’den daha kötü olmamaktır. "
                        "AI, benzer koşullarda hafif bir iyileşme (≈0.003 kg/t) hedefler. "
                        "Kazanım = Fark (kg/t) × elektrot birim fiyatı (örnekte 3 €/kg). "
                        "Ayrıca daha stabil ark, dolaylı olarak kalite ve enerji verimini destekler."
                    )
                elif tag == "tap_temp_c":
                    st.info(
                        "Tap sıcaklığının gereksiz yüksek olması, süper ısıtma nedeniyle enerji kaybına "
                        "ve daha fazla oksidasyon/gaz alımına yol açar. Literatüre göre 3 °C düşüş, "
                        "yaklaşık 0.5–1.0 kWh/t enerji tasarrufu sağlayabilir ve çeliğin temizliğini "
                        "korumaya yardımcı olur. Bu yüzden burada hem **enerji tasarrufu aralığı** "
                        "hem de **Kalite ↑** birlikte gösterilmektedir."
                    )
                elif tag == "panel_delta_t":
                    st.info(
                        "Panel ΔT, su soğutmalı panellerin giriş–çıkış suyu sıcaklık farkıdır. "
                        "Çok düşük ΔT duvarların aşırı soğutulduğunu, çok yüksek ΔT ise aşırı ısınma "
                        "ve risk anlamına gelebilir. 18–22 °C aralığı, verimli ve güvenli bir bölge olarak "
                        "kabul edilir. Optimize edilmiş ΔT, daha homojen sıcaklık alanı, daha iyi hurda erimesi "
                        "ve daha düşük iç hurda/yeniden işleme oranı ile **Kalite ↑** etkisi yaratır."
                    )
                elif tag == "slag_foaming":
                    st.info(
                        "Köpüklü cüruf (slag foaming), arkın üzerini örterek ısı kaybını azaltır, "
                        "refrakterleri korur ve ark stabilitesini artırır. Çalışmalarda iyi köpük uygulamasının "
                        "enerji tüketimini %3–10, refrakter aşınmasını %25–60 azaltabildiği gösterilmiştir. "
                        "Ayrıca daha stabil proses, daha az azot/gaz alımı ve daha temiz çelik anlamına gelir. "
                        "Bu yüzden bu satırda **Kalite ↑ + Verim ↑** ifadesi kullanılmıştır."
                    )

    # -------------------------------
    # Basit öneriler (demo mantığı)
    # -------------------------------
    st.markdown("### Model Önerileri (Örnek / Demo Mantık)")
    suggestions = []

    if pd.notna(last.get("kwh_per_t", None)) and avg_kwh_t and not pd.isna(avg_kwh_t):
        if last["kwh_per_t"] > avg_kwh_t * 1.05:
            suggestions.append(
                "🔌 Son şarjın **kWh/t değeri**, son 10 şarj ortalamasına göre yüksek. "
                "Güç profilini ve O₂ kullanımını gözden geçirerek enerji verimini artırabilirsiniz."
            )

    if tap_w > 0 and real_elec_pt is not None and avg_electrode and not pd.isna(avg_electrode):
        avg_elec_pt = float(avg_electrode) / tap_w
        if real_elec_pt > avg_elec_pt * 1.05:
            suggestions.append(
                "🧯 **Elektrot tüketimi** ortalamanın üzerinde. Ark stabilitesi, empedans ayarları "
                "ve köpük cüruf uygulamasını kontrol etmek faydalı olabilir."
            )

    if real_tap > ai_tap_target + 2:
        suggestions.append(
            "🔥 Tap sıcaklığını birkaç derece düşürerek (örneğin ~3 °C), hem enerji tasarrufu "
            "hem de oksidasyon/gaz alımında azalma elde edilebilir."
        )

    if saving_potential > 0.0:
        suggestions.append(
            f"📉 kWh/t trendine göre, son kampanyada yaklaşık **{saving_potential:.1f} kWh/t** "
            "iyileştirme potansiyeli görülüyor."
        )

    if not suggestions:
        suggestions.append(
            "✅ Model açısından belirgin bir anomali veya iyileştirme alarmı görülmüyor. "
            "Mevcut ayarlar oldukça stabil görünüyor."
        )

    for s in suggestions:
        st.markdown(f"- {s}")


# ------------------------------------------------
# UYGULAMA BAŞLAT
# ------------------------------------------------
def main():
    # Sidebar: logo + isim + menü
    with st.sidebar:
        try:
            st.image("apple-touch-icon.png", width=72)
        except Exception:
            pass
        st.markdown("### FeCr AI")

        sim_mode = st.toggle(
            "Simülasyon Modu",
            value=False,
            help="Açıkken sistem canlı veri yerine simüle edilmiş veri kullanır.",
        )

        page = st.radio(
            "Sayfa Seç",
            ["1. Setup", "2. Canlı Veri", "3. Arc Optimizer"],
        )

    if page == "1. Setup":
        show_setup_form()
    elif page == "2. Canlı Veri":
        show_runtime_page(sim_mode)
    elif page == "3. Arc Optimizer":
        show_arc_optimizer_page(sim_mode)


if __name__ == "__main__":
    main()
