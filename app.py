import os
import json
from datetime import datetime
import pandas as pd
import streamlit as st

# ----------------------------------------------
# GENEL AYARLAR
# ----------------------------------------------
st.set_page_config(
    page_title="FeCr AI",               # Sekme / iOS varsayılan adı
    page_icon="apple-touch-icon.png",   # Repo root'taki logo
    layout="wide",
)

# Sabit inputların kaydedileceği dosya
SETUP_SAVE_PATH = "data/saved_inputs.json"
# Runtime (şarj bazlı) verilerin kaydedileceği dosya
RUNTIME_SAVE_PATH = "data/runtime_data.json"

os.makedirs("data", exist_ok=True)

# ----------------------------------------------
# KAYITLI SETUP VERİLERİNİ YÜKLE
# ----------------------------------------------
if os.path.exists(SETUP_SAVE_PATH):
    with open(SETUP_SAVE_PATH, "r") as f:
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
            with open(RUNTIME_SAVE_PATH, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []

def save_runtime_data(data_list):
    try:
        with open(RUNTIME_SAVE_PATH, "w") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Runtime verileri kaydedilemedi: {e}")

runtime_data = load_runtime_data()

# ----------------------------------------------
# EXCEL OKUMA (SETUP SAYFASI İÇİN)
# ----------------------------------------------
@st.cache_data
def load_sheets():
    file_name = "dc_saf_soru_tablosu.xlsx"
    try:
        xls = pd.read_excel(file_name, sheet_name=None)
        return {k: v.dropna(how="all") for k, v in xls.items() if not v.empty}
    except Exception as e:
        st.error(f"Excel dosyası yüklenemedi: {e}")
        return {}

# ----------------------------------------------
# 1) SETUP SAYFASI – SABİT GİRDİLER
# ----------------------------------------------
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
                if unit_col_name:
                    raw_birim = row.get(unit_col_name, "")
                else:
                    raw_birim = ""

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
                            with open(SETUP_SAVE_PATH, "w") as f:
                                json.dump(saved_inputs, f, ensure_ascii=False, indent=2)

                    with unit_col:
                        unit_text = f"**{birim}**" if birim else ""
                        st.markdown(unit_text)

                with cols[4]:
                    if st.button("ℹ️", key=f"info_{row_key}"):
                        st.session_state.info_state[row_key] = not st.session_state.info_state.get(row_key, False)

                if st.session_state.info_state.get(row_key, False):
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

# ----------------------------------------------
# 2) CANLI VERİ SAYFASI – ŞARJ BAZLI ANLIK VERİ
# ----------------------------------------------
def show_runtime_page():
    st.markdown("## 2. Canlı Veri – Şarj Bazlı Anlık Veriler")
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
            tap_temp = st.number_input("Tap Sıcaklığı (°C)", min_value=0.0, max_value=2000.0, step=1.0)
        with c6:
            o2_flow = st.number_input("Ortalama O2 Debisi (Nm³/h)", min_value=0.0, step=1.0)

        c7, c8, c9 = st.columns(3)
        with c7:
            slag_foaming = st.slider("Slag Foaming Seviyesi (0–10)", 0, 10, 5)
        with c8:
            panel_delta_t = st.number_input("Panel ΔT (°C)", min_value=0.0, step=0.1)
        with c9:
            electrode_cons = st.number_input("Elektrot Tüketimi (kg/şarj)", min_value=0.0, step=0.01)

        note = st.text_input("Operatör Notu (opsiyonel)", "")

        submitted = st.form_submit_button("Kaydet")

    if submitted:
        if not heat_id:
            st.error("Heat ID / Şarj No girilmesi zorunludur.")
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

    # Kayıtlı runtime verileri tablo + basit grafik olarak göster
    if not runtime_data:
        st.info("Henüz canlı veri girilmemiş.")
        return

    df = pd.DataFrame(runtime_data)
    # timestamp’i datetime’a çevir
    try:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp_dt")
    except Exception:
        df["timestamp_dt"] = df["timestamp"]

    st.markdown("### Kayıtlı Canlı Veriler (Runtime)")
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

# ----------------------------------------------
# 3) ARC OPTIMIZER SAYFASI – MODEL OUTPUT & INSIGHTS
# ----------------------------------------------
def show_arc_optimizer_page():
    st.markdown("## 3. Arc Optimizer – Trendler, KPI ve Öneriler")
    st.markdown(
        "Bu sayfa, canlı veriler üzerinden **enerji verimliliği**, "
        "**elektrot tüketimi** ve **proses stabilitesi** ile ilgili özet KPI ve "
        "modelin önerilerini gösterir."
    )

    if not runtime_data:
        st.info("Arc Optimizer çıktıları için henüz canlı veri yok. Önce 2. sayfadan veri ekleyin.")
        return

    df = pd.DataFrame(runtime_data)
    try:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp_dt")
    except Exception:
        df["timestamp_dt"] = df["timestamp"]

    # Son şarj ve son N şarj
    last = df.iloc[-1]
    last_n = df.tail(10)

    # KPI hesapları
    avg_kwh_t = last_n["kwh_per_t"].dropna().mean()
    avg_electrode = last_n["electrode_kg_per_heat"].dropna().mean()
    avg_tap_temp = last_n["tap_temp_c"].dropna().mean()

    # Basit "iyileşme potansiyeli" hesabı (tamamen örnek / placeholder)
    if len(df) >= 10 and df["kwh_per_t"].notna().sum() >= 10:
        first5 = df["kwh_per_t"].dropna().head(5).mean()
        last5 = df["kwh_per_t"].dropna().tail(5).mean()
        saving_potential = max(0.0, first5 - last5)
    else:
        saving_potential = 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Son Şarj kWh/t", f"{last['kwh_per_t']:.1f}" if pd.notna(last["kwh_per_t"]) else "-")
    col2.metric("Son Şarj Elektrot", f"{last['electrode_kg_per_heat']:.2f} kg/şarj")
    col3.metric("Son Tap Sıcaklığı", f"{last['tap_temp_c']:.0f} °C")
    col4.metric("Son 10 Şarj Ort. kWh/t", f"{avg_kwh_t:.1f}" if pd.notna(avg_kwh_t) else "-")

    st.markdown("### Trendler")
    trend_df = df.set_index("timestamp_dt")[["kwh_per_t", "tap_temp_c", "electrode_kg_per_heat"]]
    st.line_chart(trend_df.rename(columns={
        "kwh_per_t": "kWh/t",
        "tap_temp_c": "Tap T (°C)",
        "electrode_kg_per_heat": "Elektrot (kg/şarj)",
    }))

    # Basit öneriler (placeholder mantık)
    st.markdown("### Model Önerileri (Demo Mantık)")
    suggestions = []

    if pd.notna(last["kwh_per_t"]) and avg_kwh_t and last["kwh_per_t"] > avg_kwh_t * 1.05:
        suggestions.append(
            "🔌 Son şarjın **kWh/t değeri**, son 10 şarj ortalamasına göre yüksek görünüyor. "
            "Oksijen debisini optimize etmeyi ve güç profilini gözden geçirmeyi düşünün."
        )

    if pd.notna(last["electrode_kg_per_heat"]) and avg_electrode and last["electrode_kg_per_heat"] > avg_electrode * 1.10:
        suggestions.append(
            "🧯 **Elektrot tüketimi** son şarjda yükselmiş. Ark stabilitesini (arc length, voltage) kontrol edin; "
            "aşırı salınımlar olabilir."
        )

    if pd.notna(last["tap_temp_c"]) and avg_tap_temp and last["tap_temp_c"] < avg_tap_temp - 10:
        suggestions.append(
            "🔥 Tap sıcaklığı son şarjda düşük. Bir sonraki şarj için enerji girişini hafif artırmak veya "
            "şarj sonu bekleme süresini optimize etmek gerekebilir."
        )

    if last.get("slag_foaming_index", None) is not None and last["slag_foaming_index"] >= 8:
        suggestions.append(
            "🌋 Slag foaming seviyesi yüksek (≥8). Karbon/O2 dengesini ve köpük kontrolünü gözden geçirin."
        )

    if last.get("panel_delta_t_c", None) is not None and last["panel_delta_t_c"] > 25:
        suggestions.append(
            "💧 Panel ΔT yüksek. Soğutma devresinde dengesizlik olabilir; panel debilerini kontrol edin."
        )

    if not suggestions:
        suggestions.append(
            "✅ Model açısından belirgin bir anomali veya iyileştirme alarmı görülmüyor. "
            "Mevcut ayarlar stabil görünüyor."
        )

    for s in suggestions:
        st.markdown(f"- {s}")

# ----------------------------------------------
# UYGULAMA BAŞLAT
# ----------------------------------------------
def main():
    # SOL SIDEBAR: LOGO + İSİM + MENÜ
    with st.sidebar:
        try:
            st.image("apple-touch-icon.png", width=72)
        except Exception:
            pass  # logo bulunamazsa app yine de çalışsın
        st.markdown("### FeCr AI")

        page = st.radio(
            "Sayfa Seç",
            ["1. Setup", "2. Canlı Veri", "3. Arc Optimizer"],
        )

    if page == "1. Setup":
        show_setup_form()
    elif page == "2. Canlı Veri":
        show_runtime_page()
    elif page == "3. Arc Optimizer":
        show_arc_optimizer_page()

if __name__ == "__main__":
    main()
