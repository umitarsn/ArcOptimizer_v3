import os
import json
from datetime import datetime
import pandas as pd
import streamlit as st

# ----------------------------------------------
# GENEL AYARLAR
# ----------------------------------------------
st.set_page_config(
    page_title="BG Arc Optimizer",
    layout="wide",
)

SAVE_PATH = "data/saved_inputs.json"
os.makedirs("data", exist_ok=True)

if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "r") as f:
        saved_inputs = json.load(f)
else:
    saved_inputs = {}

if "info_state" not in st.session_state:
    st.session_state.info_state = {}

# ----------------------------------------------
# EXCEL OKUMA
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
# 1) VERİ GİRİŞİ SAYFASI
# ----------------------------------------------
def show_energy_form():
    st.markdown("## 1. Veri Girişi")
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
                            with open(SAVE_PATH, "w") as f:
                                json.dump(saved_inputs, f)

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

    # Sidebar özet
    st.sidebar.subheader("📊 Veri Giriş Durumu")

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
# 2) AI MODEL SAYFASI
# ----------------------------------------------
def show_ai_model_page():
    st.markdown("## 2. AI Model")
    st.markdown(
        "Bu sayfada **BG Arc Optimizer** yapay zeka modelinin nasıl çalıştığı özetlenir.\n\n"
        "### Model Girdileri\n"
        "- Proses ve tasarım verileri\n"
        "- Şarj planı, enerji ve sıcaklık profilleri\n\n"
        "### Model Adımları (özet)\n"
        "1. Veri toplama ve temizleme\n"
        "2. Özellik çıkarma (feature engineering)\n"
        "3. Eğitimli model ile tahmin\n"
        "4. Optimizasyon döngüsü ve operatöre öneri üretimi\n"
    )

# ----------------------------------------------
# 3) ARC OPTIMIZER – TREND SAYFASI
# ----------------------------------------------
def show_arc_optimizer_page():
    st.markdown("## 3. Arc Optimizer – Trendler ve Proses Gidişatı")
    st.markdown(
        "Bu sayfada, fırın performansını ve proses gidişatını izlemek için örnek trendler gösterilmektedir. "
        "Gerçek veriye bağlandığında aynı grafik yapısı kullanılacaktır."
    )

    tarih = pd.date_range(end=datetime.now(), periods=24, freq="H")
    seri_indeks = pd.Series(range(24))

    spesifik_enerji = 420 + 15 * seri_indeks.rolling(3, min_periods=1).mean()
    tap_sicaklik = 1610 + 5 * seri_indeks.rolling(4, min_periods=1).mean()
    elektrot_tuketim = 1.8 + 0.05 * seri_indeks.rolling(5, min_periods=1).mean()

    demo_df = pd.DataFrame(
        {
            "Spesifik Enerji (kWh/t)": spesifik_enerji,
            "Tap Sıcaklığı (C)": tap_sicaklik,
            "Elektrot Tüketimi (kg/şarj)": elektrot_tuketim,
        },
        index=tarih,
    )

    st.subheader("Spesifik Enerji ve Tap Sıcaklığı")
    st.line_chart(demo_df[["Spesifik Enerji (kWh/t)", "Tap Sıcaklığı (C)"]])

    st.subheader("Elektrot Tüketimi")
    st.line_chart(demo_df[["Elektrot Tüketimi (kg/şarj)"]])

    son_spesifik_enerji = demo_df["Spesifik Enerji (kWh/t)"].iloc[-1]
    son_tap_sicaklik = demo_df["Tap Sıcaklığı (C)"].iloc[-1]
    son_elektrot_tuketim = demo_df["Elektrot Tüketimi (kg/şarj)"].iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Son Şarj Spesifik Enerji", f"{son_spesifik_enerji:.1f} kWh/t")
    col2.metric("Son Tap Sıcaklığı", f"{son_tap_sicaklik:.0f} C")
    col3.metric("Son Elektrot Tüketimi", f"{son_elektrot_tuketim:.2f} kg/şarj")

# ----------------------------------------------
# UYGULAMA BAŞLAT
# ----------------------------------------------
def main():
    with st.sidebar:
        st.title("BG Arc Optimizer")
        page = st.radio(
            "Sayfa Seç",
            ["1. Veri Girişi", "2. AI Model", "3. Arc Optimizer"],
        )

    if page == "1. Veri Girişi":
        show_energy_form()
    elif page == "2. AI Model":
        show_ai_model_page()
    elif page == "3. Arc Optimizer":
        show_arc_optimizer_page()

if __name__ == "__main__":
    main()

