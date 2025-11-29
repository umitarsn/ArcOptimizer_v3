import os
from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Enerji Verimliliği",
    layout="wide",
    page_icon="🔋",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_sheets():
    file_name = "dc_saf_soru_tablosu.xlsx"
    try:
        xls = pd.ExcelFile(file_name)
        sheets = {name: xls.parse(name) for name in xls.sheet_names}
    except FileNotFoundError:
        st.error("Excel dosyası bulunamadı.")
        return None
    except Exception as e:
        st.error(f"Hata: {e}")
        return None

    cleaned = {}
    for name, df in sheets.items():
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if not df.empty:
            cleaned[name] = df

    return cleaned

def show_input_stats(sheets):
    total = 0
    filled = 0
    required_total = 0
    required_filled = 0
    missing_required = []

    for sheet_name, df in sheets.items():
        for idx, row in df.iterrows():
            key = f"val_{sheet_name}_{idx}"
            val = st.session_state.get(key, "")
            önem = int(row.get("Önem", 3))

            total += 1
            if str(val).strip() not in ["", "None", "nan"]:
                filled += 1
                if önem == 1:
                    required_filled += 1
            elif önem == 1:
                missing_required.append(f"{sheet_name} - {row.get('Tag')}")

            if önem == 1:
                required_total += 1

    percent_total = round(100 * filled / total, 1) if total else 0
    percent_required = round(100 * required_filled / required_total, 1) if required_total else 0

    st.sidebar.subheader("📊 Veri Giriş Durumu")
    st.sidebar.metric("Toplam Giriş Oranı", f"{percent_total}%")
    st.sidebar.progress(percent_total / 100)

    st.sidebar.metric("Zorunlu Veri Girişi", f"{percent_required}%")
    st.sidebar.progress(min(percent_required, 100) / 100)

    if missing_required:
        with st.sidebar.expander("❗Eksik Zorunlu Değerler"):
            st.write("\n".join(missing_required))

def show_energy_form():
    st.title("🔋 Enerji Verimliliği Formu")
    st.markdown(
        """
        Bu form **dc_saf_soru_tablosu.xlsx** dosyasına göre hazırlanmıştır.

        1. Girişi sadece **Set Değeri** alanına yapınız.
        2. 🔴 Zorunlu (Önem: 1), 🟡 Faydalı (Önem: 2), ⚪ Opsiyonel (Önem: 3) olarak belirtilmiştir.
        3. Detaylı bilgi ve açıklama için ℹ️ simgesine tıklayınız.
        """
    )

    sheets = load_sheets()
    if sheets is None:
        return

    show_input_stats(sheets)

    if "info_state" not in st.session_state:
        st.session_state.info_state = {}

    st.markdown("### 📝 Müşteri Girdileri")

    edited_data = {}

    for sheet_idx, (sheet_name, df) in enumerate(sheets.items(), start=1):
        with st.expander(f"{sheet_idx}. {sheet_name}", expanded=(sheet_idx == 1)):
            for idx, row in df.iterrows():
                row_key = f"{sheet_name}_{idx}"
                önem = int(row.get("Önem", 3))
                renk = {1: "🔴", 2: "🟡", 3: "⚪"}.get(önem, "⚪")
                birim = row.get("Set", "")
                val_key = f"val_{row_key}"

                cols = st.columns([2.2, 2.5, 4.0, 2.5, 0.7])  # ⬅️ Buradaki oran Set hizalamasını düzeltir
                cols[0].markdown(f"**{row.get('Tag', '')}**")
                cols[1].markdown(f"{renk} {row.get('Değişken', '')}")
                cols[2].markdown(row.get("Açıklama", ""))
                input_value = st.session_state.get(val_key, "")
                if input_value in [None, "None", "nan"]:
                    input_value = ""

                with cols[3]:
                    st.text_input(
                        label="",
                        key=val_key,
                        label_visibility="collapsed",
                        placeholder=birim if birim not in ["None", "nan"] else ""
                    )

                if cols[4].button("ℹ️", key=f"info_{row_key}"):
                    st.session_state.info_state[row_key] = not st.session_state.info_state.get(row_key, False)

                if st.session_state.info_state.get(row_key, False):
                    detaylar = []
                    if pd.notna(row.get("Detaylı Açıklama")):
                        detaylar.append(f"🔷 **Detaylı Açıklama:** {row['Detaylı Açıklama']}")
                    if pd.notna(row.get("Veri Kaynağı")):
                        detaylar.append(f"📌 **Kaynak:** {row['Veri Kaynağı']}")
                    if pd.notna(row.get("Kayıt Aralığı")):
                        detaylar.append(f"⏱️ **Kayıt Aralığı:** {row['Kayıt Aralığı']}")
                    if pd.notna(row.get("Önem")):
                        detaylar.append(f"🔵 **Önem:** {int(row['Önem'])}")

                    st.info("  \n".join(detaylar))

    if st.button("💾 Kaydet"):
        os.makedirs("data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join("data", f"energy_form_{timestamp}.xlsx")

        try:
            with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
                for sheet_name, df in sheets.items():
                    for idx in df.index:
                        key = f"val_{sheet_name}_{idx}"
                        if key in st.session_state:
                            df.at[idx, "Set"] = st.session_state[key]
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            st.success("Veriler başarıyla kaydedildi.")
            st.write(f"Kaydedilen dosya: `{out_file}`")
        except Exception as e:
            st.error(f"Kaydetme sırasında hata oluştu: {e}")

def main():
    show_energy_form()

if __name__ == "__main__":
    main()
