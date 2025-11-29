import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.set_page_config(
    page_title="Enerji Verimliliği Formu",
    layout="wide",
)

@st.cache_data
def load_sheets():
    try:
        df_dict = pd.read_excel("dc_saf_soru_tablosu.xlsx", sheet_name=None)
        return df_dict
    except Exception as e:
        st.error(f"Excel dosyası yüklenemedi: {e}")
        return {}

def show_input_stats(sheets):
    total = 0
    filled = 0
    required = 0
    required_filled = 0

    for sheet in sheets.values():
        for _, row in sheet.iterrows():
            val = row.get("Set")
            if pd.notna(val) and str(val).strip() != "" and str(val).strip() not in ["%", "None", "nan"]:
                filled += 1
                if row.get("Önem") == 1:
                    required_filled += 1
            if row.get("Önem") == 1:
                required += 1
            total += 1

    pct = round(100 * filled / total, 1) if total else 0
    required_pct = round(100 * required_filled / required, 1) if required else 0
    required_missing = required - required_filled

    with st.sidebar:
        st.subheader("📊 Veri Giriş Durumu")
        st.metric("Toplam Giriş Oranı", f"{pct}%")
        st.progress(pct / 100)

        st.metric("Zorunlu Veri Girişi", f"{required_pct}%")
        st.progress(min(required_pct / 100, 1))

        if required_missing > 0:
            st.warning(f"❗ Eksik Zorunlu Değerler: {required_missing}")

def show_energy_form():
    st.markdown("## 📥 Enerji Verimliliği Formu")
    st.markdown(
        """
        Bu form **dc_saf_soru_tablosu.xlsx** dosyasına göre hazırlanmıştır.

        1. Girişi sadece **Set Değeri** alanına yapınız.  
        2. 🔴 Zorunlu (Önem: 1), 🟡 Faydalı (Önem: 2), ⚪ Opsiyonel (Önem: 3) olarak belirtilmiştir.  
        3. Detaylı bilgi ve açıklama için ℹ️ simgesine tıklayınız.
        """
    )

    sheets = load_sheets()
    if not sheets:
        return

    show_input_stats(sheets)

    if "info_state" not in st.session_state:
        st.session_state.info_state = {}

    with st.form("veri_formu"):
        for sheet_idx, (sheet_name, df) in enumerate(sheets.items(), start=1):
            with st.expander(f"{sheet_idx}. {sheet_name}", expanded=(sheet_idx == 1)):
                st.markdown(
                    """
                    <style>
                        th { text-align: left !important; }
                        td { vertical-align: top !important; padding-top: 0.3em; padding-bottom: 0.3em; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <table style="width:100%">
                        <thead>
                            <tr>
                                <th>Tag</th>
                                <th>Değişken</th>
                                <th>Açıklama</th>
                                <th>Set</th>
                                <th>Info</th>
                            </tr>
                        </thead>
                        <tbody>
                    """,
                    unsafe_allow_html=True,
                )

                for idx, row in df.iterrows():
                    row_key = f"{sheet_idx}_{idx}"
                    önem = int(row.get("Önem", 3))
                    renk = {1: "🔴", 2: "🟡", 3: "⚪"}.get(önem, "⚪")
                    tag = row.get("Tag", "")
                    name = row.get("Değişken", "")
                    desc = row.get("Açıklama", "")
                    birim = row.get("Set", "") if str(row.get("Set")).strip() not in ["None", "nan"] else ""
                    val_key = f"val_{row_key}"

                    # Form input
                    cols = st.columns([2.2, 2.5, 3.5, 2, 0.7])
                    cols[0].markdown(f"**{tag}**")
                    cols[1].markdown(f"{renk} {name}")
                    cols[2].markdown(desc)
                    df.at[idx, "Set"] = cols[3].text_input(
                        label="",
                        key=val_key,
                        label_visibility="collapsed",
                        placeholder=birim
                    )

                    if cols[4].button("ℹ️", key=f"info_{row_key}"):
                        st.session_state.info_state[row_key] = not st.session_state.info_state.get(row_key, False)

                    # Detayları göster
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

                st.markdown("</tbody></table>", unsafe_allow_html=True)

        submitted = st.form_submit_button("💾 Kaydet")
        if submitted:
            os.makedirs("data", exist_ok=True)
            filename = datetime.now().strftime("veri_formu_%Y%m%d_%H%M%S.xlsx")
            filepath = os.path.join("data", filename)
            with pd.ExcelWriter(filepath) as writer:
                for sheet_name, df in sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            st.success(f"Veriler kaydedildi: {filename}")

def main():
    show_energy_form()

if __name__ == "__main__":
    main()
