import os
from datetime import datetime

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# SAYFA AYARLARI
# ------------------------------------------------------------
st.set_page_config(
    page_title="Enerji Verimliliği",
    layout="wide",
    page_icon=None,
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# SORU TABLOSU YÜKLEME FONKSİYONU
# ------------------------------------------------------------
@st.cache_data
def load_questions():
    """
    AI - DC SAF Soru Tablosu.xlsx dosyasını yükler.
    Excel dosyası app.py ile aynı klasörde olmalı.
    """
    file_name = "AI - DC SAF Soru Tablosu.xlsx"
    try:
        df = pd.read_excel(file_name)
    except FileNotFoundError:
        st.error(f"❌ '{file_name}' dosyası bulunamadı. Lütfen proje klasörüne ekleyin.")
        return None
    return df


# ------------------------------------------------------------
# FORM SAYFASI
# ------------------------------------------------------------
def show_energy_efficiency_form():
    st.title("⚡ Enerji Verimliliği")
    st.markdown(
        """
        Bu form, **DC SAF enerji verimliliği ve proses optimizasyonu** için
        ihtiyaç duyulan verileri sistematik şekilde toplamak amacıyla hazırlanmıştır.  

        Lütfen tesisinizi en doğru şekilde yansıtacak biçimde doldurunuz.
        """
    )

    df_q = load_questions()
    if df_q is None:
        return

    # Sidebar’da basit bir başlık
    st.sidebar.header("Enerji Verimliliği")
    st.sidebar.info("Müşteri formunu doldurup kaydedebilir.")

    # Form
    with st.form("energy_efficiency_form"):
        st.subheader("Müşteri Girdileri")
        st.write("Aşağıdaki alanları kendi tesisiniz için doldurun:")

        responses = {}

        for idx, row in df_q.iterrows():
            label = row.get("Değer")
            unit = row.get("Birim")
            aciklama = row.get("Açıklama")
            veri_kaynagi = row.get("Veri Kaynağı")

            # Boş satır veya tamamen açıklama satırı ise atla
            if pd.isna(label):
                continue

            # Eğer hem Birim hem Veri Kaynağı boşsa bunu bölüm başlığı gibi göster
            if (pd.isna(unit) or str(unit).strip() == "") and (
                pd.isna(veri_kaynagi) or str(veri_kaynagi).strip() == ""
            ):
                st.markdown(f"### {label}")
                continue

            field_key = f"q_{idx}"

            full_label = str(label)
            if isinstance(unit, str) and unit.strip() not in ["", "–", "-"]:
                full_label = f"{label} ({unit})"

            help_arg = (
                aciklama if isinstance(aciklama, str) and aciklama.strip() != "" else None
            )

            # Basit widget mantığı: birim içinde tipik sayısal semboller varsa number_input
            unit_str = str(unit) if not pd.isna(unit) else ""
            numeric_unit_tokens = [
                "%",
                "°C",
                "kWh",
                "kW",
                "Nm³",
                "Nm3",
                "t",
                "kg",
                "MJ",
            ]
            if any(sym in unit_str for sym in numeric_unit_tokens):
                responses[field_key] = st.number_input(
                    full_label,
                    value=0.0,
                    step=0.1,
                    help=help_arg,
                    key=field_key,
                )
            else:
                responses[field_key] = st.text_input(
                    full_label,
                    help=help_arg,
                    key=field_key,
                )

        submitted = st.form_submit_button("Kaydet")

    if submitted:
        # Kayıtları uzun formatta kaydediyoruz: her satır bir soru-cevap
        records = []
        timestamp = datetime.now().isoformat(timespec="seconds")

        for idx, row in df_q.iterrows():
            label = row.get("Değer")
            unit = row.get("Birim")

            if pd.isna(label):
                continue

            key = f"q_{idx}"
            if key not in responses:
                continue

            records.append(
                {
                    "timestamp": timestamp,
                    "Parametre": label,
                    "Birim": "" if pd.isna(unit) else str(unit),
                    "Cevap": responses[key],
                }
            )

        if not records:
            st.warning("Herhangi bir cevap bulunamadı, form alanları boş olabilir.")
            return

        resp_df = pd.DataFrame(records)

        # Dosya kaydı
        os.makedirs("data", exist_ok=True)
        resp_file = os.path.join("data", "energy_efficiency_form_responses.csv")

        if os.path.exists(resp_file):
            existing = pd.read_csv(resp_file)
            combined = pd.concat([existing, resp_df], ignore_index=True)
        else:
            combined = resp_df

        combined.to_csv(resp_file, index=False)

        st.success("✅ Form yanıtlarınız kaydedildi.")
        st.subheader("Son Yanıtınız")
        st.dataframe(resp_df, use_container_width=True)

        with st.expander("Tüm Kayıtlı Yanıtlar (Yerel Dosya)"):
            st.dataframe(combined.tail(200), use_container_width=True)


# ------------------------------------------------------------
# ANA FONKSİYON
# ------------------------------------------------------------
def main():
    show_energy_efficiency_form()


if __name__ == "__main__":
    main()
