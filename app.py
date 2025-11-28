
import os
from datetime import datetime
import pandas as pd
import streamlit as st

# Sayfa ayarı
st.set_page_config(page_title="Enerji Verimliliği", layout="wide", page_icon=None, initial_sidebar_state="expanded")

@st.cache_data
def load_sheets():
    file_name = "dc_saf_soru_tablosu.xlsx"
    try:
        sheets = pd.read_excel(file_name, sheet_name=None, header=0)
    except FileNotFoundError:
        st.error("HATA: 'dc_saf_soru_tablosu.xlsx' bulunamadı. Dosyayı app.py ile aynı klasöre koyun.")
        return None
    except Exception as e:
        st.error(f"Excel okunurken hata oluştu: {e}")
        return None

    cleaned = {}
    for name, df in sheets.items():
        if df is not None:
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if not df.empty:
                cleaned[name] = df
    return cleaned

def show_energy_form():
    st.title("📥 Enerji Verimliliği Formu")

    st.markdown("""
    Bu form **dc_saf_soru_tablosu.xlsx** dosyasına göre hazırlanmıştır.
    - A, B, C: Açıklama alanları
    - D: Müşteri girişi yapılacak alan
    - ℹ️ işaretli satırlar seçilerek detay (E, F, G...) açıklamalar aşağıda görülebilir.
    - 🔴 Zorunlu (Önem: 1), 🟡 Faydalı (Önem: 2), ⚪ Opsiyonel (Önem: 3)
    """)

    sheets = load_sheets()
    if sheets is None:
        return

    total_rows = sum(len(df) for df in sheets.values())
    with st.sidebar:
        st.subheader("Form Özeti")
        st.info(f"Toplam satır sayısı: {total_rows}")

    edited_sheets = {}

    with st.form("energy_form"):
        st.subheader("📝 Müşteri Girdileri")

        for i, (sheet_name, df_full) in enumerate(sheets.items(), start=1):
            with st.expander(f"{i}. {sheet_name}", expanded=(i == 1)):
                if df_full.shape[1] < 4:
                    st.warning("Bu sayfa 4 sütun içermiyor, atlanıyor.")
                    continue

                col_A, col_B, col_C, col_D = df_full.columns[:4]
                detail_cols = df_full.columns[4:]

                view_df = df_full[[col_A, col_B, col_C, col_D]].copy()
    if "Önem" in df_full.columns:
        view_df[col_B] = df_full["Önem"].astype(str).map({
            "1": "🔴 " + df_full[col_B],
            "2": "🟡 " + df_full[col_B],
            "3": df_full[col_B]
        }).fillna(df_full[col_B])
    
                view_df["Info"] = "ℹ️"

                renk_map = {"1": "#FFC7CE", "2": "#FFEB9C", "3": "#FFFFFF"}
                if "Önem" in df_full.columns:
                    view_df["renk"] = df_full["Önem"].astype(str).map(renk_map).fillna("#FFFFFF")
                else:
                    view_df["renk"] = "#FFFFFF"

                st.caption("Zorunlu alanlar kırmızı, faydalı olanlar sarı ile işaretlidir.")
                edited_view = st.data_editor(
                    view_df.drop(columns=["renk"]),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        col_A: st.column_config.TextColumn(disabled=True),
                        col_B: st.column_config.TextColumn(disabled=True),
                        col_C: st.column_config.TextColumn(disabled=True),
                        "Info": st.column_config.TextColumn(disabled=True),
                        col_D: st.column_config.TextColumn(),
                    },
                    key=f"sheet_{i}_view",
                )

                # Detay INFO
                if detail_cols.any():
                    st.markdown("ℹ️ Satır seçin, açıklama gösterilsin:")
                    labels = [f"{row[col_A]} - {row[col_B]}" for idx, row in edited_view.iterrows()]
                    indices = list(edited_view.index)

                    if labels:
                        selected_label = st.selectbox("Satır seç:", options=labels, key=f"sheet_{i}_detail_select")
                        sel_idx = indices[labels.index(selected_label)]
                        detail_row = df_full.loc[sel_idx, detail_cols]
                        details = [f"- **{col}**: {val}" for col, val in detail_row.items() if pd.notna(val) and str(val).strip()]
                        if details:
                            st.info("\n".join(details))

                edited_sheets[sheet_name] = (df_full, edited_view, col_D)

        submitted = st.form_submit_button("💾 Kaydet")

    if submitted:
        os.makedirs("data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join("data", f"energy_form_{timestamp}.xlsx")

        try:
            with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
                for name, (df_full, edited_view, col_D) in edited_sheets.items():
                    df_full[col_D] = edited_view[col_D]
                    df_full.to_excel(writer, sheet_name=name[:31], index=False)
        except Exception as e:
            st.error(f"Veri kaydında hata: {e}")
            return

        st.success("✔️ Veriler başarıyla kaydedildi.")
        st.write(f"📁 Dosya adı: `{out_file}`")

def main():
    show_energy_form()

if __name__ == "__main__":
    main()
