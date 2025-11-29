import os
from datetime import datetime
import pandas as pd
import streamlit as st

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

def load_previous_inputs(file_name):
    try:
        return pd.read_csv(file_name)
    except:
        return None

def save_current_inputs(dataframes, file_name):
    combined = []
    for name, (df_full, edited_view, col_D) in dataframes.items():
        edited = edited_view[[col_D]].copy()
        edited.insert(0, "Sheet", name)
        edited.insert(1, "Tag", df_full[df_full.columns[0]])
        combined.append(edited)
    all_data = pd.concat(combined)
    all_data.to_csv(file_name, index=False)

def show_input_stats(sheets):
    st.sidebar.subheader("🧮 Veri Giriş Durumu")

    total_cells = 0
    filled_cells = 0
    required_cells = 0
    filled_required = 0
    missing_required_entries = []

    for sheet_name, df in sheets.items():
        if df is None or df.empty or df.shape[1] < 4:
            continue
        col_D = df.columns[3]
        if "Önem" not in df.columns:
            continue
        for idx, row in df.iterrows():
            val = row[col_D]
            importance = str(row["Önem"]).strip()
            total_cells += 1
            if pd.notna(val) and str(val).strip() != "":
                filled_cells += 1
                if importance == "1":
                    filled_required += 1
            elif importance == "1":
                missing_required_entries.append((sheet_name, row[0], row[1]))
                required_cells += 1

    overall_pct = int(100 * filled_cells / total_cells) if total_cells else 0
    required_pct = int(100 * filled_required / required_cells) if required_cells else 0

    st.sidebar.metric("Toplam Giriş Oranı", f"{overall_pct}%")
    st.sidebar.progress(min(overall_pct / 100, 1.0))

    st.sidebar.metric("Zorunlu Veri Girişi", f"{required_pct}%")
    st.sidebar.progress(min(required_pct / 100, 1.0))

    if missing_required_entries:
        with st.sidebar.expander("❗ Eksik Zorunlu Değerler"):
            for sheet, tag, name in missing_required_entries:
                st.write(f"📄 `{sheet}` → **{tag} - {name}**")

def show_energy_form():
    st.title("📥 Enerji Verimliliği Formu")
    st.markdown("""
    Bu form **dc_saf_soru_tablosu.xlsx** dosyasına göre hazırlanmıştır.  
    1. Girişi sadece **Set Değeri** alanına yapınız.  
    2. 🔴 Zorunlu (Önem: 1), 🟡 Faydalı (Önem: 2), ⚪ Opsiyonel (Önem: 3) olarak belirtilmiştir.  
    3. Detaylı bilgi ve açıklama için ℹ️ simgesine tıklayınız.
    """)

    sheets = load_sheets()
    if sheets is None:
        return

    show_input_stats(sheets)
    edited_sheets = {}

    # Veri Giriş Formu
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
                    renk_map = {"1": "🔴", "2": "🟡", "3": ""}
                    view_df[col_B] = df_full["Önem"].astype(str).map(renk_map).fillna("") + " " + df_full[col_B].astype(str)

                prev_inputs = load_previous_inputs("data/last_inputs.csv")
                if prev_inputs is not None:
                    for idx in view_df.index:
                        row_tag = df_full.loc[idx, col_A]
                        match = prev_inputs[(prev_inputs['Sheet'] == sheet_name) & (prev_inputs['Tag'] == row_tag)]
                        if not match.empty:
                            view_df.loc[idx, col_D] = match.iloc[0][col_D]

                edited_view = st.data_editor(
                    view_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        col_A: st.column_config.TextColumn(disabled=True),
                        col_B: st.column_config.TextColumn(disabled=True),
                        col_C: st.column_config.TextColumn(disabled=True),
                        col_D: st.column_config.TextColumn(),
                    },
                    key=f"sheet_{i}_view",
                )

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
            save_current_inputs(edited_sheets, "data/last_inputs.csv")
        except Exception as e:
            st.error(f"Veri kaydında hata: {e}")
            return
        st.success("✔️ Veriler başarıyla kaydedildi.")
        st.write(f"📁 Dosya adı: `{out_file}`")

    # Form dışında info açıklama bölümü
    st.divider()
    st.subheader("📖 Açıklamalar")
    for sheet_name, df_full in sheets.items():
        col_A, col_B = df_full.columns[:2]
        detail_cols = df_full.columns[4:]
        with st.expander(f"📄 {sheet_name}"):
            for idx, row in df_full.iterrows():
                label = f"{row[col_A]} - {row[col_B]}"
                if st.button(f"ℹ️ {label}", key=f"info_button_{sheet_name}_{idx}"):
                    detail_row = df_full.loc[idx, detail_cols]
                    details = []
                    if pd.notna(detail_row.iloc[0]):
                        details.append(f"🧾 **Detaylı Açıklama:** {detail_row.iloc[0]}")
                    if len(detail_cols) > 1 and pd.notna(detail_row.iloc[1]):
                        details.append(f"📊 **Veri Kaynağı:** {detail_row.iloc[1]}")
                    if len(detail_cols) > 2 and pd.notna(detail_row.iloc[2]):
                        details.append(f"⏱ **Kayıt Aralığı:** {detail_row.iloc[2]}")
                    if details:
                        st.info("\n\n".join(details))

def main():
    show_energy_form()

if __name__ == "__main__":
    main()
