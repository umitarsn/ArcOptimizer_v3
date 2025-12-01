    # ============================================================
    #  💰 PROSES KAZANÇ TABLOSU – TON BAŞINA
    # ============================================================
    st.markdown("### 💰 Proses Kazanç Analizi (Ton Başına)")

    ENERGY_PRICE_EUR_PER_KWH = 0.12       # örnek elektrik birim fiyatı
    ELECTRODE_PRICE_EUR_PER_KG = 3.0      # örnek elektrot fiyatı
    TYPICAL_HEAT_TON = float(last.get("tap_weight_t", 35.0) or 35.0)

    rows = []
    total_gain_per_t = 0.0

    # ---------- 1) kWh/t – Enerji Tüketimi ----------
    if pd.notna(last.get("kwh_per_t", None)) and avg_kwh_t and not pd.isna(avg_kwh_t):
        real_kwh_t = float(last["kwh_per_t"])
        target_kwh_t = max(avg_kwh_t - 5.0, 0.0)   # AI hedef: ortalamadan 5 kWh/t daha iyi
        diff_kwh_t = real_kwh_t - target_kwh_t

        # Pozitif fark → iyileştirme potansiyeli → kazanç
        gain_kwh_per_t = max(0.0, diff_kwh_t) * ENERGY_PRICE_EUR_PER_KWH
        total_gain_per_t += gain_kwh_per_t

        rows.append({
            "Tag": "kwh_per_t",
            "Değişken": "Enerji tüketimi",
            "Gerçek": f"{real_kwh_t:.1f} kWh/t",
            "Hedef": f"{target_kwh_t:.1f} kWh/t",
            "Fark": f"{diff_kwh_t:+.1f} kWh/t",
            "Tahmini Kazanç (€/t)": f"{gain_kwh_per_t:.1f} €/t" if gain_kwh_per_t > 0 else "-"
        })

    # ---------- 2) Electrode – Elektrot Tüketimi ----------
    if pd.notna(last.get("electrode_kg_per_heat", None)) and pd.notna(last.get("tap_weight_t", None)):
        tap_weight = float(last["tap_weight_t"]) if last["tap_weight_t"] else None
        if tap_weight and tap_weight > 0:
            real_electrode_per_t = float(last["electrode_kg_per_heat"]) / tap_weight  # kg/t

            if pd.notna(avg_electrode):
                # hedef: son 10 şarj ortalamasına göre kg/t
                target_electrode_per_t = max(avg_electrode / tap_weight, 0.0)
            else:
                # veri yoksa: bugünkü değerden 0.05 kg/t iyileştirme hedefi
                target_electrode_per_t = max(real_electrode_per_t - 0.05, 0.0)

            diff_electrode_per_t = real_electrode_per_t - target_electrode_per_t
            gain_electrode_per_t = max(0.0, diff_electrode_per_t) * ELECTRODE_PRICE_EUR_PER_KG
            total_gain_per_t += gain_electrode_per_t

            rows.append({
                "Tag": "electrode",
                "Değişken": "Elektrot tüketimi",
                "Gerçek": f"{real_electrode_per_t:.3f} kg/t",
                "Hedef": f"{target_electrode_per_t:.3f} kg/t",
                "Fark": f"{diff_electrode_per_t:+.3f} kg/t",
                "Tahmini Kazanç (€/t)": f"{gain_electrode_per_t:.1f} €/t" if gain_electrode_per_t > 0 else "-"
            })

    # ---------- 3) Tap Sıcaklığı – Dolaylı Etki ----------
    if pd.notna(last.get("tap_temp_c", None)) and avg_tap_temp and not pd.isna(avg_tap_temp):
        real_tap = float(last["tap_temp_c"])
        target_tap = float(avg_tap_temp)
        diff_tap = real_tap - target_tap

        rows.append({
            "Tag": "tap_temp_c",
            "Değişken": "Tap sıcaklığı",
            "Gerçek": f"{real_tap:.0f} °C",
            "Hedef": f"{target_tap:.0f} °C",
            "Fark": f"{diff_tap:+.0f} °C",
            "Tahmini Kazanç (€/t)": "Dolaylı"
        })

    # ---------- 4) Panel ΔT – Dolaylı Risk ----------
    if pd.notna(last.get("panel_delta_t_c", None)):
        real_panel = float(last["panel_delta_t_c"])
        target_panel = 20.0  # örnek hedef
        diff_panel = real_panel - target_panel

        rows.append({
            "Tag": "panel_delta_t",
            "Değişken": "Panel ΔT",
            "Gerçek": f"{real_panel:.1f} °C",
            "Hedef": f"{target_panel:.1f} °C",
            "Fark": f"{diff_panel:+.1f} °C",
            "Tahmini Kazanç (€/t)": "Dolaylı"
        })

    # ---------- 5) Slag Foaming – Dolaylı Kalite ----------
    if last.get("slag_foaming_index", None) is not None:
        real_slag = float(last["slag_foaming_index"])
        target_slag = 7.0  # örnek ideal köpük seviyesi
        diff_slag = real_slag - target_slag

        rows.append({
            "Tag": "slag_foaming",
            "Değişken": "Köpük seviyesi",
            "Gerçek": f"{real_slag:.1f}",
            "Hedef": f"{target_slag:.1f}",
            "Fark": f"{diff_slag:+.1f}",
            "Tahmini Kazanç (€/t)": "Dolaylı"
        })

    # ---------- 6) Raw_Cr2O3_Percent – Cevher Kalitesi ----------
    # Örnek: 10% → 20% yapılırsa yaklaşık 40k€/heat kazanç → ton başına
    real_cr = 10.0
    target_cr = 20.0
    diff_cr = target_cr - real_cr
    gain_cr_per_t = 40000.0 / TYPICAL_HEAT_TON  # 40k€ / heat ≈ €/t
    total_gain_per_t += gain_cr_per_t

    rows.append({
        "Tag": "Raw_Cr2O3_Percent",
        "Değişken": "Cevher kalite farkı (Cr₂O₃)",
        "Gerçek": f"{real_cr:.1f} %",
        "Hedef": f"{target_cr:.1f} %",
        "Fark": f"{diff_cr:+.1f} %",
        "Tahmini Kazanç (€/t)": f"≈ {gain_cr_per_t:,.0f} €/t"
    })

    # ---------- Tabloyu oluştur ve göster ----------
    profit_df = pd.DataFrame(rows, columns=[
        "Tag",
        "Değişken",
        "Gerçek",
        "Hedef",
        "Fark",
        "Tahmini Kazanç (€/t)",
    ])

    st.dataframe(
        profit_df,
        use_container_width=True,
        hide_index=True,
    )

    # ---------- Toplam ton başına kazanç ----------
    st.markdown(
        f"**Toplam Potansiyel Kazanç (AI tahmini, ton başına):** "
        f"≈ **{total_gain_per_t:,.1f} €/t**"
    )
