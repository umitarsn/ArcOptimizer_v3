    # -----------------------------
    # Proses Kazanç Analizi (Ton Başına)
    # -----------------------------
    ELECTRICITY_PRICE_EUR_PER_MWH = 50.0   # 50 €/MWh  => 0.05 €/kWh
    ELECTRODE_PRICE_EUR_PER_KG = 3.0       # örnek: 3 €/kg

    profit_rows = []

    # 1) Enerji tüketimi (kwh_per_t) - zaten vardı, dokunmuyoruz
    if pd.notna(last.get("kwh_per_t", None)) and not pd.isna(avg_kwh_t):
        real_kwh_t = float(last["kwh_per_t"])
        target_kwh_t = avg_kwh_t  # potansiyel için basit yaklaşım: son 10 ort.
        diff_kwh_t = real_kwh_t - target_kwh_t  # pozitif = iyileştirme alanı

        if diff_kwh_t > 0:
            gain_eur_t = diff_kwh_t * (ELECTRICITY_PRICE_EUR_PER_MWH / 1000.0)
        else:
            gain_eur_t = 0.0

        profit_rows.append(
            {
                "tag": "kwh_per_t",
                "deg": "Enerji tüketimi",
                "akt": f"{real_kwh_t:.1f} kWh/t",
                "pot": f"{target_kwh_t:.1f} kWh/t",
                "fark": f"{diff_kwh_t:+.1f} kWh/t",
                "kazanc": f"{gain_eur_t:.2f} €/t",
                "type": "cost",
            }
        )

    # 2) Elektrot tüketimi (DAİMA daha iyi veya en kötü eşit)
    tap_w = float(last.get("tap_weight_t", 0.0) or 0.0)
    if tap_w > 0 and pd.notna(last.get("electrode_kg_per_heat", None)):
        real_elec_pt = float(last["electrode_kg_per_heat"]) / tap_w  # kg/t

        # Ortalama varsa: hedef = min(aktüel, ortalama)
        if pd.notna(avg_electrode):
            avg_elec_pt = float(avg_electrode) / tap_w
            target_elec_pt = min(real_elec_pt, avg_elec_pt)
        else:
            # Ortalama yoksa, hafif iyileştirme hedefi ama asla aktüelden kötü değil
            target_elec_pt = max(real_elec_pt - 0.003, 0.0)

        # Eğer zaten hedeften iyi ise kazanç = 0, fark = 0 göster
        if real_elec_pt > target_elec_pt:
            diff_elec_pt = real_elec_pt - target_elec_pt
            gain_elec_eur_t = diff_elec_pt * ELECTRODE_PRICE_EUR_PER_KG
        else:
            # zaten ortalamadan iyi → ilave iyileştirme alanı yok
            diff_elec_pt = 0.0
            gain_elec_eur_t = 0.0
            target_elec_pt = real_elec_pt  # potansiyel en kötü eşit kalsın

        profit_rows.append(
            {
                "tag": "electrode",
                "deg": "Elektrot tüketimi",
                "akt": f"{real_elec_pt:.3f} kg/t",
                "pot": f"{target_elec_pt:.3f} kg/t",
                "fark": f"{diff_elec_pt:+.3f} kg/t",
                "kazanc": f"{gain_elec_eur_t:.2f} €/t",
                "type": "cost",
            }
        )
