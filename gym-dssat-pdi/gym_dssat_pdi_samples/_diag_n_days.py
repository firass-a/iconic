from capql_inference import load_actor, run_season

actor = load_actor()
for name, w in [
    ('balanced', [1 / 3, 1 / 3, 1 / 3]),
    ('max_yield', [1, 0, 0]),
    ('n_eff', [0, 1, 0]),
]:
    r = run_season(w, seed=123, actor=actor, weather_id='wgen-nasa-algiers')
    nz = [d for d in r.days if d.nitrogen_kg_ha > 0.5]
    top = sorted(r.days, key=lambda d: d.nitrogen_kg_ha, reverse=True)[:5]
    print(
        f"{name}: total_N={r.total_n_kg_ha:.1f} kg/ha, "
        f"days_with_N={len(nz)}/{len(r.days)}"
    )
    print(f"  top: {[(d.day, d.dap, round(d.nitrogen_kg_ha, 1)) for d in top]}")
