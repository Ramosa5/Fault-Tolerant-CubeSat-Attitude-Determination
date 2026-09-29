from . import sun_pointing, sun_pointing_comparison, magnetic_sun_pointing_comparison
SCENARIOS={'sun_pointing':sun_pointing.run,'sun_pointing_comparison':sun_pointing_comparison.run,'magnetic_sun_pointing_comparison':magnetic_sun_pointing_comparison.run}

def run_scenario(name,cfg,global_cfg,out_dir):
    key=str(name).strip().lower()
    if key not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Available scenarios: {', '.join(sorted(SCENARIOS))}")
    return SCENARIOS[key](cfg,global_cfg,out_dir)
