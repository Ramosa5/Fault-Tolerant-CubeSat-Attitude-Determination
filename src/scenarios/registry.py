from . import sun_pointing, sun_pointing_comparison, magnetic_sun_pointing_comparison, magnetic_control_orbit_matrix, reaction_wheel_sun_pointing_comparison, reaction_wheel_fault_campaign
SCENARIOS={'sun_pointing':sun_pointing.run,'sun_pointing_comparison':sun_pointing_comparison.run,'magnetic_sun_pointing_comparison':magnetic_sun_pointing_comparison.run,'magnetic_control_orbit_matrix':magnetic_control_orbit_matrix.run,'reaction_wheel_sun_pointing_comparison':reaction_wheel_sun_pointing_comparison.run,'reaction_wheel_fault_campaign':reaction_wheel_fault_campaign.run}

def run_scenario(name,cfg,global_cfg,out_dir):
    key=str(name).strip().lower()
    if key not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Available scenarios: {', '.join(sorted(SCENARIOS))}")
    return SCENARIOS[key](cfg,global_cfg,out_dir)
