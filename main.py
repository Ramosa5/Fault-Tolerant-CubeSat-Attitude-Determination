from __future__ import annotations

import argparse
from pathlib import Path

from src.simulation.pipeline import build_dataframe, load_config, save_outputs
from src.visualization.dashboard import build_dashboard
from src.visualization.static_plots import generate_static_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="CubeSat FDIR thesis — Phase 1 orbit/environment simulation")
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = root / args.config
    config = load_config(config_path)

    df, summary = build_dataframe(config)
    save_outputs(df, summary, config, root)

    results_dir = root / config["output"]["results_dir"]
    if config["visualization"].get("make_static_png", True):
        generate_static_plots(df, results_dir, int(config["visualization"].get("dpi", 160)))
    if config["visualization"].get("make_interactive_html", True):
        build_dashboard(df, results_dir / "phase1_dashboard.html")

    print("\nCubeSat FDIR thesis — Phase 1 complete")
    print("--------------------------------------")
    print(f"Samples:                  {summary['samples']}")
    print(f"Kepler period:            {summary['period_min']:.3f} min")
    print(f"Mean altitude:            {summary['altitude_mean_km']:.3f} km")
    print(f"Altitude range:           {summary['altitude_min_km']:.3f} .. {summary['altitude_max_km']:.3f} km")
    print(f"Mean speed:               {summary['speed_mean_km_s']:.4f} km/s")
    print(f"Eclipse fraction:         {100*summary['eclipse_fraction']:.1f} %")
    print(f"Magnetic field range:     {summary['mag_min_uT']:.2f} .. {summary['mag_max_uT']:.2f} µT")
    print(f"Sun–B angle range:        {summary['sun_mag_angle_min_deg']:.1f} .. {summary['sun_mag_angle_max_deg']:.1f} deg")
    print(f"CSV:                      {config['output']['csv']}")
    print(f"Dashboard:                {config['output']['results_dir']}/phase1_dashboard.html")


if __name__ == "__main__":
    main()
