use std::path::Path;

use serde_json::{json, Value};
use sidereon_core::astro::frames::orientation::TdbEarthOrientationProvider;
use sidereon_core::astro::propagator::{ForceModelKind, IntegratorKind};
use sidereon_core::ephemeris::Sp3;
use sidereon_core::observables::{
    emission_media_batch_at_j2000_s, EmissionMediaBatchOptions, ObservableTroposphereCorrection,
};
use sidereon_core::orbit_determination::{fit_sp3_ecef_precise_orbit, OrbitFitOptions};

pub fn sections(core: &Path, fixtures: &Path) -> Vec<(&'static str, Value)> {
    let source = super::load_sp3(core, "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3");
    let satellites =
        ["G08", "G10", "G16", "S20"].map(|satellite| satellite.parse().expect("satellite"));
    let epochs = [source.epochs_j2000_seconds()[20]; 4];
    let mut options = EmissionMediaBatchOptions::default();
    options.media.troposphere = Some(ObservableTroposphereCorrection::default());
    options.min_elevation_rad = Some(0.0);
    let batch = emission_media_batch_at_j2000_s(
        &source,
        &satellites,
        &epochs,
        [4_484_127.99232578, 550_581.68657014, 4_487_560.54090027],
        options,
    )
    .expect("emission batch");
    let emission = json!({
        "positions_ecef_m": batch.positions_ecef_m.iter()
            .map(|position| position.as_ref().map(|position| super::hexes(position)))
            .collect::<Vec<_>>(),
        "clocks_s": batch.clocks_s.iter().map(|clock| clock.map(super::hex)).collect::<Vec<_>>(),
        "troposphere_delays_m": batch.troposphere_delays_m.iter()
            .map(|delay| delay.map(super::hex)).collect::<Vec<_>>(),
    });
    let orbit_source = Sp3::parse(
        &std::fs::read(fixtures.join("sp3/IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"))
            .expect("read orbit product"),
    )
    .expect("parse orbit product");
    let mut options = OrbitFitOptions::default();
    options.force_model = ForceModelKind::two_body();
    options.integrator = IntegratorKind::Dp54;
    options.integrator_options.abs_tol = 1.0e-9;
    options.integrator_options.rel_tol = 1.0e-12;
    options.integrator_options.initial_step = 60.0;
    options.integrator_options.min_step = 1.0e-6;
    options.integrator_options.max_step = 3600.0;
    options.integrator_options.max_steps = 200_000;
    options.min_ledger_samples = 3;
    let satellite = "G01".parse().expect("satellite");
    let orbit = fit_sp3_ecef_precise_orbit(
        &orbit_source,
        satellite,
        &TdbEarthOrientationProvider::new(),
        &options,
    )
    .expect("orbit fit");
    vec![
        ("emission_media", emission),
        (
            "precise_orbit_fit",
            json!({
                "fit_rms_3d_m": super::hex(orbit.fits[&satellite].fit_rms_3d_m),
                "ledger_rms_3d_m": super::hex(orbit.ledger.per_sat[&satellite].rms_3d_m),
            }),
        ),
    ]
}
