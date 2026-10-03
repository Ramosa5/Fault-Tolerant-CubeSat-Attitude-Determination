from __future__ import annotations
from dataclasses import dataclass

DETUMBLING = "DETUMBLING"
SUN_ACQUISITION = "SUN_ACQUISITION"
SUN_POINTING = "SUN_POINTING"
ECLIPSE_RATE_DAMPING = "ECLIPSE_RATE_DAMPING"
SUN_REACQUISITION = "SUN_REACQUISITION"

# Kept as an alias for older plots/scripts that may still import the old name.
ECLIPSE_DRIFT = ECLIPSE_RATE_DAMPING

MODE_ORDER = (DETUMBLING, SUN_ACQUISITION, SUN_POINTING,
              ECLIPSE_RATE_DAMPING, SUN_REACQUISITION)
MODE_CODE = {name: i for i, name in enumerate(MODE_ORDER)}


@dataclass
class SunPointingModeManager:
    """Operational mode manager for magnetorquer Sun pointing.

    A mode transition into nominal Sun pointing requires *both* pointing accuracy and
    sufficiently low body rate for a continuous hold interval.  This prevents a
    spacecraft from being declared acquired while it is merely flying through the
    target with substantial angular momentum.
    """
    threshold_deg: float = 2.0
    hold_s: float = 20.0
    dt_s: float = 0.1
    rate_threshold_deg_s: float = 0.05
    rate_hold_s: float = 10.0
    rate_recovery_trigger_deg_s: float = 0.20
    pointing_exit_threshold_deg: float | None = None
    mode: str = SUN_ACQUISITION
    _point_within_s: float = 0.0
    _rate_within_s: float = 0.0

    def _reset_holds(self):
        self._point_within_s = 0.0
        self._rate_within_s = 0.0

    def _rate_settled(self, rate_deg_s: float) -> bool:
        if float(rate_deg_s) <= self.rate_threshold_deg_s:
            self._rate_within_s += self.dt_s
        else:
            self._rate_within_s = 0.0
        return self._rate_within_s >= self.rate_hold_s

    def _pointing_and_rate_settled(self, pointing_deg: float, rate_deg_s: float | None) -> bool:
        if float(pointing_deg) <= self.threshold_deg:
            self._point_within_s += self.dt_s
        else:
            self._point_within_s = 0.0
        if rate_deg_s is None:
            # Backward compatibility for the older ideal-torque scenario, which did
            # not model an explicit rate-based mode transition.
            return self._point_within_s >= self.hold_s
        if float(rate_deg_s) <= self.rate_threshold_deg_s:
            self._rate_within_s += self.dt_s
        else:
            self._rate_within_s = 0.0
        return self._point_within_s >= self.hold_s and self._rate_within_s >= self.rate_hold_s

    def update(self, *, in_full_eclipse: bool, real_sun_available: bool,
               estimated_pointing_error_deg: float, estimated_rate_deg_s: float | None = None) -> str:
        # Full umbra always disables the Sun-pointing objective, but magnetic rate
        # damping remains active.
        if in_full_eclipse:
            if self.mode != ECLIPSE_RATE_DAMPING:
                self._reset_holds()
            self.mode = ECLIPSE_RATE_DAMPING
            return self.mode

        # Leaving eclipse: only begin Sun reacquisition once a real Sun measurement
        # is available again.
        if self.mode == ECLIPSE_RATE_DAMPING:
            if real_sun_available:
                self.mode = SUN_REACQUISITION
                self._reset_holds()
            return self.mode

        # Initial rate damping.  Pointing is intentionally ignored here.
        if self.mode == DETUMBLING:
            if estimated_rate_deg_s is not None and self._rate_settled(estimated_rate_deg_s):
                self.mode = SUN_ACQUISITION
                self._reset_holds()
            return self.mode

        # DETUMBLING is deployment-only.  After the initial release phase the
        # spacecraft never returns to DETUMBLING.  High rate during acquisition,
        # nominal pointing or post-eclipse reacquisition is handled by stronger
        # magnetic rate damping inside the active Sun-seeking mode.
        #
        # The scenario controller reads ``rate_recovery_trigger_deg_s`` and boosts
        # derivative/rate damping when this threshold is exceeded, while the state
        # machine remains in SUN_ACQUISITION or SUN_REACQUISITION.

        if self.mode == SUN_ACQUISITION:
            if self._pointing_and_rate_settled(estimated_pointing_error_deg, estimated_rate_deg_s):
                self.mode = SUN_POINTING
                self._reset_holds()
            return self.mode

        if self.mode == SUN_REACQUISITION:
            if not real_sun_available:
                self._reset_holds()
            elif self._pointing_and_rate_settled(estimated_pointing_error_deg, estimated_rate_deg_s):
                self.mode = SUN_POINTING
                self._reset_holds()
            return self.mode

        # SUN_POINTING.  If the spacecraft loses pointing badly in sunlight, return
        # to acquisition instead of silently remaining in the nominal mode.
        if self.mode == SUN_POINTING:
            exit_thr = float(self.pointing_exit_threshold_deg) if self.pointing_exit_threshold_deg is not None else max(3.0*self.threshold_deg, 5.0)
            if float(estimated_pointing_error_deg) > exit_thr:
                self.mode = SUN_ACQUISITION
                self._reset_holds()
            return self.mode

        return self.mode


def sun_pointing_control_enabled(mode: str) -> bool:
    return mode in (SUN_ACQUISITION, SUN_POINTING, SUN_REACQUISITION)


def magnetic_rate_damping_enabled(mode: str) -> bool:
    return mode in (DETUMBLING, ECLIPSE_RATE_DAMPING)


def control_enabled(mode: str) -> bool:
    """Backward-compatible alias for the Sun-pointing controller only.

    Older ideal-torque scenarios rely on this being false during eclipse.  The
    physical magnetorquer scenario calls ``magnetic_rate_damping_enabled``
    separately for DETUMBLING/ECLIPSE_RATE_DAMPING.
    """
    return sun_pointing_control_enabled(mode)
