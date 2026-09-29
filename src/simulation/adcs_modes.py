from __future__ import annotations
from dataclasses import dataclass

SUN_ACQUISITION = "SUN_ACQUISITION"
SUN_POINTING = "SUN_POINTING"
ECLIPSE_DRIFT = "ECLIPSE_DRIFT"
SUN_REACQUISITION = "SUN_REACQUISITION"

MODE_ORDER = (SUN_ACQUISITION, SUN_POINTING, ECLIPSE_DRIFT, SUN_REACQUISITION)
MODE_CODE = {name: i for i, name in enumerate(MODE_ORDER)}


@dataclass
class SunPointingModeManager:
    """Small deterministic mode manager for the Sun-pointing scenario.

    Transitions use the estimated pointing error, because that is what an onboard
    controller would actually have available. Full umbra always forces ECLIPSE_DRIFT.
    After eclipse, reacquisition begins only when a real Sun-sensor measurement is
    available again. The controller may then settle back into SUN_POINTING.
    """
    threshold_deg: float = 1.0
    hold_s: float = 10.0
    dt_s: float = 0.1
    mode: str = SUN_ACQUISITION
    _within_s: float = 0.0

    def _settled(self, estimated_pointing_error_deg: float) -> bool:
        if estimated_pointing_error_deg <= self.threshold_deg:
            self._within_s += self.dt_s
        else:
            self._within_s = 0.0
        return self._within_s >= self.hold_s

    def update(self, *, in_full_eclipse: bool, real_sun_available: bool,
               estimated_pointing_error_deg: float) -> str:
        # Physical eclipse has top priority.
        if in_full_eclipse:
            self.mode = ECLIPSE_DRIFT
            self._within_s = 0.0
            return self.mode

        if self.mode == ECLIPSE_DRIFT:
            # Reacquire only after the physical Sun sensor sees the Sun again.
            if real_sun_available:
                self.mode = SUN_REACQUISITION
                self._within_s = 0.0
            return self.mode

        if self.mode == SUN_ACQUISITION:
            if self._settled(estimated_pointing_error_deg):
                self.mode = SUN_POINTING
            return self.mode

        if self.mode == SUN_REACQUISITION:
            if not real_sun_available:
                # Sun is physically visible but unavailable to the sensor (e.g. fault):
                # remain in reacquisition rather than claiming nominal pointing mode.
                self._within_s = 0.0
            elif self._settled(estimated_pointing_error_deg):
                self.mode = SUN_POINTING
            return self.mode

        # SUN_POINTING: sunlight continues; large errors do not by themselves create
        # a different mode, but eclipse will always transition on the next call.
        self._within_s = 0.0
        return self.mode


def control_enabled(mode: str) -> bool:
    """Sun-pointing torque is disabled only in eclipse drift mode."""
    return mode != ECLIPSE_DRIFT
