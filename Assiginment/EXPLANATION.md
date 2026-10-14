# Robotic Arm Joint Health Monitoring — Design Explanation

*23IF02 System Automation · Bonus Challenge*

## 1. Why these immediate-stop thresholds

The hard limits are **torque > 130 Nm, temperature > 85 °C, vibration > 8 mm/s RMS**
(`SAFETY_LIMITS` in `joint_health_monitor.py`). They are deliberately set *above*
the worst "CRITICAL wear" readings in the training data, which top out around
125 Nm / 84 °C / 7.9 mm/s. That gap is the whole design: the model gets to warn
us in the band between *worn* and *dangerous*, while the hard check only fires on
genuine physical danger.

- **Temperature 85 °C** — industrial servo motors and harmonic-drive grease begin
  to degrade well before the ~90 °C thermal cutoff; 85 °C leaves a safety margin
  before insulation and lubricant damage.
- **Vibration 8 mm/s** — ISO 10816 places sustained vibration above roughly
  7.1 mm/s in the "unacceptable" severity zone for this class of machine, so 8 is
  a defensible hard ceiling.
- **Torque 130 Nm** — represents overload beyond the actuator/gearing rating,
  where tooth or bearing damage becomes likely under sustained load.

This check is pure arithmetic over the raw readings and **never calls the model**,
so it still protects the line if the model is wrong, stale, or missing entirely.

## 2. How the model and the maintenance log combine

Each arm produces four independent signals; they are fused in strict priority
order (`decide_for_arm`):

1. **Deterministic safety** → if any joint crosses a hard limit, `EMERGENCY_STOP`
   immediately, before anything else runs.
2. **Sensor integrity** → if a spoof is suspected (see §3), torque is untrusted.
3. **Predictive wear** → a RandomForest trained on `joint_failure_history.csv`
   classifies each joint `HEALTHY / WEAR / CRITICAL`; the arm inherits its
   **worst** joint.
4. **Maintenance notes** → a keyword scan of the free-text log yields a per-arm
   concern flag.

The model and the log then combine as a **corroboration ladder**:

| Model (worst joint) | Technician note? | Decision |
|---|---|---|
| CRITICAL (conf ≥ 60%) | either | `SCHEDULE_MAINTENANCE` |
| WEAR | **yes** | `SCHEDULE_MAINTENANCE` (note promotes it) |
| WEAR / CRITICAL (low conf) | — | `MONITOR` |
| HEALTHY | **yes** | `MONITOR` (note alone is enough to watch) |
| HEALTHY | no | `SAFE` |

The log is a genuinely *independent* second opinion: a human-reported concern can
pull a sensor-healthy arm up to `MONITOR` (Arm 6, a wrist squeak), and it upgrades
borderline `WEAR` straight to maintenance. Housekeeping lines (lights, coffee,
paint) contain no mechanical keyword and are dropped, so the log never raises a
false alarm on its own.

## 3. The spoofed-sensor defence

**The attack:** someone feeds a *fake low torque* reading to make an overloaded
joint look lightly loaded, hiding a developing failure from the model.

**The defence (`check_spoofing`):** heat and vibration are physical *side effects*
of real mechanical load — an attacker who clamps the torque value cannot also
stop the joint from getting hot and shaking. So if a joint is clearly hot
(≥ 68 °C) **and** vibrating hard (≥ 5.5 mm/s) **but** its torque reads low
(< 60 Nm), the torque channel contradicts physics and is flagged as a probable
spoof. A plausibility bound (torque outside 0–200 Nm) catches crude garbage/replay
values too. When a spoof is flagged we stop trusting torque and fall back to the
channels we *can* trust: if temperature or vibration are themselves over a hard
limit → `EMERGENCY_STOP`; otherwise → `SCHEDULE_MAINTENANCE` for a physical
inspection (Arm 5 in the demo).

**What it catches:** single-channel torque tampering — the most direct way to mask
load — because the independent thermal and vibration channels still betray the
real stress.

**What it would *not* catch:** a *coordinated* spoof that fakes torque **and**
temperature **and** vibration together into one physically consistent (but false)
picture, or a spoof that keeps every value inside a plausible joint region. It
also can't distinguish a spoof from a genuinely faulty torque sensor — but the
response (don't trust it, inspect physically) is the correct, safe action for both.

## Note on model accuracy

The reported test accuracy is ~100% because the synthetic classes are cleanly
separable by construction. On messier real-world data the same pipeline would see
lower, more realistic numbers; the architecture (confidence gating, human-note
corroboration, model-independent safety) is what matters, not the headline score.
