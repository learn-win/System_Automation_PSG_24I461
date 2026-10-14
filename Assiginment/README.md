# Robotic Arm Joint Health Monitoring System

**23IF02 System Automation — Bonus Challenge (optional).**

An automated health monitor for a fleet of 6-axis robotic arms. Each arm has
three joints (shoulder, elbow, wrist) reporting **torque, temperature and
vibration**. The system fuses four independent signals into one decision per
arm — `SAFE`, `MONITOR`, `SCHEDULE_MAINTENANCE`, or `EMERGENCY_STOP`.

## The five requirements → where they live

| # | Requirement | Implementation |
|---|---|---|
| 1 | Immediate, deterministic safety stop | `check_safety()` — pure threshold check, **never** touches the model |
| 2 | Predictive wear detection | `train_wear_model()` / `predict_joint_wear()` — RandomForest on historical data |
| 3 | Maintenance notes as a second signal | `scan_maintenance_log()` — keyword scan of free-text log |
| 4 | Spoofed-sensor defence | `check_spoofing()` — torque vs. temperature/vibration physics check |
| 5 | One final decision per arm | `decide_for_arm()` — priority fusion of all four signals |

## Files

```
Assiginment/
├── joint_health_monitor.py        # main pipeline (run this)
├── generate_dataset.py            # recreates the historical dataset
├── EXPLANATION.md                 # the required written write-up
├── data/
│   ├── joint_failure_history.csv  # 60 labeled historical joint readings (Req. 2)
│   ├── current_fleet_readings.csv # live readings for the 6 arms
│   └── maintenance_log.txt        # 10 free-text notes, relevant + irrelevant (Req. 3)
├── joint_wear_model.pkl           # trained model (created on run)
└── fleet_decision_report.txt      # saved report (created on run)
```

## How to run

```bash
pip install -r ../requirements.txt scikit-learn joblib
python generate_dataset.py      # optional: regenerate the dataset
python joint_health_monitor.py  # train + evaluate the whole fleet
```

## All four outcomes in one run

The provided `current_fleet_readings.csv` is engineered so a single run reaches
every possible decision (the program prints `All four outcomes reached: YES`):

| Arm | Decision | Triggered by |
|---|---|---|
| Arm_1 | `SAFE` | all signals nominal |
| Arm_2 | `MONITOR` | model predicts developing wear at the wrist, no note |
| Arm_6 | `MONITOR` | technician note (wrist squeak) on otherwise healthy sensors |
| Arm_3 | `SCHEDULE_MAINTENANCE` | model predicts CRITICAL wear + corroborating grinding note |
| Arm_5 | `SCHEDULE_MAINTENANCE` | **spoofed torque** caught (hot + vibrating but low torque) |
| Arm_4 | `EMERGENCY_STOP` | shoulder temperature 92 °C > 85 °C hard limit (model-free) |

See [EXPLANATION.md](EXPLANATION.md) for the threshold justification, the
model+log decision logic, and the spoof-defence analysis.
