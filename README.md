# System Automation — Predictive Maintenance & Machine Fault Detection

**Course:** 23IF02 · System Automation
**Student ID:** PSG_24I461

A hands-on workshop project that builds a machine fault-detection / predictive
maintenance system in stages — from a simple deterministic rule engine (Day 1),
to trained machine-learning models (Day 2), to a full multi-signal monitoring
system that fuses rules, ML, free-text logs and sensor-spoof defence (Bonus
Challenge).

---

## Repository layout

```
System_Automation_PSG_24I461/
├── README.md               # this file
├── requirements.txt        # pinned Python dependencies
├── Day_1/                  # rule-based automation (Sense → Think → Act)
├── Day_2/                  # machine learning (Decision Tree & Random Forest)
└── Assiginment/            # Bonus Challenge — Robotic Arm Joint Health Monitor
```

---

## Day 1 — Rule-based automation

A classic **Sense → Think → Act** loop. Simulated sensors (temperature,
pressure, vibration, current) are evaluated against priority-ordered rules, and
every decision is logged.

| File | What it does |
|---|---|
| [hands_on_starter.py](Day_1/hands_on_starter.py) | Minimal teaching skeleton — two temperature rules |
| [Base.py](Day_1/Base.py) | Single machine, 4 prioritised rules, first-match logic, logs to `day1_log.txt` |
| [challenge.py](Day_1/challenge.py) | Five machines, severity-ranked rules, picks the most urgent machine |

```bash
cd Day_1
python Base.py
python challenge.py
```

---

## Day 2 — Machine learning

Learns machine condition from a labelled sensor dataset. Features:
`temperature, pressure, vibration, current, rpm` → target `state`. All scripts
share a robust loader (column normalisation, numeric coercion, invalid-row
filtering, stratified-split guards).

| File | Model | Outputs |
|---|---|---|
| [model_predictive_maintenance_Decision_tree.py](Day_2/model_predictive_maintenance_Decision_tree.py) | Decision Tree (depth 5) | Tree image, exported rules, confusion matrix |
| [model_predictive_maintenance.py](Day_2/model_predictive_maintenance.py) | Random Forest (200 trees) | Feature importance, example tree, confusion matrix, saved model |
| [combined.py](Day_2/combined.py) | Rules **+** Random Forest | Side-by-side comparison for 5 machines, agreement table, detailed log |
| [generate.py](Day_2/generate.py) | — | Synthetic sensor-data generator |

Dataset: [Day_2/data/day2_sample.csv](Day_2/data/day2_sample.csv) (250 labelled rows).

```bash
cd Day_2
python model_predictive_maintenance.py           # Random Forest
python model_predictive_maintenance_Decision_tree.py
python combined.py                                # hybrid rules + ML
```

---

## Bonus Challenge — Robotic Arm Joint Health Monitoring

A fleet of 6-axis robotic arms (shoulder / elbow / wrist joints, each reporting
torque, temperature and vibration). The system fuses **four independent signals**
into one decision per arm: `SAFE`, `MONITOR`, `SCHEDULE_MAINTENANCE`, or
`EMERGENCY_STOP`.

1. **Deterministic safety stop** — pure threshold check, never uses the model.
2. **Predictive wear** — Random Forest trained on self-made failure history.
3. **Maintenance notes** — keyword scan of a free-text technician log.
4. **Spoofed-sensor defence** — physics check that catches a faked torque reading.

```bash
cd Assiginment
python joint_health_monitor.py
```

Full details, run map and the design write-up are in
[Assiginment/README.md](Assiginment/README.md) and
[Assiginment/EXPLANATION.md](Assiginment/EXPLANATION.md).

---

## Setup

Requires **Python 3.10+** (developed on 3.14).

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins `scikit-learn==1.4.2`, which has no wheels
> for Python 3.14. On 3.14, install the current release instead:
> `pip install scikit-learn joblib` (tested with scikit-learn 1.9.1).

### Dependencies

`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`,
`jupyter`/`notebook`, `streamlit`.

---

## Concept progression

| Stage | Core idea |
|---|---|
| Day 1 | Deterministic rules, priority, severity, logging |
| Day 2 | Supervised learning, model evaluation, rule-vs-ML comparison |
| Bonus | Multi-signal fusion, log scanning, sensor-integrity / security validation |
