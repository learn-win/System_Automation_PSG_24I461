import random
import time

LOG_FILE = "day1_log.txt"


# STEP 1 — Generate simulated sensor data
def generate_sensor_data():
    """Return simulated temperature, pressure, vibration, and current."""
    return {
        "temperature": round(random.uniform(50, 95), 2),
        "pressure": round(random.uniform(20, 60), 2),
        "vibration": round(random.uniform(0.1, 2.0), 2),
        "current": round(random.uniform(5, 25), 2)
    }


# STEP 2 — Write automation rules
# STEP 7 — Add a rule of your own
# STEP 8 — Add priority rules
RULES = [
    {
        "name": "HIGH_TEMP",
        "priority": 1,
        "condition": lambda r: r["temperature"] > 80,
        "action": "STOP_MACHINE"
    },
    {
        "name": "HIGH_VIBRATION",
        "priority": 2,
        "condition": lambda r: r["vibration"] > 1.5,
        "action": "CHECK_MACHINE"
    },
    {
        "name": "HIGH_PRESSURE",
        "priority": 3,
        "condition": lambda r: r["pressure"] > 50,
        "action": "REDUCE_PRESSURE"
    },
    {
        "name": "HIGH_CURRENT",
        "priority": 4,
        "condition": lambda r: r["current"] > 20,
        "action": "CHECK_MOTOR"
    }
]


# STEP 3 — Detect machine state from rules
def evaluate_rules(reading, rules):
    """Return the first matching rule in priority order."""
    sorted_rules = sorted(rules, key=lambda r: r["priority"])

    for rule in sorted_rules:
        if rule["condition"](reading):
            return rule["name"], rule["action"]

    return "NO_RULE_FIRED", "CONTINUE_NORMAL"


# STEP 4 — Trigger an action
def take_action(action):
    """Return a message describing the action."""
    actions = {
        "STOP_MACHINE": "Machine stopped due to high temperature.",
        "CHECK_MACHINE": "Warning! Check machine vibration.",
        "REDUCE_PRESSURE": "Warning! Reduce machine pressure.",
        "CHECK_MOTOR": "Warning! Check motor current.",
        "CONTINUE_NORMAL": "Machine is operating normally."
    }

    return actions.get(action, f"No handler for action: {action}")


# STEP 5 — Log every decision
def log_decision(reading, rule_name, action):
    """Append the reading, rule, and action to the log file."""
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            f"Reading: {reading} | "
            f"Rule: {rule_name} | "
            f"Action: {action}\n"
        )


# STEP 6 — Add a second sensor: current
# The current sensor is included in generate_sensor_data()
# and HIGH_CURRENT is included in RULES.


def main():
    for tick in range(1, 11):
        reading = generate_sensor_data()
        rule_name, action = evaluate_rules(reading, RULES)
        outcome = take_action(action)

        print(f"[Tick {tick:02d}] SENSE -> {reading}")
        print(f"           THINK -> {rule_name}")
        print(f"           ACT   -> {outcome}\n")

        log_decision(reading, rule_name, action)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
