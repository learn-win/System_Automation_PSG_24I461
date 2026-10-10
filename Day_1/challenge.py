
import random
import time

LOG_FILE = "day1_log.txt"


# STEP 1: Generate sensor data for a machine
def generate_sensor_data():
    return {
        "temperature": round(random.uniform(50, 95), 2),
        "pressure": round(random.uniform(20, 60), 2),
        "vibration": round(random.uniform(0.1, 2.0), 2),
        "current": round(random.uniform(5, 25), 2)
    }


# STEP 2: Define machine rules
RULES = [
    {
        "name": "HIGH_TEMP",
        "priority": 1,
        "condition": lambda r: r["temperature"] > 80,
        "action": "STOP_MACHINE",
        "severity": 3
    },
    {
        "name": "HIGH_VIBRATION",
        "priority": 2,
        "condition": lambda r: r["vibration"] > 1.5,
        "action": "CHECK_MACHINE",
        "severity": 2
    },
    {
        "name": "HIGH_PRESSURE",
        "priority": 3,
        "condition": lambda r: r["pressure"] > 50,
        "action": "REDUCE_PRESSURE",
        "severity": 2
    },
    {
        "name": "HIGH_CURRENT",
        "priority": 4,
        "condition": lambda r: r["current"] > 20,
        "action": "CHECK_MOTOR",
        "severity": 2
    }
]


# STEP 3: Evaluate rules for one machine
def evaluate_rules(reading, rules):
    matching_rules = []

    # Check all rules
    for rule in sorted(rules, key=lambda r: r["priority"]):
        if rule["condition"](reading):
            matching_rules.append(rule)

    # No problem found
    if not matching_rules:
        return "NO_RULE_FIRED", "CONTINUE_NORMAL", 0

    # Find the most severe problem for this machine
    worst_rule = max(
        matching_rules,
        key=lambda rule: (rule["severity"], -rule["priority"])
    )

    return (
        worst_rule["name"],
        worst_rule["action"],
        worst_rule["severity"]
    )


# STEP 4: Describe the required action
def take_action(action):
    actions = {
        "STOP_MACHINE": "Stop machine and inspect temperature.",
        "CHECK_MACHINE": "Inspect machine vibration.",
        "REDUCE_PRESSURE": "Check and reduce excessive pressure.",
        "CHECK_MOTOR": "Inspect motor current.",
        "CONTINUE_NORMAL": "Machine is operating normally."
    }

    return actions.get(action, "No action defined.")


# STEP 5: Log every machine decision
def log_decision(machine, reading, rule_name, action, severity):
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            f"{machine} | Reading={reading} | "
            f"Problem={rule_name} | Severity={severity} | "
            f"Action={action}\n"
        )


# STEP 6: Simulate five machines
def main():
    machines = [
        "Machine 1",
        "Machine 2",
        "Machine 3",
        "Machine 4",
        "Machine 5"
    ]

    results = []

    print("=" * 55)
    print("       FIVE-MACHINE RULE ENGINE SIMULATION")
    print("=" * 55)

    for machine in machines:
        reading = generate_sensor_data()

        rule_name, action, severity = evaluate_rules(
            reading, RULES
        )

        outcome = take_action(action)

        print(f"\n{machine}")
        print(f"Sensor Data : {reading}")
        print(f"Problem     : {rule_name}")
        print(f"Severity    : {severity}")
        print(f"Action      : {outcome}")

        results.append({
            "machine": machine,
            "reading": reading,
            "problem": rule_name,
            "action": action,
            "severity": severity
        })

        log_decision(
            machine, reading, rule_name, action, severity
        )

        time.sleep(0.2)

    # STEP 7: Find the machine needing immediate attention
    print("\n" + "=" * 55)
    print("             FINAL ACTION SUMMARY")
    print("=" * 55)

    # Higher severity first; machine number breaks ties
    most_urgent = max(
        results,
        key=lambda item: (
            item["severity"],
            -int(item["machine"].split()[-1])
        )
    )

    urgent_machines = [
        item for item in results
        if item["severity"] == most_urgent["severity"]
        and item["severity"] > 0
    ]

    if most_urgent["severity"] == 0:
        print("All five machines are operating normally.")
        print("No immediate action is required.")
    else:
        print("IMMEDIATE ATTENTION: " + most_urgent["machine"])
        print("Detected problem : " + most_urgent["problem"])
        print("Required action  : " + most_urgent["action"])
        print("Recommendation   : " + take_action(most_urgent["action"]))

        if len(urgent_machines) > 1:
            print("\nOther machines with the same severity:")
            for item in urgent_machines:
                if item["machine"] != most_urgent["machine"]:
                    print(
                        f"- {item['machine']}: {item['problem']}"
                    )

    print("\nAll machine results:")
    for item in results:
        status = (
            "ACTION REQUIRED"
            if item["severity"] > 0
            else "NORMAL"
        )
        print(
            f"{item['machine']}: {item['problem']} -> {status}"
        )

    print(f"\nLog saved to: {LOG_FILE}")
    print("=" * 55)


if __name__ == "__main__":
    main()
