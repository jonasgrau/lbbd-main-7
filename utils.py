import json
import os
import gurobipy as gp
from gurobipy import GRB
from model import Operation, Train, DisplibInstance

def parse_displib_instance(path: str) -> DisplibInstance:
    """Parse a DISPLIB JSON instance file."""
    # This function remains unchanged
    with open(path) as f:
        raw = json.load(f)
    trains = []
    for t_id, op_list in enumerate(raw["trains"]):
        ops = []
        for o_id, op in enumerate(op_list):
            ops.append(
                Operation(
                    train_id=t_id, op_id=o_id, min_duration=op["min_duration"],
                    resources=op.get("resources", []), successors=op.get("successors", []),
                    start_lb=op.get("start_lb"), start_ub=op.get("start_ub"),
                )
            )
        trains.append(Train(train_id=t_id, operations=ops))
    objectives = raw.get("objective", [])
    return DisplibInstance(trains=trains, objectives=objectives)


def instance_to_data(instance: DisplibInstance) -> dict:
    """Convert a DisplibInstance to the data dictionary used by the models."""
    # This function remains unchanged
    return {
        "trains": list(range(len(instance.trains))), "train_objects": instance.trains,
        "paths": instance.generate_paths_dict(),
        "durations": { str((op.train_id, op.op_id)): op.min_duration for train in instance.trains for op in train.operations },
        "conflicts": instance.get_conflicts(), "no_swaps": instance.get_no_swap_pairs(),
        "objective": instance.objectives,
    }

# ====================================================================
# === NEUE, VERSCHOBENE FUNKTIONEN ===
# ====================================================================

def save_solution(instance_path: str, objective_value: float, events: list):
    """Speichert die gefundene Lösung im DISPLIB JSON-Format."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    solution_path = os.path.join(output_dir, f"solution_{os.path.basename(instance_path)}")
    sorted_events = sorted(events, key=lambda e: (e["time"], e["train"], e["operation"]))
    with open(solution_path, 'w') as f:
        json.dump({"objective_value": int(round(objective_value)), "events": sorted_events}, f, indent=4)
    print(f"\n✓ Lösung in '{solution_path}' gespeichert.")


def build_monolithic_model(problem: DisplibInstance):
    """Baut ein einzelnes, großes Gurobi-Modell für einen Lösungsversuch."""
    print("Baue monolithisches Modell...")
    mono_model = gp.Model("Monolithic")
    mono_model.Params.OutputFlag = 0
    
    x = mono_model.addVars(
        ((train.train_id, op.op_id) for train in problem.trains for op in train.operations),
        vtype=GRB.INTEGER, name="x"
    )
    
    BIG_M = 200000
    for train in problem.trains:
        for op in train.operations:
            # Hier müsste die vollständige Logik zum Bauen des monolithischen Modells implementiert werden,
            # inklusive Pfadwahl (z-Variablen), Reihenfolge (y-Variablen) und aller Constraints.
            # Dies ist eine sehr komplexe Aufgabe.
            pass # Placeholder
            
    # Placeholder für Zielfunktion
    mono_model.setObjective(0, GRB.MINIMIZE)
    return mono_model, x