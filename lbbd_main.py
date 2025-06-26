import time
import json
import os
import argparse
import gurobipy as gp
from gurobipy import GRB

from model import ProblemInstance
from master_model import MasterModel
from subproblem_gurobi import SubproblemGurobi
from cuts import FeasibilityCut, OptimalityCut

# Globale Variablen für den Callback, um die beste Lösung zu speichern
best_obj = float('inf')
best_solution_events = None

def benders_callback(model, where):
    """Dieser Callback wird von Gurobi aufgerufen, wenn eine neue Master-Lösung gefunden wurde."""
    global best_obj, best_solution_events

    if where == GRB.Callback.MIPSOL:
        # 1. Hole die aktuelle Master-Lösung (Variablen-Objekte -> Werte)
        master_vars_map = model._master_model.y | model._master_model.z
        master_solution = {var_obj: model.cbGetSolution(var_obj) for var_obj in master_vars_map.values()}

        # 2. Löse das Subproblem mit dieser Lösung
        subproblem = SubproblemGurobi(model._problem, master_solution, model._master_model)
        cut = subproblem.solve()

        # 3. Füge den entsprechenden Cut als Lazy Constraint hinzu
        if isinstance(cut, OptimalityCut):
            if cut.objective_value < best_obj:
                best_obj = cut.objective_value
                best_solution_events = cut.events
        
        # Das Cut-Objekt weiß selbst, wie es sich dem Modell hinzufügt
        cut.add_to_model(model, where)


def solve_instance(instance_path: str, time_limit: int):
    """Löst die Instanz mit einem Branch-and-Cut-Ansatz."""
    global best_obj, best_solution_events
    best_obj = float('inf') # Reset für jeden Lauf
    best_solution_events = None

    print(f"--- Starte Branch-and-Cut Solver für: {instance_path} ---")
    problem = ProblemInstance(instance_path)
    master = MasterModel(problem)

    master.model.Params.LazyConstraints = 1
    master.model.setParam('TimeLimit', time_limit)

    # Übergebe die notwendigen Objekte an den Callback via "private" Attribute
    master.model._problem = problem
    master.model._master_model = master

    # Starte die Optimierung mit dem Callback
    master.model.optimize(benders_callback)

    print("-----------------------------------------------------------------")
    print(f"Optimierung beendet. Laufzeit: {master.model.Runtime:.2f} Sekunden.")
    
    if best_solution_events:
        print(f"\nBeste gefundene Lösung mit Zielfunktionswert: {best_obj:.2f}")
        # Speichern der Lösung...
        output_dir = "solutions"
        os.makedirs(output_dir, exist_ok=True)
        solution_path = os.path.join(output_dir, f"solution_{os.path.basename(instance_path)}")
        sorted_events = sorted(best_solution_events, key=lambda e: (e["time"], e["train"], e["operation"]))
        with open(solution_path, 'w') as f:
            json.dump({"objective_value": int(round(best_obj)), "events": sorted_events}, f, indent=4)
        print(f"✓ Beste Lösung in '{solution_path}' gespeichert.")
    else:
        print("\nKeine zulässige Lösung innerhalb der Limits gefunden.")


if __name__ == "__main__":
    # Dieser Teil bleibt wie von Ihnen gewünscht, um den Pfad direkt im Code zu setzen
    instance_file_to_run = "data/displib_instances_phase1/line1_critical_0.json"
    TIME_LIMIT_SECONDS = 600
    
    solve_instance(instance_file_to_run, TIME_LIMIT_SECONDS)