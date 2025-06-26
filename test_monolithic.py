import gurobipy as gp
from gurobipy import GRB

# KORREKTUR: Importiere jetzt aus 'utils.py'
from utils import parse_displib_instance, build_monolithic_model, save_solution

def solve_monolithic_fully(instance_path: str, time_limit: int):
    """
    Versucht, eine Instanz als einzelnes, großes MIP-Modell zu lösen.
    """
    print(f"--- Starte monolithischen Lösungsversuch für: {instance_path} ---")
    problem = parse_displib_instance(instance_path)
    
    # Baue das monolithische Modell
    mono_model, x_vars = build_monolithic_model(problem)

    # Setze das Zeitlimit und aktiviere die Gurobi-Ausgabe
    mono_model.Params.TimeLimit = time_limit
    mono_model.Params.OutputFlag = 1

    # Starte die Optimierung
    mono_model.optimize()

    # --- Ergebnisse auswerten ---
    print("----------------------------------------------------------")
    print(f"Monolithische Optimierung beendet. Gurobi-Status: {mono_model.Status}")

    if mono_model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and mono_model.SolCount > 0:
        print(f"\n✓ Lösung gefunden! Zielfunktionswert: {mono_model.ObjVal:.2f}")
        events = [{'train': t, 'operation': o, 'time': int(round(x.X))} for (t, o), x in x_vars.items()]
        save_solution(instance_path.replace(".json", "_mono.json"), mono_model.ObjVal, events)
    else:
        print("\nKeine zulässige Lösung innerhalb des Zeitlimits gefunden.")

if __name__ == "__main__":
    INSTANCE_TO_TEST = "data/displib_instances_phase1/line1_critical_0.json"
    TIME_LIMIT_SECONDS = 300
    solve_monolithic_fully(INSTANCE_TO_TEST, TIME_LIMIT_SECONDS)