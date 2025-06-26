import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict

class Cut:
    """Basisklasse für alle Schnittgleichungen."""
    def add_to_model(self, model_instance, where):
        raise NotImplementedError

class FeasibilityCut(Cut):
    """Ein Feasibility-Cut (No-Good-Cut), generiert aus einem Konflikt."""
    def __init__(self, conflict_vars: List[gp.Var]):
        self.conflict_vars = conflict_vars

    def add_to_model(self, model_instance, where):
        """Fügt den Cut als Lazy Constraint hinzu."""
        expr = gp.quicksum(v for v in self.conflict_vars) <= len(self.conflict_vars) - 1
        model_instance.cbLazy(expr)

class OptimalityCut(Cut):
    """Ein Benders-Optimalitäts-Cut."""
    def __init__(self, objective_value: float, events: List[Dict], master_solution_vars: List[gp.Var]):
        self.objective_value = objective_value
        self.events = events
        # Speichert die Listen der aktiven (Wert 1) und inaktiven (Wert 0) Master-Variablen
        self.active_vars = [v for v, x in master_solution_vars.items() if x > 0.5]
        self.inactive_vars = [v for v, x in master_solution_vars.items() if x < 0.5]

    def add_to_model(self, model_instance, where):
        """Fügt den Cut als Lazy Constraint hinzu. KORRIGIERTE VERSION."""
        # Die Variablen sind bereits korrekt gespeichert, kein Nachschlagen nötig.
        deviation_expr = gp.quicksum(1 - v for v in self.active_vars) + gp.quicksum(v for v in self.inactive_vars)
        
        M = 1000000  # Eine sichere obere Schranke
        theta = model_instance._master_model.theta
        expr = theta >= self.objective_value - M * deviation_expr
        model_instance.cbLazy(expr)