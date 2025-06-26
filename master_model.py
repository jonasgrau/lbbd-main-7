import gurobipy as gp
from gurobipy import GRB
import model
import cuts
from typing import Dict

class MasterModel:
    def __init__(self, problem_instance: model.ProblemInstance):
        self.problem = problem_instance
        self.model = gp.Model("Master-LBBD")
        self.model.Params.OutputFlag = 0
        self._create_variables()
        self._create_initial_constraints()
        self.model.update()

    def _create_variables(self):
        self.theta = self.model.addVar(name="theta", vtype=GRB.CONTINUOUS, lb=0.0)
        self.z: Dict[tuple, gp.Var] = {
            (t_idx, o_idx, s_idx): self.model.addVar(vtype=GRB.BINARY, name=f"z_{t_idx}_{o_idx}_{s_idx}")
            for t_idx, train in enumerate(self.problem.trains)
            for o_idx, op in enumerate(train.operations) if len(op.successors) > 1
            for s_idx in op.successors
        }
        self.y: Dict[tuple, gp.Var] = {
            key: self.model.addVar(vtype=GRB.BINARY, name=f"y_{key[0]}_{key[1]}_{key[2]}_{key[3]}")
            for t1, o1, t2, o2 in (
                (c[0][0], c[0][1], c[1][0], c[1][1]) for c in self.problem.get_conflicts()
            ) for key in [(t1, o1, t2, o2), (t2, o2, t1, o1)]
        }

    def _create_initial_constraints(self):
        print("Erstelle initiale Master-Constraints...")
        # Path Pruning
        paths_pruned = 0
        for t_idx, train in enumerate(self.problem.trains):
            shortest_paths = train.get_shortest_paths_to_exit()
            exit_op_ub = train.operations[-1].start_ub
            if exit_op_ub == float('inf'): continue
            for o_idx, op in enumerate(train.operations):
                if len(op.successors) > 1:
                    for s_idx in op.successors:
                        earliest_finish = op.start_lb + op.min_duration + shortest_paths.get(s_idx, float('inf'))
                        if earliest_finish > exit_op_ub and (t_idx, o_idx, s_idx) in self.z:
                            self.model.addConstr(self.z[t_idx, o_idx, s_idx] == 0)
                            paths_pruned += 1
        if paths_pruned > 0: print(f"  -> {paths_pruned} unmögliche Pfade entfernt.")
        # Path Choice
        for t_idx, train in enumerate(self.problem.trains):
            for o_idx, op in enumerate(train.operations):
                if len(op.successors) > 1:
                    self.model.addConstr(gp.quicksum(self.z[t_idx, o_idx, s_idx] for s_idx in op.successors) == 1)
        # Ordering
        for (t1, o1), (t2, o2) in self.problem.get_conflicts():
            self.model.addConstr(self.y[t1, o1, t2, o2] + self.y[t2, o2, t1, o1] == 1)
        # 2-Train Swaps
        for c1, c2 in self.problem.get_2_train_swap_constraints():
            if c1 in self.y and c2 in self.y: self.model.addConstr(self.y[c1] == self.y[c2])
        # 3-Train Cycles
        for y1, y2, y3 in self.problem.get_3_train_cycle_constraints():
            if y1 in self.y and y2 in self.y and y3 in self.y: self.model.addConstr(self.y[y1] + self.y[y2] + self.y[y3] <= 2)
        self.model.setObjective(self.theta, GRB.MINIMIZE)

    def get_solution(self) -> Dict[gp.Var, float]:
        solution = {}
        for var in self.model.getVars():
            if var.VarName.startswith(('y', 'z')):
                try: solution[var] = var.X
                except gp.GurobiError: solution[var] = -1.0
        return solution

    def add_cut(self, cut: cuts.Cut):
        self.model.addConstr(cut.get_expr(self))