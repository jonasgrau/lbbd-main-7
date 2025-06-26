import gurobipy as gp
from gurobipy import GRB
import model
import cuts
import master_model as master_module

class SubproblemGurobi:
    def __init__(self, problem: model.ProblemInstance, master_solution: dict, master_model: master_module.MasterModel):
        self.problem = problem
        self.master_solution = master_solution
        self.master_model = master_model
        self.model = gp.Model("Subproblem-Gurobi")
        self.model.Params.OutputFlag = 0
        self.x = {}

    def solve(self) -> cuts.Cut:
        self._build_model()
        self.model.optimize()

        if self.model.Status == GRB.OPTIMAL:
            events = [{'train': t, 'operation': o, 'time': int(round(v.X))} for (t, o), v in self.x.items()]
            return cuts.OptimalityCut(self.model.ObjVal, events, self.master_solution)
        
        elif self.model.Status == GRB.INFEASIBLE:
            print("  -> Unzulässiges Subproblem. Finde minimalen Konflikt (IIS)...")
            self.model.computeIIS()
            
            conflict_vars = []
            for c in self.model.getConstrs():
                if c.IISConstr:
                    constr_name = c.ConstrName
                    if constr_name.startswith("res_y_"):
                        parts = constr_name.split('_')
                        t1, o1, t2, o2 = map(int, parts[2:])
                        # Finde die verantwortliche y-Variable
                        y_var1 = self.master_model.y.get((t1, o1, t2, o2))
                        y_var2 = self.master_model.y.get((t2, o2, t1, o1))
                        if y_var1 and self.master_solution.get(y_var1, 0.0) > 0.5:
                             conflict_vars.append(y_var1)
                        elif y_var2:
                             conflict_vars.append(y_var2)
                    elif constr_name.startswith("path_z_"):
                        parts = constr_name.split('_')
                        t, o, s = map(int, parts[2:])
                        z_var = self.master_model.z.get((t,o,s))
                        if z_var: conflict_vars.append(z_var)

            if not conflict_vars:
                print("    -> Warnung: IIS-Analyse ohne Ergebnis, verwende allgemeinen No-Good-Cut.")
                conflict_vars = [var for var, val in self.master_solution.items() if val > 0.5]

            print(f"    -> IIS Cut generiert mit {len(conflict_vars)} Variablen.")
            return cuts.FeasibilityCut(conflict_vars)
        else:
            raise RuntimeError(f"Gurobi Subproblem endete mit Status: {self.model.Status}")

    def _build_model(self):
        self.x = self.model.addVars(
            ((t, o) for t, tr in enumerate(self.problem.trains) for o, op in enumerate(tr.operations)),
            vtype=GRB.INTEGER, lb=0, name="x"
        )
        
        for t_idx, train in enumerate(self.problem.trains):
            for o_idx, op in enumerate(train.operations):
                if op.start_ub != float('inf'): self.model.addConstr(self.x[t_idx, o_idx] <= op.start_ub)
                s_idx = self._get_chosen_successor(t_idx, o_idx)
                if s_idx != -1:
                    constr_name = f"path_z_{t_idx}_{o_idx}_{s_idx}" if len(op.successors) > 1 else ""
                    self.model.addConstr(self.x[t_idx, s_idx] >= self.x[t_idx, o_idx] + op.min_duration, name=constr_name)
        
        for (t1, o1), (t2, o2) in self.problem.get_conflicts():
            y_var = self.master_model.y.get((t1, o1, t2, o2))
            if y_var is None: continue
            op1, op2 = self.problem.trains[t1].operations[o1], self.problem.trains[t2].operations[o2]
            s1_idx, s2_idx = self._get_chosen_successor(t1, o1), self._get_chosen_successor(t2, o2)
            if s1_idx == -1 or s2_idx == -1: continue
            res_map1 = {r['resource']: r.get('release_time', 0) for r in op1.resources}
            res_map2 = {r['resource']: r.get('release_time', 0) for r in op2.resources}
            shared = set(res_map1.keys()) & set(res_map2.keys())
            if not shared: continue
            max_rel1, max_rel2 = max(res_map1[r] for r in shared), max(res_map2[r] for r in shared)

            if self.master_solution.get(y_var, 0.0) > 0.5:
                self.model.addConstr(self.x[t2, o2] >= self.x[t1, s1_idx] + max_rel1, name=f"res_y_{t1}_{o1}_{t2}_{o2}")
            else:
                self.model.addConstr(self.x[t1, o1] >= self.x[t2, s2_idx] + max_rel2, name=f"res_y_{t2}_{o2}_{t1}_{o1}")
                
    def _get_chosen_successor(self, t_idx: int, o_idx: int) -> int:
        op = self.problem.trains[t_idx].operations[o_idx]
        if not op.successors: return -1
        if len(op.successors) == 1: return op.successors[0]
        for s_idx in op.successors:
            z_var = self.master_model.z.get((t_idx, o_idx, s_idx))
            if z_var and self.master_solution.get(z_var, 0.0) > 0.5:
                return s_idx
        return -1 # Wichtig: Wenn kein Pfad gewählt wurde, gibt es keinen Nachfolger