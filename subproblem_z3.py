import z3
import gurobipy as gp
import model
import cuts
import master_model as master_module

class SubproblemZ3:
    def __init__(self, problem: model.ProblemInstance, master_solution: dict, master_model: master_module.MasterModel):
        self.problem, self.master_solution, self.master_model = problem, master_solution, master_model
        self.solver = z3.Solver()
        self.solver.set(unsat_core=True)
        self.tracker_map, self.assumptions = {}, []
        self._create_z3_variables()

    def _create_z3_variables(self):
        self.x = {(t, o): z3.Int(f"x_{t}_{o}") for t, tr in enumerate(self.problem.trains) for o, op in enumerate(tr.operations)}

    def _get_chosen_successor(self, t_idx: int, o_idx: int) -> int:
        op = self.problem.trains[t_idx].operations[o_idx]
        if not op.successors: return -1
        if len(op.successors) == 1: return op.successors[0]
        for s_idx in op.successors:
            if self.master_solution.get(self.master_model.z[t_idx, o_idx, s_idx], 0.0) > 0.5: return s_idx
        return -1

    def solve(self) -> cuts.Cut:
        self._add_constraints()
        result = self.solver.check(self.assumptions)
        if result == z3.sat: return self._handle_sat()
        if result == z3.unsat: return self._handle_unsat()
        raise RuntimeError(f"Z3 Solver Status: {result}")

    def _add_constraints(self):
        for t_idx, train in enumerate(self.problem.trains):
            for o_idx, op in enumerate(train.operations):
                if op.start_lb is not None: self.solver.add(self.x[t_idx, o_idx] >= op.start_lb)
                if op.start_ub != float('inf'): self.solver.add(self.x[t_idx, o_idx] <= op.start_ub)
                s_idx = self._get_chosen_successor(t_idx, o_idx)
                if s_idx != -1:
                    self.solver.add(self.x[t_idx, s_idx] >= self.x[t_idx, o_idx] + op.min_duration)
                    if len(op.successors) > 1:
                        z_var = self.master_model.z[t_idx, o_idx, s_idx]
                        tracker = z3.Bool(f"z_{t_idx}_{o_idx}_{s_idx}")
                        self.tracker_map[tracker] = z_var
                        self.assumptions.append(tracker)
                        self.solver.assert_and_track(self.x[t_idx, o_idx] >= 0, tracker)
        
        for (t1, o1), (t2, o2) in self.problem.get_conflicts():
            op1, op2 = self.problem.trains[t1].operations[o1], self.problem.trains[t2].operations[o2]
            s1_idx, s2_idx = self._get_chosen_successor(t1, o1), self._get_chosen_successor(t2, o2)
            if s1_idx == -1 or s2_idx == -1: continue
            
            res_map1 = {r['resource']: r.get('release_time', 0) for r in op1.resources}
            res_map2 = {r['resource']: r.get('release_time', 0) for r in op2.resources}
            shared = set(res_map1.keys()) & set(res_map2.keys())
            if not shared: continue
            
            max_rel1, max_rel2 = max(res_map1[r] for r in shared), max(res_map2[r] for r in shared)
            y12 = self.master_model.y[t1, o1, t2, o2]
            y21 = self.master_model.y[t2, o2, t1, o1]

            if self.master_solution.get(y12, 0.0) > 0.5:
                tracker = z3.Bool(f"y_{t1}_{o1}_{t2}_{o2}")
                self.tracker_map[tracker] = y12
                self.assumptions.append(tracker)
                self.solver.assert_and_track(self.x[t2, o2] >= self.x[t1, s1_idx] + max_rel1, tracker)
            else:
                tracker = z3.Bool(f"y_{t2}_{o2}_{t1}_{o1}")
                self.tracker_map[tracker] = y21
                self.assumptions.append(tracker)
                self.solver.assert_and_track(self.x[t1, o1] >= self.x[t2, s2_idx] + max_rel2, tracker)

    def _handle_sat(self) -> cuts.OptimalityCut:
        m = self.solver.model()
        events = [{'train': t, 'operation': o, 'time': m.eval(x).as_long()} for (t, o), x in self.x.items()]
        return cuts.OptimalityCut(self._calculate_objective(m), events, self.master_solution)

    def _calculate_objective(self, model: z3.ModelRef) -> float:
        cost = 0.0
        for obj in self.problem.objective_components:
            delay = max(0, model.eval(self.x[obj.train, obj.operation]).as_long() - obj.threshold)
            cost += obj.coeff * delay + (obj.increment if delay > 0 else 0)
        return cost

    def _handle_unsat(self) -> cuts.FeasibilityCut:
        return cuts.FeasibilityCut([self.tracker_map[t] for t in self.solver.unsat_core()])