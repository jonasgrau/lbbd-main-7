import json
import os
from typing import List, Dict, Any

class Operation:
    """Stellt eine einzelne Operation innerhalb des Graphen eines Zuges dar."""
    def __init__(self, train_idx: int, op_idx: int, op_data: Dict[str, Any]):
        self.train_idx = train_idx
        self.op_idx = op_idx
        self.start_lb: int = op_data.get("start_lb", 0)
        self.start_ub: int = op_data.get("start_ub", float('inf'))
        self.min_duration: int = op_data["min_duration"]
        self.successors: List[int] = op_data.get("successors", [])
        self.resources: List[Dict] = op_data.get("resources", [])

class Train:
    """Stellt einen einzelnen Zug mit all seinen Operationen dar."""
    def __init__(self, train_idx: int, operations_data: List[Dict[str, Any]]):
        self.train_idx = train_idx
        self.operations: List[Operation] = [
            Operation(train_idx, op_idx, op_data)
            for op_idx, op_data in enumerate(operations_data)
        ]

    def get_shortest_paths_to_exit(self) -> dict:
        """Berechnet für jede Operation die kürzeste verbleibende Dauer bis zum Ende des Zuges."""
        num_ops = len(self.operations)
        dist = {i: float('inf') for i in range(num_ops)}
        exit_op_idx = num_ops - 1
        if exit_op_idx >= 0:
            dist[exit_op_idx] = 0
        for i in range(exit_op_idx - 1, -1, -1):
            op = self.operations[i]
            if not op.successors: continue
            min_succ_dist = float('inf')
            for s_idx in op.successors:
                path_dist = op.min_duration + dist.get(s_idx, float('inf'))
                min_succ_dist = min(min_succ_dist, path_dist)
            dist[i] = min_succ_dist
        return dist

class ObjectiveComponent:
    """Stellt eine Komponente der Zielfunktion dar."""
    def __init__(self, data: Dict[str, Any]):
        self.type: str = data["type"]
        self.train: int = data["train"]
        self.operation: int = data["operation"]
        self.threshold: int = data.get("threshold", 0)
        self.increment: int = data.get("increment", 0)
        self.coeff: int = data.get("coeff", 0)

class ProblemInstance:
    """Lädt und speichert eine vollständige DISPLIB-Probleminstanz."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.trains: List[Train] = []
        self.objective_components: List[ObjectiveComponent] = []
        self._load_from_json()

    def _load_from_json(self):
        print(f"Lade Probleminstanz von: {self.filepath}")
        with open(self.filepath, 'r') as f: data = json.load(f)
        for train_idx, train_ops_data in enumerate(data.get("trains", [])):
            self.trains.append(Train(train_idx, train_ops_data))
        for obj_data in data.get("objective", []):
            self.objective_components.append(ObjectiveComponent(obj_data))
        print(f"Laden erfolgreich: {len(self.trains)} Züge und {len(self.objective_components)} Zielfunktions-Komponenten gefunden.")

    def get_resource_usage_map(self) -> Dict[str, list]:
        usage_map = {}
        for t_idx, train in enumerate(self.trains):
            for o_idx, op in enumerate(train.operations):
                for res in op.resources:
                    res_name = res.get("resource")
                    if res_name:
                        usage_map.setdefault(res_name, []).append((t_idx, o_idx))
        return usage_map

    def get_conflicts(self) -> set:
        """Erstellt eine Menge aller eindeutigen Ressourcenkonflikte."""
        usage = self.get_resource_usage_map()
        conflicts = set()
        for res, ops in usage.items():
            for i in range(len(ops)):
                for j in range(i + 1, len(ops)):
                    t1, o1 = ops[i]
                    t2, o2 = ops[j]
                    if t1 != t2:
                        conflicts.add(tuple(sorted(((t1, o1), (t2, o2)))))
        return conflicts

    def get_2_train_swap_constraints(self) -> list:
        """Findet Paare von Konflikten, die zu 2-Zug-Deadlocks führen können."""
        train_res = {
            train.train_idx: {
                r.get("resource"): op.op_idx
                for op in train.operations
                for r in op.resources if r.get("resource")
            }
            for train in self.trains
        }
        pairs = []
        trains_list = sorted(list(train_res.keys()))
        for i in range(len(trains_list)):
            for j in range(i + 1, len(trains_list)):
                t1, t2 = trains_list[i], trains_list[j]
                m1, m2 = train_res[t1], train_res[t2]
                shared = sorted(list(set(m1.keys()) & set(m2.keys())))
                for a in range(len(shared)):
                    for b in range(a + 1, len(shared)):
                        rA, rB = shared[a], shared[b]
                        o1A, o1B = m1[rA], m1[rB]
                        o2A, o2B = m2[rA], m2[rB]
                        if (o1A < o1B and o2A > o2B) or (o1A > o1B and o2A < o2B):
                            pairs.append(((t1, o1A, t2, o2A), (t1, o1B, t2, o2B)))
        return pairs

    def get_3_train_cycle_constraints(self) -> list:
        """Findet Tripletts von Konflikten, die 3-Zug-Deadlock-Zyklen bilden."""
        cycles = []
        conflicts = self.get_conflicts()
        conflict_map = {}
        for c in conflicts:
            (t1, o1), (t2, o2) = c
            conflict_map[(t1, t2)] = (o1, o2)
            conflict_map[(t2, t1)] = (o2, o1)
        trains_list = sorted(list(set(t for c in conflicts for pair in c for t in pair)))
        for i in range(len(trains_list)):
            for j in range(i + 1, len(trains_list)):
                for k in range(j + 1, len(trains_list)):
                    t1, t2, t3 = trains_list[i], trains_list[j], trains_list[k]
                    c12, c23, c31 = conflict_map.get((t1, t2)), conflict_map.get((t2, t3)), conflict_map.get((t3, t1))
                    if c12 and c23 and c31:
                        cycles.append(((t1, c12[0], t2, c12[1]), (t2, c23[0], t3, c23[1]), (t3, c31[0], t1, c31[1])))
                    c13, c32, c21 = conflict_map.get((t1, t3)), conflict_map.get((t3, t2)), conflict_map.get((t2, t1))
                    if c13 and c32 and c21:
                        cycles.append(((t1, c13[0], t3, c13[1]), (t3, c32[0], t2, c32[1]), (t2, c21[0], t1, c21[1])))
        return cycles