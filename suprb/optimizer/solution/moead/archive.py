import numpy as np

from suprb.optimizer.solution.base import Solution
from suprb.optimizer.solution.nsga2.sorting import fast_non_dominated_sort
from suprb.optimizer.solution.archive import SolutionArchive

class NonDominatedArchive(SolutionArchive):
    """Maintains all non-dominated solutions found during optimization.
    
    Used in MOEA/D to preserve Pareto-optimal solutions that may be discarded
    during local neighborhood updates based on scalarization.
    """

    def __init__(self):
        super().__init__()
        self.pop_complexity_map = {}  # To track unique solutions based on complexity

    def __call__(self, new_population: list[Solution]):

        for solution in new_population:
            if self.pop_complexity_map.get(solution.complexity_, None) is None:
                self.pop_complexity_map[solution.complexity_] = solution
                continue
            existing_solution = self.pop_complexity_map[solution.complexity_]
            if solution.error_ < existing_solution.error_:
                self.pop_complexity_map[solution.complexity_] = solution
        
        self.population_ = list(self.pop_complexity_map.values())
