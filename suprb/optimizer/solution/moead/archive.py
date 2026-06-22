import numpy as np

from suprb.optimizer.solution.base import Solution
from suprb.optimizer.solution.nsga2.sorting import fast_non_dominated_sort
from suprb.optimizer.solution.archive import SolutionArchive

class NonDominatedArchive(SolutionArchive):
    """Maintains all non-dominated solutions found during optimization.
    
    Used in MOEA/D to preserve Pareto-optimal solutions that may be discarded
    during local neighborhood updates based on scalarization.
    """

    def __init__(self, max_population_size):
        super().__init__()
        self.max_population_size = max_population_size

    def __call__(self, new_population: list[Solution]):
        # Combine new solutions with archived ones
        combined_population = new_population + self.population_
        
        if not combined_population:
            return
        
        # Extract fitness values and identify non-dominated solutions
        fitness_values = np.array([solution.fitness_ for solution in combined_population])
        pareto_ranks = fast_non_dominated_sort(fitness_values)
        
        # Keep only Pareto-dominant solutions (rank 0)
        pareto_solutions = np.array(combined_population)[pareto_ranks == 0]
        self.population_ = [solution.clone() for solution in pareto_solutions]
        # Limit the archive size
        # TODO: Implement a strategy to select which solution to keep if the number of non-dominated solutions exceeds max_population_size that is supported by literature (e.g. crowding distance, hypervolume contribution, etc.)
        # The current implementation does not limit the number of solution giving moead an unfair advantage as it can keep all non-dominated solutions found during optimization