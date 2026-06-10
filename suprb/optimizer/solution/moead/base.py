import numpy as np
import scipy.stats as stats

from suprb import Solution
from suprb.optimizer.solution.nsga2.sorting import fast_non_dominated_sort
from suprb.solution.initialization import SolutionInit, RandomInit
from ..base import MOSolutionComposition
from suprb.solution.fitness import NormalizedMOSolutionFitness

from suprb.optimizer.solution.nsga2.mutation import SolutionMutation, BitFlips
from suprb.optimizer.solution.nsga2.crossover import SolutionCrossover, NPoint
from ..sampler import SolutionSampler, BetaSolutionSampler


class MultiObjectiveEvolutionaryAlgorithmDecomposition(MOSolutionComposition):
    """MOEA/D — Multi-Objective Evolutionary Algorithm based on Decomposition.

    A simple MOEA/D implementation using Tchebycheff scalarization and neighbourhood
    updates. This implementation follows the common template of other solution
    composition optimizers in this project (see `nsga2`, `spea2`).

    Parameters
    ----------
    n_iter: int
        Iterations the metaheuristic will perform.
    population_size: int
        Number of subproblems / solutions.
    neighborhood_size: int
        Number of neighbouring weight vectors considered for mating and update.
    mutation: SolutionMutation
    crossover: SolutionCrossover
    sampler: SolutionSampler
    init: SolutionInit
    random_state : int, RandomState instance or None, default=None
    warm_start: bool
    n_jobs: int
    """

    def __init__(
        self,
        n_iter: int = 32,
        population_size: int = 32,
        neighborhood_size: int | None = None,
        mutation: SolutionMutation = BitFlips(),
        crossover: SolutionCrossover = NPoint(n=3),
        sampler: SolutionSampler = BetaSolutionSampler(1.5, 1.5),
        mutation_rate: float = 0.025,
        crossover_rate: float = 0.75,
        init: SolutionInit = RandomInit(fitness=NormalizedMOSolutionFitness()),
        random_state: int = None,
        n_jobs: int = 1,
        warm_start: bool = True,
        early_stopping_patience: int = -1,
        early_stopping_delta: float = 0,
    ):
        super().__init__(
            n_iter=n_iter,
            population_size=population_size,
            init=init,
            archive=None,
            sampler=sampler,
            random_state=random_state,
            n_jobs=n_jobs,
            warm_start=warm_start,
            early_stopping_patience=early_stopping_patience,
            early_stopping_delta=early_stopping_delta,
        )

        self.neighborhood_size = (
            neighborhood_size if neighborhood_size is not None else int(max(2, population_size**0.5))
        )
        self.mutation = mutation
        self.crossover = crossover
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # internals initialised in _optimize
        self.weights_ = None
        self.neighbours_ = None
        self.ideal_point_ = None
        self.scalarized_ = None

    @staticmethod
    def _tchebycheff(weight: np.ndarray, fitness: np.ndarray, ideal: np.ndarray) -> float:
        # For minimization problems we use the ideal point as component-wise minimum
        # and compute the weighted Chebyshev distance; smaller is better.
        diff = np.abs(ideal - fitness)
        vals = weight * diff
        # if any weight is zero, ignore that term by treating weight*diff as 0
        return float(np.max(vals))

    def _init_decomposition(self, n_objectives: int):
        """Create weight vectors and neighbourhood structure for the decomposition."""
        # generate weight vectors on the simplex using Dirichlet sampling
        rng = self.random_state_
        self.weights_ = rng.dirichlet(np.ones(n_objectives), size=self.population_size)

        # compute neighbourhood indices by Euclidean distance in weight space
        dist = np.linalg.norm(self.weights_[:, None, :] - self.weights_[None, :, :], axis=2)
        neighbours = np.argsort(dist, axis=1)[:, : self.neighborhood_size]
        self.neighbours_ = neighbours

    def _optimize(self, X: np.ndarray, y: np.ndarray):
        # Initialise and fit population
        self.fit_population(X, y)

        n_obj = len(self.init.fitness.objective_func_)
        # initial setup of weight vectors and neighbourhood if needed
        if self.weights_ is None or self.weights_.shape[1] != n_obj or len(self.weights_) != self.population_size:
            self._init_decomposition(n_obj)

        # initial ideal point: component-wise minimum since we minimize objectives
        fitness_values = np.array([s.fitness_ for s in self.population_])
        self.ideal_point_ = np.min(fitness_values, axis=0)

        # compute scalarized values for current population
        self.scalarized_ = np.array(
            [
                self._tchebycheff(self.weights_[i], self.population_[i].fitness_, self.ideal_point_)
                for i in range(len(self.population_))
            ]
        )

        for _ in range(self.n_iter):
            # iterate subproblems
            for i in range(self.population_size):
                neigh_idx = self.neighbours_[i]

                # parent selection: sample two parents from neighbourhood uniformly
                p_indices = self.random_state_.choice(neigh_idx, size=2, replace=False)
                A = self.population_[p_indices[0]]
                B = self.population_[p_indices[1]]

                # crossover and mutation
                child = self.crossover(A, B, random_state=self.random_state_)
                child = self.mutation(child, random_state=self.random_state_).fit(X, y)

                # update ideal point
                ideal_updated = False
                for m in range(len(self.ideal_point_)):
                    # For minimization problems a smaller objective improves the ideal point
                    if child.fitness_[m] < self.ideal_point_[m]:
                        self.ideal_point_[m] = child.fitness_[m]
                        ideal_updated = True

                # If the ideal point changed, scalarized values of all subproblems
                # must be recomputed because the scalarisation depends on the ideal.
                if ideal_updated:
                    self.scalarized_ = np.array(
                        [
                            self._tchebycheff(self.weights_[k], self.population_[k].fitness_, self.ideal_point_)
                            for k in range(len(self.population_))
                        ]
                    )

                # update neighbours: replace if child improves scalarized value for neighbour
                for k in neigh_idx:
                    g_child = self._tchebycheff(self.weights_[k], child.fitness_, self.ideal_point_)
                    if g_child < self.scalarized_[k]:
                        # replace solution k
                        self.population_[k] = child
                        self.scalarized_[k] = g_child

            if self.check_early_stopping():
                break

    def pareto_front(self) -> list[Solution]:
        if not hasattr(self, "population_") or not self.population_:
            return []
        fitness_values = np.array([solution.fitness_ for solution in self.population_])
        pareto_ranks = fast_non_dominated_sort(fitness_values)
        pareto_front = np.array(self.population_)[pareto_ranks == 0]
        return sorted(pareto_front, key=lambda x: x.fitness_[0], reverse=True)
