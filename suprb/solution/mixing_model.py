import numpy as np

from suprb.rule import Rule
from suprb.utils import check_random_state, RandomState
from . import MixingModel


class FilterSubpopulation:
    def __init__(self, rule_amount: int = 6, random_state: RandomState = 42):
        self.rule_amount = rule_amount
        self.random_state = check_random_state(random_state)

    def __call__(self, subpopulation: list[Rule]) -> list[Rule]:
        return subpopulation

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class NBestFitness(FilterSubpopulation):
    def __call__(self, subpopulation: list[Rule]) -> list[Rule]:
        fitnesses = np.array([rule.fitness_ for rule in subpopulation])
        ind = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[-self.rule_amount :]
        return [subpopulation[i] for i in ind]


class NRandom(FilterSubpopulation):
    def __call__(self, subpopulation: list[Rule]) -> list[Rule]:
        choice_size = min(len(subpopulation), self.rule_amount)
        return self.random_state.choice(subpopulation, size=choice_size, replace=False)


class RouletteWheel(FilterSubpopulation):
    def __call__(self, subpopulation: list[Rule]) -> list[Rule]:
        fitnesses = np.array([rule.fitness_ for rule in subpopulation])
        weights = fitnesses / np.sum(fitnesses)
        choice_size = min(len(subpopulation), self.rule_amount)
        return self.random_state.choice(subpopulation, p=weights, size=choice_size, replace=False)


class ExperienceCalculation:
    def __init__(self, lower_bound: float = -np.inf, upper_bound: float = np.inf):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, subpopulation: list[Rule], dim: int = None) -> list[Rule]:
        return np.array([rule.experience_ for rule in subpopulation])

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class CapExperience(ExperienceCalculation):
    def __call__(self, subpopulation: list[Rule], dim: int = None) -> list[Rule]:
        experiences = np.array([rule.experience_ for rule in subpopulation])
        return np.clip(experiences, self.lower_bound, self.upper_bound)


class CapExperienceWithDimensionality(ExperienceCalculation):
    def __call__(self, subpopulation: list[Rule], dim: int = None) -> list[Rule]:
        experiences = np.array([rule.experience_ for rule in subpopulation])
        return np.clip(experiences, self.lower_bound * dim, self.upper_bound * dim)


class ErrorExperienceHeuristic(MixingModel):
    """
    Performs mixing similar to the Inverse Variance Heuristic from
    https://researchportal.bath.ac.uk/en/studentTheses/learning-classifier-systems-from-first-principles-a-probabilistic,
    but using (error / experience) as a mixing function.
    """

    def __init__(
        self,
        filter_subpopulation: FilterSubpopulation = FilterSubpopulation(),
        experience_calculation: ExperienceCalculation = ExperienceCalculation(),
        experience_weight: float = 1,
    ):
        self.input_size = None
        self.filter_subpopulation = filter_subpopulation
        self.experience_calculation = experience_calculation
        self.experience_weight = experience_weight

    def __call__(self, X: np.ndarray, subpopulation: list[Rule], cache=False) -> np.ndarray:
        self.input_size = X.shape[0]

        # No need to perform any calculation if no rule was selected.
        if not subpopulation:
            return np.zeros(self.input_size)

        subpopulation = self.filter_subpopulation(subpopulation)

        local_pred, matches = self._get_local_pred(X, subpopulation, cache)
        taus = self._get_taus(subpopulation, X.shape[1])

        # Stack all local predictions and sum them weighted with tau
        pred = np.sum(local_pred * taus[:, None], axis=0)

        # Normalize
        tau_sum = self._get_tau_sum(subpopulation, matches, taus)
        out = pred / tau_sum
        return out

    def _get_local_pred(self, X: np.ndarray, subpopulation: list[Rule], cache: bool):
        local_pred = np.zeros((len(subpopulation), self.input_size))

        if cache:
            # Use the precalculated matches and predictions from fit()
            matches = [rule.match_set_ for rule in subpopulation]
            for i, rule in enumerate(subpopulation):
                local_pred[i][matches[i]] = rule.pred_
        else:
            # Generate all data new
            matches = [rule.match(X) for rule in subpopulation]
            for i, rule in enumerate(subpopulation):
                if not matches[i].any():
                    continue
                local_pred[i][matches[i]] = rule.predict(X[matches[i]])

        return local_pred, matches

    def _get_taus(self, subpopulation: list[Rule], dim: int):
        # Get errors and experience of all rules in subpopulation
        experiences = self.experience_calculation(subpopulation, dim)
        errors = np.array([rule.error_ for rule in subpopulation])

        return (1 / errors) * (experiences * self.experience_weight)

    def _get_tau_sum(self, subpopulation: list[Rule], matches: list[Rule], taus: list[int]):
        # Sum all taus
        local_taus = np.zeros((len(subpopulation), self.input_size))
        for i in range(len(subpopulation)):
            local_taus[i][matches[i]] = taus[i]

        tau_sum = np.sum(local_taus, axis=0)
        tau_sum[tau_sum == 0] = 1  # Needed, otherwise "out = pred / tau_sum" might become a divison by 0

        return tau_sum
