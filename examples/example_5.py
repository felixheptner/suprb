import sklearn

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.datasets import fetch_openml

from matplotlib import pyplot as plt

from suprb import SupRB
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.nsga2 import NonDominatedSortingGeneticAlgorithm2
from suprb.optimizer.solution.spea2 import StrengthParetoEvolutionaryAlgorithm2
from suprb.logging.multi_objective import MOLogger

from utils import log_scores


if __name__ == "__main__":
    random_state = 42

    data, _ = fetch_openml(name="Concrete_Data", version=1, return_X_y=True)
    data = data.to_numpy()

    X, y = data[:, :8], data[:, 8]
    X, y = sklearn.utils.shuffle(X, y, random_state=random_state)

    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    model = SupRB(
        rule_discovery=ES1xLambda(
            n_iter=32,
            lmbda=16,
            operator="+",
            delay=150,
            random_state=random_state,
            n_jobs=1,
        ),
        solution_composition=StrengthParetoEvolutionaryAlgorithm2(
            n_iter=32,
            population_size=32,
            random_state=random_state,
            n_jobs=1,
        ),
        logger=MOLogger()
    )

    scores = cross_validate(
        model,
        X,
        y,
        cv=2,
        n_jobs=1,
        verbose=10,
        scoring=["r2", "neg_mean_squared_error"],
        return_estimator=True,
        fit_params={"cleanup": True},
    )

    log_scores(scores)
    pareto_front = scores["estimator"][0].logger_.pareto_front_
    x = pareto_front[:, 0]
    y = pareto_front[:, 1]
    plt.plot(x, y, color="black", linestyle="-", marker="o")
    plt.show()
    print("Finished!")
