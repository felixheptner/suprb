import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.utils import shuffle

import suprb.optimizer.rule.mutation
from suprb import SupRB
from suprb import rule
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import es

if __name__ == "__main__":
    random_state = 42

    data, _ = fetch_openml(name="Concrete_Data", version=1, return_X_y=True)
    data = data.to_numpy()

    X, y = data[:, :8], data[:, 8]
    X, y = shuffle(X, y, random_state=random_state)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    models = [
        LinearRegression(),
        DecisionTreeRegressor(random_state=random_state),
        RandomForestRegressor(random_state=random_state),
        ExtraTreeRegressor(random_state=random_state),
        ExtraTreesRegressor(random_state=random_state),
        GradientBoostingRegressor(random_state=random_state),
        HistGradientBoostingRegressor(random_state=random_state),
        AdaBoostRegressor(random_state=random_state),
        SVR(),
        KNeighborsRegressor(),
        SupRB(
            rule_discovery=es.ES1xLambda(
                operator="&",
                init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu(alpha=0.8)),
                mutation=suprb.optimizer.rule.mutation.HalfnormIncrease(sigma=2),
            ),
            solution_composition=ga.GeneticAlgorithm(
                n_iter=128,
                crossover=ga.crossover.Uniform(),
                selection=ga.selection.Tournament(),
                mutation=ga.mutation.BitFlips(),
            ),
            n_iter=32,
            n_rules=4,
            logger=StdoutLogger(),
            random_state=random_state,
        ),
    ]
    models = {model.__class__.__name__: model for model in models}

    def run(name, model):
        print(f"[EVALUATION] {name}")
        return pd.Series(
            cross_val_score(
                model,
                X,
                y,
                cv=4,
                n_jobs=4,
                verbose=10,
                scoring="neg_root_mean_squared_error",
            ),
            name="negated RMSE",
        )

    scores = pd.concat({name: run(name=name, model=model) for name, model in models.items()}, axis=0).to_frame()
    scores.index.names = ["model", "cv"]

    print(scores.groupby(by="model").describe().to_string())
