import utils
import pickle 

from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, RegressorMixin
from skopt.space import Real, Categorical, Integer


# load data
X, y, X_test, y_test = utils.load_data()

# load model and define grid (base model needs to be updated to tune number of layers)
class MLP_Tuneable_Classifier(BaseEstimator, RegressorMixin):
    def __init__(self, epsilon=0.0001, beta_1=0.99, beta_2=0.9, validation_fraction=0.1, early_stopping=True,
                 learning_rate_init=0.001, learning_rate="constant", batch_size=16, 
                 alpha=0.01, solver="adam", activation="relu", layer_1=32, layer_2=64, layer_3=128):
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.layer_3 = layer_3
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.validation_fraction = validation_fraction
        self.early_stopping = early_stopping
        self.learning_rate_init = learning_rate_init
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.alpha = alpha
        self.solver = solver
        self.activation = activation

    def fit(self, X, y):
        
        hidden_layers = []
        if self.layer_2 < 10:
            hidden_layers = [self.layer_1]
        elif self.layer_3 < 10:
            if self.layer_2 > 10:
                hidden_layers = [self.layer_1, self.layer_2]
            else:
                hidden_layers = [self.layer_1]
        else:
            hidden_layers = [self.layer_1, self.layer_2, self.layer_3]

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            epsilon=self.epsilon,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            validation_fraction=self.validation_fraction,
            early_stopping=self.early_stopping,
            learning_rate_init=self.learning_rate_init,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            alpha=self.alpha,
            solver=self.solver,
            activation=self.activation
        )
        model.fit(X, y)
        self.model = model
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

model = MLP_Tuneable_Classifier()
search_space = {
    "layer_1": Integer(10, 512),
    "layer_2": Integer(0, 512),
    "layer_3": Integer(0, 512),
    "epsilon": Real(1e-10, 0.1, prior='uniform'),
    "beta_1": Real(1e-10,0.99999, prior="uniform"),
    "beta_2": Real(1e-10,0.99999, prior="uniform"),
    "validation_fraction": [0.1],
    "early_stopping": [True],
    "learning_rate_init": Real(1e-10, 0.5, prior="uniform"),
    "learning_rate": Categorical(["constant", "invscaling", "adaptive"]),
    "batch_size": Categorical([4, 8, 16, 32, 64, 128, 256]),
    "alpha": Real(1e-10, 10, prior="uniform"),
    "solver": ["adam"],
    "activation": ["relu"]
    }

# retrieve best model
search = utils.perform_hyperparameter_tuning(model, search_space, X, y)
print(search.best_params_)
model = search.best_estimator_

# eval results
y_pred = model.predict(X)
utils.eval_metrics(y_pred, y, "MLP_train", "./Baseline Models/results")

y_pred = model.predict(X_test)
utils.eval_metrics(y_pred, y_test, "MLP_test", "./Baseline Models/results")

# save results
with open('./Baseline Models/results/MLP_params.pkl', 'wb') as f:
    pickle.dump(search.best_estimator_, f)

with open('./Baseline Models/results/MLP_cv.pkl', 'wb') as f:
    pickle.dump(search.cv_results_, f)
