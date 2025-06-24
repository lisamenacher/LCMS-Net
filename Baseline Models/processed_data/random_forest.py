import utils
import pickle 

from sklearn.ensemble import RandomForestClassifier
from skopt.space import Real, Categorical, Integer


# load data
X, y, X_test, y_test = utils.load_data()

# load model and define grid
model = RandomForestClassifier()
grid = {
    "n_estimators": Integer(10, 250),
    "criterion": Categorical(["gini", "entropy", "log_loss"]),
    "max_depth": Integer(2, 100),
    "min_samples_split": Integer(2, 50),
    "min_samples_leaf": Integer(2, 25),
    "max_leaf_nodes": Integer(5, 200),
    "max_features": Categorical(["sqrt", "log2", None]),
    "class_weight": Categorical(["balanced", None, "balanced_subsample"]),
    "bootstrap": Categorical([True, False])}

# retrieve best model
search = utils.perform_hyperparameter_tuning(model, grid, X, y)
print(search.best_params_)
model = search.best_estimator_

# eval results
y_pred = model.predict(X)
utils.eval_metrics(y_pred, y, "RF_train", "./Baseline Models/results")

y_pred = model.predict(X_test)
utils.eval_metrics(y_pred, y_test, "RF_test", "./Baseline Models/results")

# save results
with open('./Baseline Models/results/RF_params.pkl', 'wb') as f:
    pickle.dump(search.best_estimator_, f)

with open('./Baseline Models/results/RF_cv.pkl', 'wb') as f:
    pickle.dump(search.cv_results_, f)