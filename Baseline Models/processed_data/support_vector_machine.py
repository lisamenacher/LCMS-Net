import utils
import pickle 

from sklearn.svm import SVC
from skopt.space import Real, Categorical, Integer

# load data
X, y, X_test, y_test = utils.load_data()

# load model and define grid
model = SVC()
grid = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'degree': Integer(1,8),
    'kernel': Categorical(['linear', 'poly', 'rbf']),
    "class_weight": Categorical([None, "balanced"])
    }

# retrieve best model
search = utils.perform_hyperparameter_tuning(model, grid, X, y)
print(search.best_params_)
model = search.best_estimator_

# eval results
y_pred = model.predict(X)
utils.eval_metrics(y_pred, y, "SVC_train", "./Baseline Models/results")

y_pred = model.predict(X_test)
utils.eval_metrics(y_pred, y_test, "SVC_test", "./Baseline Models/results")

# save results
with open('./Baseline Models/results/SVC_params.pkl', 'wb') as f:
    pickle.dump(search.best_estimator_, f)


with open('./Baseline Models/results/SVC_cv.pkl', 'wb') as f:
    pickle.dump(search.cv_results_, f)
