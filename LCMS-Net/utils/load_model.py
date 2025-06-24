import numpy as np
from tensorflow.keras.models import load_model

class EnsembleModel():
    def __init__(self, model_paths):

        # load all models from specified paths
        models = []
        for model_path in model_paths:
            model = load_model(model_path)
            models.append(model)

            self.input_size = model.input.shape[1:]

        # save models
        self.models = models

    def predict(self, x):

        # predict label with each model
        y_preds = []
        for model in self.models:
            y_pred = model.predict(x)
            y_preds.append(y_pred)

        # perform majority vote
        y_mean_pred = np.mean(y_preds, axis=0)

        return y_mean_pred


class SingleModel():
    def __init__(self, model_path):

        # load model
        self.model = load_model(model_path)

    def predict(self, x):

        # predict class label
        return self.model.predict(x)