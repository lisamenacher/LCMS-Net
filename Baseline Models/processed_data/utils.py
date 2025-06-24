import pandas as pd
import numpy as np
import sys

from skopt import BayesSearchCV
from imblearn.metrics import specificity_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, make_scorer
import sklearn.preprocessing as sklearn_preprocess


def load_data(data_path="./processed_data.xlsx", train_path="./meta_train.xlsx", test_path="./meta_test.xlsx", normalization_type="pqn", imputation_type="half-min"):
        """ Loads train and test dataset from excel file and processes it for CoD prediction.
        
        Parameters: \\
        data_path (str):            Filepath of processed LC-MS data (xlsx) \\    
        train_path (str):           Filepath of meta data for training samples (xlsx) \\
        test_path (str):            Filepath of meta data for test samples (xlsx) \\
        normalization_type (str):   Normalization for LC-MS features (either pqn, log-norm or None) \\
        imputation_type (str):      Missing value imputation type (either half-min or zero) \\


        Returns: \\
        np.array:                   Training Data
        np.array:                   Training Labels
        np.array:                   Test Data
        np.array:                   Test Labels
        """

        
        # load meta data, add numeric labels and shuffle samples
        meta_train = pd.read_excel(train_path, index_col=0)
        meta_train["labels_num"] = np.unique(meta_train["Group"], return_inverse=True)[1]
        meta_train = meta_train.sample(frac=1)

        meta_test = pd.read_excel(test_path, index_col=0)
        meta_test["labels_num"] = np.unique(meta_test["Group"], return_inverse=True)[1]
        meta_test = meta_test.sample(frac=1)

        # load processed data (remove unnecessary columns/rows from processing) and split in train/test dataset
        data = pd.read_excel(data_path, index_col=0)
        data = data.iloc[:,20:]
        data = data[:-5]

        data_train = np.array(data.loc[meta_train.index])
        data_test = np.array(data.loc[meta_test.index])

        # impute missing values
        if imputation_type == "half-min":
                # use half-minimum imputation
                min_vals = np.nanmin(data_train, axis=0)
                for c in range(data_train.shape[1]):
                       data_train[pd.isna(data_train[:,c]),c] = min_vals[c]/2

                for c in range(data_test.shape[1]):
                       data_test[pd.isna(data_test[:,c]),c] = min_vals[c]/2
        elif imputation_type == "zero":
                # fill in zero values
                data = data.fillna(1)
                data_train = np.array(data.loc[meta_train.index])
                data_test = np.array(data.loc[meta_test.index])
        else:
                print('ERROR - Undefined Imputation!')
                sys.exit()

        # normalize data 
        if normalization_type == "pqn":
                # use probabilistic quotient normalization
                median_intensities = np.median(data_train, axis=0)
                quotients = data_train / median_intensities
                sample_medians = np.median(quotients, axis=1)
                data_train = data_train / sample_medians[:, np.newaxis]

                quotients = data_test / median_intensities
                sample_medians = np.median(quotients, axis=1) 
                data_test = data_test / sample_medians[:, np.newaxis]        
                
                # apply log-transform and standard scaler
                data_train = np.log(data_train.astype(np.float64))
                scaler = sklearn_preprocess.StandardScaler()
                scaler.fit(data_train)
                data_train = scaler.transform(data_train)

                data_test = np.log(data_test.astype(np.float64))
                data_test = scaler.transform(data_test)
        elif normalization_type == "log-norm":
               # use log transformation and standard scaler
               data_train = np.log(data_train.astype(np.float64))
               scaler = sklearn_preprocess.StandardScaler()
               scaler.fit(data_train)
               data_train = scaler.transform(data_train)

               data_test = np.log(data_test.astype(np.float64))
               data_test = scaler.transform(data_test)
        elif normalization_type is None:
                data_train = data_train
                data_test = data_test
        else: 
                print('ERROR - Undefined Normalization!')
                sys.exit()

        # return datasets
        return data_train, np.array(meta_train["labels_num"]), data_test, np.array(meta_test["labels_num"])


def perform_hyperparameter_tuning(model, search_spaces, data, labels):
        """ Perform Bayesian Hyperparameter-Tuning for a model on a predefined search space.
        
        Parameters: \\
        model:                      Sklearn Classification Model \\    
        search_spaces (dict):       Dictonary that defines the search space of all parameters that need to be optimized \\
        data (np.array):            Processed training data \\
        labels (np.array):          Labels of training data \\


        Returns: \\
        skopt object
        """

        def macro_f1_scorer(y_true, y_pred):
                return f1_score(y_true, y_pred, average="macro")
        score = make_scorer(macro_f1_scorer, greater_is_better=True)

        opt = BayesSearchCV(model, search_spaces, n_iter=100, cv=5, n_jobs=5, random_state=42, verbose=5, scoring=score)
        opt.fit(data, labels)

        return opt


def eval_metrics(y_pred, y_true, model_name, save_path="./Baseline Models/processed_data/results"):
        """ Evaluates a CoD classification model.
        
        Parameters: \\
        y_pred (np.array):           Predicted CoD labels \\    
        y_true (np.array):           True CoD labels \\
        model_name (str):            Name of CoD prediction model (used to store results) \\
        save_path (str):             Path to store results \\


        Returns: \\
        """    

        # define labels
        labels = ["Acidosis", "Drug", "Hanging", "IHD", "Pneumonia"]

        # create and save confusion matrix plots
        cm_num = confusion_matrix(y_true, y_pred)
        cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
        cm_num_plot = ConfusionMatrixDisplay(confusion_matrix=cm_num, display_labels=labels)
        cm_norm_plot = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)

        cm_num_plot.plot()
        cm_num_plot.figure_.savefig(save_path + f"/{model_name}_cm_numeric.png")
        cm_norm_plot.plot()
        cm_norm_plot.figure_.savefig(save_path + f"/{model_name}_cm_normalized.png")

        # create and save classification report
        # class-wise metrics
        f1_scores = []
        accuracies = []
        precisions = []
        recalls = []
        specificities = []
        roc_scores = []
        for i in range(len(labels)):
                temp_true = [1 if x == i else 0 for x in y_true]
                temp_pred = [1 if x == i else 0 for x in y_pred]
                
                f1_scores.append(f1_score(temp_true, temp_pred, average="binary"))
                accuracies.append(accuracy_score(temp_true, temp_pred))
                precisions.append(precision_score(temp_true, temp_pred, average="binary"))
                recalls.append(recall_score(temp_true, temp_pred, average="binary"))
                specificities.append(specificity_score(temp_true, temp_pred, average="binary"))
                roc_scores.append(roc_auc_score(temp_true, temp_pred))

        dict = {'F1-Score': f1_scores, 'Accuracy': accuracies, 'Precision': precisions, "Recall": recalls, "Specificity": specificities, "AUROC": roc_scores}
        class_metrics = pd.DataFrame(dict, index=pd.Index(labels))
        class_metrics.to_csv(save_path + f"/{model_name}_class_metrics.csv")

        # overall 
        weighted_f1_scores = []
        macro_f1_scores = []
        accuracies = []
        macro_precisions = []
        weighted_precisions = []
        macro_recalls = []
        weighted_recalls = []
        macro_specificities = []
        weighted_specificities = []
        roc_scores = []
                
        weighted_f1_scores.append(f1_score(y_true, y_pred, average="weighted"))
        macro_f1_scores.append(f1_score(y_true, y_pred, average="macro"))
        accuracies.append(accuracy_score(y_true, y_pred))
        weighted_precisions.append(precision_score(y_true, y_pred, average="weighted"))
        macro_precisions.append(precision_score(y_true, y_pred, average="macro"))
        weighted_recalls.append(recall_score(y_true, y_pred, average="weighted"))
        macro_recalls.append(recall_score(y_true, y_pred, average="macro"))
        weighted_specificities.append(specificity_score(y_true, y_pred, average="weighted"))
        macro_specificities.append(specificity_score(y_true, y_pred, average="macro"))


        dict = {'(Macro) F1-Score': macro_f1_scores, '(Weighted) F1-Score': weighted_f1_scores, 'Accuracy': accuracies, '(Macro) Precision': macro_precisions, '(Weighted) Precision': weighted_precisions, "(Macro) Recall": macro_recalls, "(Weighted) Recall": weighted_recalls, "(Macro) Specificity": macro_specificities, "(Weighted) Specificity": weighted_specificities}
        overall_metrics = pd.DataFrame(dict)
        overall_metrics.to_csv(save_path + f"/{model_name}_metrics.csv")