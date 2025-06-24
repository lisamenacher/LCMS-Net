import sys
import platform
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from imblearn.metrics import specificity_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def get_reject_threshold(model, data, y_true, min_sensitivitiy=0.3, min_specificity=0.9, save_path="./Baseline Models/results",):
    """ Get optimal threshold (i.e. margin between highest and second highest class probability) within boundaries.

    Parameter: \\
    model (keras model):        Fitted keras model \\
    data (np.array):            Array of processed LC-MS data or Generator object that produces data \\
    y_true (np.array):          Array of class labels \\
    min_sensitivity (float):    Minimum sensitivity to consider threshold \\
    min_specificity (float):    Minimum specificity to consider threshold \\
    save_path (str):            Path to directory where results will be stored \\


    Returns: \\
    int
    """
    best_thr = 0
    best_specificty = 0

    store_f1_scores = []
    store_specificities = []
    store_unclassified = []

    # predict samples
    y_prob = model.predict(data)

    # get ground truth
    y_true = np.argmax(y_true, axis=1)

    # create list with possible thresholds
    possible_thrs = np.linspace(0, 0.5, num=50)

    for thr in possible_thrs:
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        specificities = []
        roc_scores = []

        # assign class label based on current threshold
        curr_pred = predict_with_reject(y_prob, thr)
            
        # evaluate results based on current threshold
        for i in range(len(np.unique(y_true))):

            labels = [1 if x == i else 0 for x in y_true]

            temp_pred = [1 if x == i else 0 for x in curr_pred]
                
            f1_scores.append(f1_score(labels, temp_pred, average="binary"))
            accuracies.append(accuracy_score(labels, temp_pred))
            precisions.append(precision_score(labels, temp_pred, average="binary"))
            recalls.append(recall_score(labels, temp_pred, average="binary"))
            specificities.append(specificity_score(labels, temp_pred, average="binary"))
            roc_scores.append(roc_auc_score(labels, temp_pred))

        dict = {'F1-Score': f1_scores, 'Accuracy': accuracies, 'Precision': precisions, "Recall": recalls, "Specificity": specificities, "AUROC": roc_scores}
        class_metrics = pd.DataFrame(dict)

        # save as best, if better than previous best specificity and within boundaries
        if (class_metrics["Recall"]>=min_sensitivitiy).all() & (class_metrics["Specificity"]>=min_specificity).all() & (best_specificty < (class_metrics["Specificity"].sum()/len(np.unique(y_true)))) & ((np.unique(curr_pred, return_counts=True)[1][0]/len(curr_pred)) <= 0.35):
            best_thr = thr
            best_specificty = class_metrics["Specificity"].sum()/len(np.unique(y_true))

        store_specificities.append(class_metrics["Specificity"].sum()/len(np.unique(y_true)))
        store_f1_scores.append(class_metrics["F1-Score"].sum()/len(np.unique(y_true)))
        store_unclassified.append(np.unique(curr_pred, return_counts=True)[1][0]/len(curr_pred))

    res = pd.DataFrame({"Threshold": possible_thrs, "F1-Score": store_f1_scores, "Specificity": store_specificities, "Unclassified": store_unclassified})
    res.to_csv(save_path + "best_thr.csv")
    
    return best_thr


def predict_with_reject(probs, thr=0.20):
    """ Allow to reject samples with low predictive confidence by not assigning a class label.

    Parameter: \\
    probs:          Predicted class probabilities (num_samples, num_classes) \\
    thr:            Minimum differences between probability scores for the two most likely classes \\

    Returns: \\
    np.array
    """

    predicted_labels = []
    for i in range(probs.shape[0]):

        x_prob = probs[i,:]
        idx = np.argsort(x_prob)

        if (x_prob[idx[-1]] - x_prob[idx[-2]]) > thr:
            predicted_labels.append(idx[-1])
        else:
            predicted_labels.append(-1)
    
    return np.array(predicted_labels)


def eval(model, data, labels, display_labels=["Group 1", "Group 2"], model_name="CNN_train", save_path="./Baseline Models/results", with_reject=False):
    """ Evaluate classification results with various metrics.
    Parameter: \\
    model (keras model):        Fitted keras model \\
    data (np.array):            Array of processed LC-MS data or Generator object that produces data \\
    labels (np.array):          Array of class labels \\
    display_labels (list):      List of class label names for display\\
    model_name (str):           Name of current run, will be used to save results \\
    save_path (str):            Path to directory where results will be stored \\
    with_reject (bool):         Use reject option for prediction

    Returns: \\
    """

    # predict CoD of sample
    y_probs = model.predict(data)
    y_pred = None
    class_labels = None
    if with_reject:
        y_pred = predict_with_reject(y_probs)
        class_labels = list(range(len(display_labels)))
        class_labels.append(-1)
        display_labels.append("Unclassified")
    else:
        y_pred = np.argmax(y_probs, axis=1)
        class_labels = list(range(len(display_labels)))   

    # get ground truth
    y_true = np.argmax(labels, axis=1)

    # create and save confusion matrix plots
    cm_num = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=class_labels, normalize="true")
    cm_num_plot = ConfusionMatrixDisplay(confusion_matrix=cm_num, display_labels=display_labels)
    cm_norm_plot = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=display_labels)

    cm_num_plot.plot()
    cm_num_plot.figure_.savefig(save_path + f"/{model_name}_cm_numeric.png")
    plt.close()
    cm_norm_plot.plot()
    cm_norm_plot.figure_.savefig(save_path + f"/{model_name}_cm_normalized.png")
    plt.close()

    # create and save classification report
    # class-wise metrics
    f1_scores = []
    accuracies = []
    precisions = []
    recalls = []
    specificities = []
    roc_scores = []
    for i in range(len(np.unique(y_true))):
            temp_true = [1 if x == i else 0 for x in y_true]
            temp_pred = [1 if x == i else 0 for x in y_pred]
            
            f1_scores.append(f1_score(temp_true, temp_pred, average="binary"))
            accuracies.append(accuracy_score(temp_true, temp_pred))
            precisions.append(precision_score(temp_true, temp_pred, average="binary"))
            recalls.append(recall_score(temp_true, temp_pred, average="binary"))
            specificities.append(specificity_score(temp_true, temp_pred, average="binary"))
            roc_scores.append(roc_auc_score(temp_true, temp_pred))

    dict = {'F1-Score': f1_scores, 'Accuracy': accuracies, 'Precision': precisions, "Recall": recalls, "Specificity": specificities, "AUROC": roc_scores}
    class_metrics = pd.DataFrame(dict, index=pd.Index(display_labels[:5]))
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
            
    weighted_f1_scores.append(f1_score(y_true, y_pred, average="weighted", labels=np.unique(y_true)))
    macro_f1_scores.append(f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true)))
    accuracies.append(accuracy_score(y_true, y_pred))
    weighted_precisions.append(precision_score(y_true, y_pred, average="weighted", labels=np.unique(y_true)))
    macro_precisions.append(precision_score(y_true, y_pred, average="macro", labels=np.unique(y_true)))
    weighted_recalls.append(recall_score(y_true, y_pred, average="weighted", labels=np.unique(y_true)))
    macro_recalls.append(recall_score(y_true, y_pred, average="macro", labels=np.unique(y_true)))
    weighted_specificities.append(specificity_score(y_true, y_pred, average="weighted", labels=np.unique(y_true)))
    macro_specificities.append(specificity_score(y_true, y_pred, average="macro", labels=np.unique(y_true)))


    dict = {'(Macro) F1-Score': macro_f1_scores, '(Weighted) F1-Score': weighted_f1_scores, 'Accuracy': accuracies, '(Macro) Precision': macro_precisions, '(Weighted) Precision': weighted_precisions, "(Macro) Recall": macro_recalls, "(Weighted) Recall": weighted_recalls, "(Macro) Specificity": macro_specificities, "(Weighted) Specificity": weighted_specificities}
    overall_metrics = pd.DataFrame(dict)
    overall_metrics.to_csv(save_path + f"/{model_name}_metrics.csv")


def eval_ensemble(model, data, labels, save_path="./Baseline Models/results", max_members=25):
    """ Evaluate classification results per number of ensemble members

    Parameter: \\
    model (keras model):        Fitted keras model \\
    data (np.array):            Array of processed LC-MS data or Generator object that produces data \\
    labels (np.array):          Array of class labels \\
    save_path (str):            Path to directory where results will be stored
    max_members (int):          Maximum number of ensemble members

    Returns: \\
    """
     
    res_mean = pd.DataFrame()
    res_std = pd.DataFrame()
    model_ids = [[], [], [], [], []]
    all_models = np.array(model.models)

    # get ground truth
    y_true = np.argmax(labels, axis=1)

    # compute results per number of ensemble members
    for i in range(1,max_members):

        weighted_f1_scores = []
        macro_f1_scores = []
        accuracies = []
        macro_precisions = []
        weighted_precisions = []
        macro_recalls = []
        weighted_recalls = []
        macro_specificities = []
        weighted_specificities = []

        # repeat evaluation with 5 different ensemble models
        for j in range(5):
            # randomly get +1 esemble members
            model_ids[j].extend(random.sample(list(set(list(range(0,30))) - set(model_ids[j])), 1))
            model.models = all_models[model_ids[j]]

            # predict samples
            y_probs = model.predict(data)
            y_pred = np.argmax(y_probs, axis=1)

            # compute metrics 
            weighted_f1_scores.append(f1_score(y_true, y_pred, average="weighted", labels=np.unique(y_true)))
            macro_f1_scores.append(f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true)))
            accuracies.append(accuracy_score(y_true, y_pred))
            weighted_precisions.append(precision_score(y_true, y_pred, average="weighted", labels=np.unique(y_true)))
            macro_precisions.append(precision_score(y_true, y_pred, average="macro", labels=np.unique(y_true)))
            weighted_recalls.append(recall_score(y_true, y_pred, average="weighted", labels=np.unique(y_true)))
            macro_recalls.append(recall_score(y_true, y_pred, average="macro", labels=np.unique(y_true)))
            weighted_specificities.append(specificity_score(y_true, y_pred, average="weighted", labels=np.unique(y_true)))
            macro_specificities.append(specificity_score(y_true, y_pred, average="macro", labels=np.unique(y_true)))

        # compute average + std deviation
        dict_mean = {'Ensemble Members': [i], '(Macro) F1-Score': [np.mean(macro_f1_scores)], '(Weighted) F1-Score': [np.mean(weighted_f1_scores)], 'Accuracy': [np.mean(accuracies)], '(Macro) Precision': [np.mean(macro_precisions)], '(Weighted) Precision': [np.mean(weighted_precisions)], "(Macro) Recall": [np.mean(macro_recalls)], "(Weighted) Recall": [np.mean(weighted_recalls)], "(Macro) Specificity": [np.mean(macro_specificities)], "(Weighted) Specificity": [np.mean(weighted_specificities)]}
        dict_std = {'Ensemble Members': [i], '(Macro) F1-Score': [np.std(macro_f1_scores)], '(Weighted) F1-Score': [np.std(weighted_f1_scores)], 'Accuracy': [np.std(accuracies)], '(Macro) Precision': [np.std(macro_precisions)], '(Weighted) Precision': [np.std(weighted_precisions)], "(Macro) Recall": [np.std(macro_recalls)], "(Weighted) Recall": [np.std(weighted_recalls)], "(Macro) Specificity": [np.std(macro_specificities)], "(Weighted) Specificity": [np.std(weighted_specificities)]}
        metrics_mean = pd.DataFrame(dict_mean)
        metrics_std = pd.DataFrame(dict_std)
        res_mean = pd.concat([res_mean, metrics_mean])
        res_std = pd.concat([res_std, metrics_std])

    # save results
    num_members = res_mean["Ensemble Members"]
    mean_f1 = res_mean["(Macro) F1-Score"]
    std_f1 = res_std["(Macro) F1-Score"]

    plt.plot(num_members, mean_f1, label='Mean', color='blue')
    plt.fill_between(num_members, mean_f1 - std_f1, mean_f1 + std_f1, color='blue', alpha=0.2, label='Std. Dev')
    plt.xlabel('Number of ensemble members')
    plt.ylabel('Macro F1-score')
    plt.title('')
    plt.savefig(save_path + "/eval_ensemble.png")
    plt.close()

    res_mean.to_csv(save_path + f"/ensemble_members_mean.csv")
    res_std.to_csv(save_path + f"/ensemble_members_std.csv")


def get_pred_with_contributing_cod(model, data, meta_df):
    """ Predict samples and retrieve contributing CoD information

    Parameter: \\
    model (keras model):        Fitted keras model \\
    data (np.array):            Array of processed LC-MS data or Generator object that produces data \\
    meta_df (pd.DataFrame):     Pandas dataframe with meta information of test samples. \\

    Returns: \\
    pd.DataFrame
    """
    
    # available class labels
    class_labels = ["Acidosis", "Drug", "Hanging", "IHD", "Pneumonia"]

    # predict CoD of sample
    y_probs = model.predict(data)
    y_pred = np.argmax(y_probs, axis=1)

    # get ground truth
    y_true = np.unique(meta_df["Group"], return_inverse=True)[1]

    # get ICD9 codes for each CoD
    meta = pd.read_excel("./CoD Prediction/meta.xlsx", index_col=0)
    icd9_codes_per_group = meta.groupby('Group')['Code 1A'].agg(['unique'])
    all_icd9 = [item for sublist in icd9_codes_per_group['unique'] for item in sublist]

    # analyse false positives per class
    preds = pd.DataFrame()
    false_positives = pd.DataFrame()

    for i in range(len(class_labels)):
        curr_idx = np.where(y_true == i)[0]
        curr_preds = y_pred[curr_idx]
        curr_meta = meta_df.iloc[curr_idx]

        acidosis = 0
        drug = 0
        ihd = 0
        hanging = 0
        pneumonia = 0
        none = 0
        other = 0
        
        for idx in range(len(curr_idx)):
            # get all contributing CoDs
            contributing_cods = [curr_meta.iloc[idx]["Code 2A"], curr_meta.iloc[idx]["Code 2B"], curr_meta.iloc[idx]["Code 2C"]]
            contributing_cods = [x for x in contributing_cods if x is not np.nan]

            # get primary CoD
            primary_cods = [curr_meta.iloc[idx]["Code 1A"], curr_meta.iloc[idx]["Code 1B"], curr_meta.iloc[idx]["Code 1C"], curr_meta.iloc[idx]["Code 1D"]]
            primary_cods = [x for x in primary_cods if x is not np.nan]
            
            # save formated meta information for each sample
            if any(x in icd9_codes_per_group.loc["Acidosis"].values[0] for x in contributing_cods):
                idx_df = pd.DataFrame({"Group": [class_labels[i]], "Pred": [class_labels[curr_preds[idx]]], "Secondary CoD": ["Acidosis"], "Codes_1": [primary_cods], "Codes_2": [contributing_cods], "Sample": [curr_meta.iloc[idx].name]})
                false_positives = pd.concat([false_positives, idx_df])
            if any(x in icd9_codes_per_group.loc["Drug"].values[0] for x in contributing_cods):
                idx_df = pd.DataFrame({"Group": [class_labels[i]], "Pred": [class_labels[curr_preds[idx]]], "Secondary CoD": ["Drug"], "Codes_1": [primary_cods], "Codes_2": [contributing_cods], "Sample": [curr_meta.iloc[idx].name]})
                false_positives = pd.concat([false_positives, idx_df])
            if any(x in icd9_codes_per_group.loc["IHD"].values[0] for x in contributing_cods):
                idx_df = pd.DataFrame({"Group": [class_labels[i]], "Pred": [class_labels[curr_preds[idx]]], "Secondary CoD": ["IHD"], "Codes_1": [primary_cods], "Codes_2": [contributing_cods], "Sample": [curr_meta.iloc[idx].name]})
                false_positives = pd.concat([false_positives, idx_df])
            if any(x in icd9_codes_per_group.loc["Hanging"].values[0] for x in contributing_cods):
                idx_df = pd.DataFrame({"Group": [class_labels[i]], "Pred": [class_labels[curr_preds[idx]]], "Secondary CoD": ["Hanging"], "Codes_1": [primary_cods], "Codes_2": [contributing_cods], "Sample": [curr_meta.iloc[idx].name]})
                false_positives = pd.concat([false_positives, idx_df])
            if any(x in icd9_codes_per_group.loc["Pneumonia"].values[0] for x in contributing_cods):
                idx_df = pd.DataFrame({"Group": [class_labels[i]], "Pred": [class_labels[curr_preds[idx]]], "Secondary CoD": ["Pneumonia"], "Codes_1": [primary_cods], "Codes_2": [contributing_cods], "Sample": [curr_meta.iloc[idx].name]})
                false_positives = pd.concat([false_positives, idx_df])
            if all((x not in all_icd9) for x in contributing_cods) & (len(contributing_cods) != 0):
                idx_df = pd.DataFrame({"Group": [class_labels[i]], "Pred": [class_labels[curr_preds[idx]]], "Secondary CoD": ["Other"], "Codes_1": [primary_cods], "Codes_2": [contributing_cods], "Sample": [curr_meta.iloc[idx].name]})
                false_positives = pd.concat([false_positives, idx_df])
            if len(contributing_cods) == 0:
                idx_df = pd.DataFrame({"Group": [class_labels[i]], "Pred": [class_labels[curr_preds[idx]]], "Secondary CoD": ["None"], "Codes_1": [primary_cods], "Codes_2": [contributing_cods], "Sample": [curr_meta.iloc[idx].name]})
                false_positives = pd.concat([false_positives, idx_df])

        curr_df = pd.DataFrame({"Group": class_labels[i], "Contributing Acidosis": [acidosis], "Contributing Drug": [drug], "Contributing IHD": [ihd], "Contributing None": [none]})
        preds = pd.concat([preds, curr_df])

    return preds


def save_split_data(samples_train, samples_test, save_dir):
    """ Save train test split (by sample ids) at specified path.

    Parameter: \\
    samples_train (np.array):        Array of sample identifiers of the training dataset \\
    samples_test (np.array):         Array of sample identifiers of the test dataset \\
    save_dir (str):                  Path to diractory where train test split will be save 
    """

    # check if path is valid
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save train and test lists
    with open(os.path.join(save_dir, 'samples_train.txt'), 'w') as fw:
        fw.write('\n'.join(samples_train))
    with open(os.path.join(save_dir, 'samples_test.txt'), 'w') as fw:
        fw.write('\n'.join(samples_test))


def split_samples(samples, labels, split_size=0.2, seed=None):

    """ Split sample identifiers into (stratified) train/test subsets. 

    Parameter: \\
    samples (np.array):     Array of sample identifiers \\
    labels (np.array):      Array of class labels \\
    split_size (float):     Size of test dataset \\
    seed (int):             Split seed for reproducability

    Return: \\
    list:                   Training sample identifiers
    list:                   Test sample identifier
    """

    # split dataset
    samples_train, samples_test, y_train, y_test = train_test_split(samples, labels.to_list(), stratify=labels.to_list(), test_size=split_size, random_state=seed)

    return samples_train.to_list(), samples_test.to_list() 


def create_args_df(args):
    """ Creates df of arguments of a training run.

    Parameter: \\
    args (list):        List of arguments for a training run

    Return: \\
    pd.DataFrame:       Arguments DataFrame
    """

    # retrieve arguments
    args_dict = vars(args)
    args_dict['script'] = sys.argv[0]
    args_dict['time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    args_dict['platform'] = platform.platform()
    args_dict['python'] = platform.python_version()

    # create df
    args_df = pd.DataFrame()
    args_df['key'] = args_dict.keys()
    args_df['value'] = args_dict.values()

    return args_df

