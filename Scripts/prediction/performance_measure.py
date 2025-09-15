import sys
import os
from pycm import ConfusionMatrix
from sklearn import metrics
import math
import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.inspection import permutation_importance


HEADER = "Dataset,Project,Train Version,Test Version,ML Algorithm,Sampling Type,Sampling Method,Dimensionality Reduction,Defect Ratio,Validation,Test,Parameters,Accuracy,F1,AUC,MCC,Precision,Recall(pd),TNR-Specificity,pf,bal,g-mean,gmeasure,auc-prc,tn,fp,fn,tp"
HEADER_MUTATED = "Dataset,Project,Train Version,Test Version,ML Algorithm,Max Bug Per File,Mutation Version,Dimensionality Reduction,Defect Ratio,Validation,Test,Parameters,Accuracy,F1,AUC,MCC,Precision,Recall(pd),TNR-Specificity,pf,bal,g-mean,g-measure,auc-prc,tn,fp,fn,tp"
HEADER_ATTRIBUTES = ",wmc,dit,noc,cbo,rfc,lcom,ca,ce,npm,lcom3,loc,dam,moa,mfa,cam,ic,cbm,amc,max_cc,avg_cc"
FEATURE_IMPORTANCES_ENABLED = False
RANDOM_STATE = 42


class PerformanceMeasureCSV:
    def __init__(self, csv_path, is_mutated=False):
        self.csv_path = csv_path
        self.data = {}
        self.header = HEADER
        if is_mutated:
            self.header = HEADER_MUTATED
        if os.path.exists(csv_path):
            self.read_csv(csv_path)

    def read_csv(self, csv_path):
        file = open(csv_path, "r")
        lines = file.readlines()
        file.close()

        if lines[0][:-1] != self.header:
            print("Invalid Perfomance Measure file header!")
            sys.exit(os.EX_IOERR)

        for line in lines[1:]:
            row = PerformanceMeasureRow([line])
            self.data[row.row_id] = row

    def append_measure(self, row):
        self.data[row.row_id] = row

    def save_csv(self):
        file = open(self.csv_path, "w")
        column_count = list(self.data.values())[0].get_line().count(",")
        if column_count > self.header.count(","):
            self.header += HEADER_ATTRIBUTES
        elif column_count < self.header.count(","):
            print("ERROR ! HEADER FAIL")
            sys.exit()
        file.write(self.header + "\n")
        for key, val in self.data.items():
            file.write(val.get_line())
        file.close()


class PerformanceMeasureRow:
    def __init__(self, row):
        if len(row) == 1:
            row = row[0]
            tokens = row.split(",")
            self.dataset = tokens[0]
            self.project = tokens[1]
            self.train_version = tokens[2]
            self.test_version = tokens[3]
            self.ml_algorithm = tokens[4]
            self.sampling_type = tokens[5]
            self.sampling = tokens[6]
            self.dimensionality_reduction = tokens[7]
            self.ratio = tokens[8]
            self.validation = tokens[9]
            self.test = tokens[10]
            self.parameters = tokens[11].replace(",", ";")
            self.accuracy = tokens[12]
            self.f1 = tokens[13]
            self.auc = tokens[14]
            self.mcc = tokens[15]
            self.precision = tokens[16]
            self.recall = tokens[17]
            self.tnr = tokens[18]
            self.pf = tokens[19]
            self.bal = tokens[20]
            self.gmean = tokens[21]
            self.gmeasure = tokens[22]
            self.auc_prc = tokens[23]
            self.tn = tokens[24]
            self.fp = tokens[25]
            self.fn = tokens[26]
            self.tp = tokens[27][:-1]
            # TODO :: feature importances is ignored for reading
            self.row_id = (
                str(self.dataset)
                + str(self.sampling)
                + str(self.sampling_type)
                + str(self.dimensionality_reduction)
                + str(self.ratio)
                + str(self.validation)
                + str(self.test)
                + str(self.parameters)
            )
        else:
            self.dataset = row[0]
            self.project = row[1]
            self.train_version = row[2]
            self.test_version = row[3]
            self.ml_algorithm = row[4]
            self.sampling_type = row[5]
            self.sampling = row[6]
            self.dimensionality_reduction = row[7]
            self.ratio = row[8]
            self.validation = row[9]
            self.test = row[10]
            self.parameters = row[11].replace(",", ";")
            self.accuracy = row[12]
            self.f1 = row[13]
            self.auc = row[14]
            self.mcc = row[15]
            self.precision = row[16]
            self.recall = row[17]
            self.tnr = row[18]
            self.pf = row[19]
            self.bal = row[20]
            self.gmean = row[21]
            self.gmeasure = row[22]
            self.importances = row[23]
            self.auc_prc = row[24]
            self.tn = row[25]
            self.fp = row[26]
            self.fn = row[27]
            self.tp = row[28]
            self.row_id = (
                str(self.dataset)
                + str(self.sampling)
                + str(self.sampling_type)
                + str(self.dimensionality_reduction)
                + str(self.ratio)
                + str(self.validation)
                + str(self.test)
                + str(self.parameters)
            )

    def get_line(self):
        return_str = (
            str(self.dataset)
            + ","
            + str(self.project)
            + ","
            + str(self.train_version)
            + ","
            + str(self.test_version)
            + ","
            + str(self.ml_algorithm)
            + ","
            + str(self.sampling_type)
            + ","
            + str(self.sampling)
            + ","
            + str(self.dimensionality_reduction)
            + ","
            + str(self.ratio)
            + ","
            + str(self.validation)
            + ","
            + str(self.test)
            + ","
            + str(self.parameters)
            + ","
            + str(self.accuracy)
            + ","
            + str(self.f1)
            + ","
            + str(self.auc)
            + ","
            + str(self.mcc)
            + ","
            + str(self.precision)
            + ","
            + str(self.recall)
            + ","
            + str(self.tnr)
            + ","
            + str(self.pf)
            + ","
            + str(self.bal)
            + ","
            + str(self.gmean)
            + ","
            + str(self.gmeasure)
            + ","
            + str(self.auc_prc)
            + ","
            + str(self.tn)
            + ","
            + str(self.fp)
            + ","
            + str(self.fn)
            + ","
            + str(self.tp)
        )
        if self.importances != []:
            return_str += ","
            for i in self.importances:
                return_str += str(i) + ","
            return_str = return_str[:-1]
        return return_str + "\n"


def calculate_bal(pf, pd):
    return 1 - (math.sqrt((0 - pf) ** 2 + (1 - pd) ** 2) / math.sqrt(2))


def calculate_performance_metrics(
    f,
    clf,
    X,
    y,
    project,
    train_version,
    test_version,
    ml_algorithm,
    sampling_type,
    sampling_name,
    dimensionality_reduction,
    ratio,
    val,
    test,
    parameters,
    y_gt,
    y_pred,
):
    cm = ConfusionMatrix(actual_vector=y_gt, predict_vector=y_pred)

    acc = metrics.accuracy_score(y_true=y_gt, y_pred=y_pred)
    f1 = metrics.f1_score(y_true=y_gt, y_pred=y_pred)
    try:
        auc = metrics.roc_auc_score(y_true=y_gt, y_score=y_pred)
    except:
        auc = 0
    mcc = metrics.matthews_corrcoef(y_true=y_gt, y_pred=y_pred)
    precision = metrics.precision_score(y_true=y_gt, y_pred=y_pred)
    recall = metrics.recall_score(y_true=y_gt, y_pred=y_pred)
    try:
        tnr = cm.class_stat["TNR"][1]
        if tnr == "None":
            tnr = 0
    except:
        print(f)
        tnr = 0
    tn, fp, fn, tp = metrics.confusion_matrix(y_gt, y_pred).ravel().tolist()
    pf = fp / (tn + fp)
    bal = calculate_bal(pf, recall)
    gmean = geometric_mean_score(y_true=y_gt, y_pred=y_pred)
    gmeasure = metrics.fowlkes_mallows_score(y_gt, y_pred)
    auc_prc = metrics.average_precision_score(y_true=y_gt, y_score=y_pred)

    result_importances = []
    if FEATURE_IMPORTANCES_ENABLED:
        result_importances = permutation_importance(
            clf, X, y, scoring="roc_auc", random_state=RANDOM_STATE
        ).importances_mean

    return PerformanceMeasureRow(
        [
            f,
            project,
            train_version,
            test_version,
            ml_algorithm,
            sampling_type,
            sampling_name,
            dimensionality_reduction,
            ratio,
            val,
            test,
            parameters,
            acc,
            f1,
            auc,
            mcc,
            precision,
            recall,
            tnr,
            pf,
            bal,
            gmean,
            gmeasure,
            result_importances,
            auc_prc,
            tn,
            fp,
            fn,
            tp,
        ]
    )


def loss_function_bal(y_gt, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_gt, y_pred).ravel()
    pf = fp / (tn + fp)
    pd = tp / (tp + fn)
    return calculate_bal(pf, pd)


def loss_function_auc(y_gt, y_pred):
    try:
        auc = metrics.roc_auc_score(y_true=y_gt, y_score=y_pred)
    except:
        auc = 0
    return auc

def loss_function_auc_prc(y_gt, y_pred):
    try:
        auc_prc = metrics.average_precision_score(y_true=y_gt, y_score=y_pred)
    except:
        auc_prc = 0
    return auc_prc

def loss_function_mcc(y_gt, y_pred):
    try:
        mcc = metrics.matthews_corrcoef(y_true=y_gt, y_pred=y_pred)
    except:
        mcc = 0
    return mcc

def get_best_index_5split(cv_results):
    idx = cv_results["split0_test_score"].argmax()
    val = cv_results["split0_test_score"][idx]

    tmp_idx = np.argmax(cv_results["split1_test_score"])
    tmp_val = cv_results["split1_test_score"][tmp_idx]
    if tmp_val > val:
        idx = tmp_idx
    tmp_idx = np.argmax(cv_results["split2_test_score"])
    tmp_val = cv_results["split2_test_score"][tmp_idx]
    if tmp_val > val:
        idx = tmp_idx
    tmp_idx = np.argmax(cv_results["split3_test_score"])
    tmp_val = cv_results["split3_test_score"][tmp_idx]
    if tmp_val > val:
        idx = tmp_idx
    tmp_idx = np.argmax(cv_results["split4_test_score"])
    tmp_val = cv_results["split4_test_score"][tmp_idx]
    if tmp_val > val:
        idx = tmp_idx

    return idx
