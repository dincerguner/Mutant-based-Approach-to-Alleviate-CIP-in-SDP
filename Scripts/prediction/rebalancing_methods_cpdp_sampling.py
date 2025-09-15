import os
from decision_tree import apply_decision_tree_cross_release
from knn import apply_knn_cross_release
from naive_bayes import apply_naive_bayes_cross_release
from random_forest import apply_random_forest_cross_release
from svm import apply_svm_cross_release
from xg_boost import apply_xgboost_cross_release
from nn import apply_neuralnet_cross_release

from performance_measure import PerformanceMeasureCSV
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    SMOTEN,
    BorderlineSMOTE,
    SVMSMOTE,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE, MDS
from dataloader import create_merged_data_folder

DATA_FOLDER =  "../../Software Measures Datasets"
SAMPLED_DATA_FOLDER = (
    "sampled_dataset"
)
RESULT_CSV_PATH = "../../Performance Results/Rebalancing Methods/CPDP/performance_cpdp_sampling_71.csv"
RANDOM_STATE = 71 # 42, 51, 92, 14, 71
TEMP_PATH = "temp71"

save_file = PerformanceMeasureCSV(RESULT_CSV_PATH)
dataset_mapping_train = {
    "ant": [
        "ant_1.3_org_java_1.3_ext_2.2_j16_45.csv",
        "ant_1.4_org_java_1.3_ext_2.2_j16_45.csv",
        "ant_1.5_org_java_1.3_ext_2.2_j16_45.csv",
        "ant_1.6_org_java_1.4.1_ext_2.2_j16_45.csv",
        "ant_1.7_org_java_1.5_ext_2.2_j16_45.csv",
    ],
    "jEdit": [
        "jEdit_3.2.1_org_java_1.3_ext_2.2_j16_45.csv",
        "jEdit_4.0_org_java_1.4.0_ext_2.2_j16_45.csv",
        "jEdit_4.1_org_java_1.4.0_ext_2.2_j16_45.csv",
        "jEdit_4.2_org_java_1.5_ext_2.2_j16_45.csv",
        "jEdit_4.3_org_java_1.5_ext_2.2_j16_45.csv",
    ],
    "lucene": ["lucene_2.0_org_java_1.4.2_ext_2.2_j16_45.csv"],
    "pBeans": ["pBeans_2.0_org_java_1.5_ext_2.2_j16_45.csv"],
    "poi": ["poi_2.0RC1_org_java_1.4.0_ext_2.2_j16_45.csv"],
    "synapse": [
        "synapse_1.1_org_java_1.5_ext_2.2_j16_45.csv",
        "synapse_1.2_org_java_1.5_ext_2.2_j16_45.csv",
    ],
    "velocity": ["velocity_1.6.1_org_java_1.4.0_ext_2.2_j16_45.csv"],
    "xalan": ["xalan_2.6_org_java_1.3_ext_2.2_j16_45.csv"],
    "xerces": [
        "xerces_1.2.0_org_java_1.3_ext_2.2_j16_45.csv",
        "xerces_1.3.0_org_java_1.3_ext_2.2_j16_45.csv",
    ],
}  # 19

dataset_mapping_test = {
    "ant": [
        "ant_1.3_org_java_1.3_ext_2.2_j16_45.csv",
        "ant_1.4_org_java_1.3_ext_2.2_j16_45.csv",
        "ant_1.5_org_java_1.3_ext_2.2_j16_45.csv",
        "ant_1.6_org_java_1.4.1_ext_2.2_j16_45.csv",
        "ant_1.7_org_java_1.5_ext_2.2_j16_45.csv",
    ],
    "jEdit": [
        "jEdit_3.2.1_org_java_1.3_ext_2.2_j16_45.csv",
        "jEdit_4.0_org_java_1.4.0_ext_2.2_j16_45.csv",
        "jEdit_4.1_org_java_1.4.0_ext_2.2_j16_45.csv",
        "jEdit_4.2_org_java_1.5_ext_2.2_j16_45.csv",
        "jEdit_4.3_org_java_1.5_ext_2.2_j16_45.csv",
    ],
    "lucene": [
        "lucene_2.0_org_java_1.4.2_ext_2.2_j16_45.csv",
        "lucene_2.2_org_java_1.5_ext_2.2_j16_45.csv",
        "lucene_2.4_org_java_1.5_ext_2.2_j16_45.csv",
    ],
    "pBeans": [
        "pBeans_1.0_org_java_1.4.0_ext_2.2_j16_45.csv",
        "pBeans_2.0_org_java_1.5_ext_2.2_j16_45.csv",
    ],
    "poi": [
        "poi_1.5_org_java_1.3_ext_2.2_j16_45.csv",
        "poi_2.0RC1_org_java_1.4.0_ext_2.2_j16_45.csv",
        "poi_2.5.1_org_java_1.4.0_ext_2.2_j16_45.csv",
        "poi_3.0_org_java_1.4.0_ext_2.2_j16_45.csv",
    ],
    "synapse": [
        "synapse_1.0_org_java_1.5_ext_2.2_j16_45.csv",
        "synapse_1.1_org_java_1.5_ext_2.2_j16_45.csv",
        "synapse_1.2_org_java_1.5_ext_2.2_j16_45.csv",
    ],
    "velocity": [
        "velocity_1.4_org_java_1.4.0_ext_2.2_j16_45.csv",
        "velocity_1.5_org_java_1.4.0_ext_2.2_j16_45.csv",
        "velocity_1.6.1_org_java_1.4.0_ext_2.2_j16_45.csv",
    ],
    "xalan": [
        "xalan_2.6_org_java_1.3_ext_2.2_j16_45.csv",
        "xalan_2.7_org_java_1.3_ext_2.2_j16_45.csv",
    ],
    "xerces": [
        "xerces_1.2.0_org_java_1.3_ext_2.2_j16_45.csv",
        "xerces_1.3.0_org_java_1.3_ext_2.2_j16_45.csv",
    ],
}  # 29

dimensionalityreductions = {
    "None": None,
    # 'PCA': PCA(random_state=RANDOM_STATE),
    # 'SVD': TruncatedSVD(random_state=RANDOM_STATE)
    # 'LDA': LinearDiscriminantAnalysis(),
    # 'tSNE': TSNE(random_state=RANDOM_STATE),
    # 'MDS': MDS(random_state=RANDOM_STATE)
}

for (
    dimensionalityreduction_name,
    dimensionalityreduction,
) in dimensionalityreductions.items():
    for train, versions in dataset_mapping_train.items():
        # OVER SAMPLE
        sampling_type = "over_sample"
        ratio = 0.5
        samplings = {'smote': SMOTE(sampling_strategy=ratio, random_state=RANDOM_STATE),
                                'random_over_sampler': RandomOverSampler(sampling_strategy=ratio, random_state=RANDOM_STATE),
                                'smote_n': SMOTEN(sampling_strategy=ratio, random_state=RANDOM_STATE),
                                'borderline_smote': BorderlineSMOTE(sampling_strategy=ratio, random_state=RANDOM_STATE),
                                'borderline_smote_svm': SVMSMOTE(sampling_strategy=ratio, random_state=RANDOM_STATE)}
        for name, sampling in samplings.items():
            print(train)
            train_path = os.path.join(
                TEMP_PATH, train + "_not_" + sampling_type + "_" + name + ".csv"
            )
            train_versions_path = []
            for train_, versions_ in dataset_mapping_train.items():
                if train == train_:
                    continue
                else:
                    for version in versions_:
                        train_versions_path.append(os.path.join(DATA_FOLDER, version))
            create_merged_data_folder(train_versions_path, train_path)

            t_ratio = 0.5
            for test in dataset_mapping_test[train]:
                try:
                    out = apply_knn_cross_release(
                        train_path,
                        os.path.join(DATA_FOLDER, test),
                        sampling_type,
                        name,
                        t_ratio,
                        sampling,
                        dimensionalityreduction,
                        dimensionalityreduction_name,
                        is_sampling=True,
                    )
                    for pmRow in out:
                        save_file.append_measure(pmRow)
                except:
                    print("error")
                try:
                    out = apply_naive_bayes_cross_release(
                        train_path,
                        os.path.join(DATA_FOLDER, test),
                        sampling_type,
                        name,
                        t_ratio,
                        sampling,
                        dimensionalityreduction,
                        dimensionalityreduction_name,
                        is_sampling=True,
                    )
                    for pmRow in out:
                        save_file.append_measure(pmRow)
                except:
                    print("error")
                try:
                    out = apply_decision_tree_cross_release(
                        train_path,
                        os.path.join(DATA_FOLDER, test),
                        sampling_type,
                        name,
                        t_ratio,
                        sampling,
                        dimensionalityreduction,
                        dimensionalityreduction_name,
                        is_sampling=True,
                    )
                    for pmRow in out:
                        save_file.append_measure(pmRow)
                except:
                    print("error")
                try:
                    out = apply_random_forest_cross_release(
                        train_path,
                        os.path.join(DATA_FOLDER, test),
                        sampling_type,
                        name,
                        t_ratio,
                        sampling,
                        dimensionalityreduction,
                        dimensionalityreduction_name,
                        is_sampling=True,
                    )
                    for pmRow in out:
                        save_file.append_measure(pmRow)
                except:
                    print("error")
                try:
                    out = apply_svm_cross_release(
                        train_path,
                        os.path.join(DATA_FOLDER, test),
                        sampling_type,
                        name,
                        t_ratio,
                        sampling,
                        dimensionalityreduction,
                        dimensionalityreduction_name,
                        is_sampling=True,
                    )
                    for pmRow in out:
                        save_file.append_measure(pmRow)
                except:
                    print("error")
                try:
                    out = apply_xgboost_cross_release(
                        train_path,
                        os.path.join(DATA_FOLDER, test),
                        sampling_type,
                        name,
                        t_ratio,
                        sampling,
                        dimensionalityreduction,
                        dimensionalityreduction_name,
                        is_sampling=True,
                    )
                    for pmRow in out:
                        save_file.append_measure(pmRow)
                except:
                    print("error")
                try:
                    out = apply_neuralnet_cross_release(
                        train_path,
                        os.path.join(DATA_FOLDER, test),
                        sampling_type,
                        name,
                        t_ratio,
                        sampling,
                        dimensionalityreduction,
                        dimensionalityreduction_name,
                        is_sampling=True,
                    )
                    for pmRow in out:
                        save_file.append_measure(pmRow)
                except:
                    print("error")

save_file.save_csv()
