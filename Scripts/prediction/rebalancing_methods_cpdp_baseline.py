import os
from decision_tree import apply_decision_tree_cross_release
from knn import apply_knn_cross_release
from naive_bayes import apply_naive_bayes_cross_release
from random_forest import apply_random_forest_cross_release
from svm import apply_svm_cross_release
from xg_boost import apply_xgboost_cross_release
from nn import apply_neuralnet_cross_release

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE, MDS
from performance_measure import PerformanceMeasureCSV
from dataloader import DataLoader, create_merged_data_folder
from collections import Counter


DATA_FOLDER = "../../Software Measures Datasets"
RESULT_CSV_PATH = (
    "../../Performance Results/Rebalancing Methods/CPDP/performance_cpdp_baseline.csv"
)
RANDOM_STATE = 42
TEMP_PATH = "temp"

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
    "None": None
    # 'PCA': PCA(random_state=RANDOM_STATE),
    # 'SVD': TruncatedSVD(random_state=RANDOM_STATE)
    #'LDA': LinearDiscriminantAnalysis(),
    #'tSNE': TSNE(random_state=RANDOM_STATE),
    #'MDS': MDS(random_state=RANDOM_STATE)
}

for (
    dimensionalityreduction_name,
    dimensionalityreduction,
) in dimensionalityreductions.items():
    for train, versions in dataset_mapping_train.items():
        print(train)
        train_path = os.path.join(TEMP_PATH, train + "_not" + ".csv")
        train_versions_path = []
        for train_, versions_ in dataset_mapping_train.items():
            if train == train_:
                continue
            else:
                for version in versions_:
                    train_versions_path.append(os.path.join(DATA_FOLDER, version))
        create_merged_data_folder(train_versions_path, train_path)

        raw_data = DataLoader(train_path)
        counter = Counter(raw_data.y)
        total_file_count = counter[0] + counter[1]
        ratio = counter[1] / total_file_count
        mbpf_val = "baseline"
        mutation_version = "None"

        for test in dataset_mapping_test[train]:
            try:
                out = apply_knn_cross_release(
                    train_path,
                    os.path.join(DATA_FOLDER, test),
                    mbpf_val,
                    mutation_version,
                    ratio,
                    None,
                    dimensionalityreduction,
                    dimensionalityreduction_name,
                    is_sampling=False,
                )
                for pmRow in out:
                    save_file.append_measure(pmRow)
            except:
                print("error")
            try:
                out = apply_naive_bayes_cross_release(
                    train_path,
                    os.path.join(DATA_FOLDER, test),
                    mbpf_val,
                    mutation_version,
                    ratio,
                    None,
                    dimensionalityreduction,
                    dimensionalityreduction_name,
                    is_sampling=False,
                )
                for pmRow in out:
                    save_file.append_measure(pmRow)
            except:
                print("error")
            try:
                out = apply_decision_tree_cross_release(
                    train_path,
                    os.path.join(DATA_FOLDER, test),
                    mbpf_val,
                    mutation_version,
                    ratio,
                    None,
                    dimensionalityreduction,
                    dimensionalityreduction_name,
                    is_sampling=False,
                )
                for pmRow in out:
                    save_file.append_measure(pmRow)
            except:
                print("error")
            try:
                out = apply_random_forest_cross_release(
                    train_path,
                    os.path.join(DATA_FOLDER, test),
                    mbpf_val,
                    mutation_version,
                    ratio,
                    None,
                    dimensionalityreduction,
                    dimensionalityreduction_name,
                    is_sampling=False,
                )
                for pmRow in out:
                    save_file.append_measure(pmRow)
            except:
                print("error")
            try:
                out = apply_svm_cross_release(
                    train_path,
                    os.path.join(DATA_FOLDER, test),
                    mbpf_val,
                    mutation_version,
                    ratio,
                    None,
                    dimensionalityreduction,
                    dimensionalityreduction_name,
                    is_sampling=False,
                )
                for pmRow in out:
                    save_file.append_measure(pmRow)
            except:
                print("error")
            try:
                out = apply_xgboost_cross_release(
                    train_path,
                    os.path.join(DATA_FOLDER, test),
                    mbpf_val,
                    mutation_version,
                    ratio,
                    None,
                    dimensionalityreduction,
                    dimensionalityreduction_name,
                    is_sampling=False,
                )
                for pmRow in out:
                    save_file.append_measure(pmRow)
            except:
                print("error")
            try:
                out = apply_neuralnet_cross_release(
                    train_path,
                    os.path.join(DATA_FOLDER, test),
                    mbpf_val,
                    mutation_version,
                    ratio,
                    None,
                    dimensionalityreduction,
                    dimensionalityreduction_name,
                    is_sampling=False,
                )
                for pmRow in out:
                    save_file.append_measure(pmRow)
            except:
                print("error")

save_file.save_csv()
