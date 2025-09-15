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
from dataloader import create_merged_data_folder


DATA_FOLDER = "../../Software Measures Datasets"
RESULT_CSV_PATH = (
    "../../Performance Results/Stability of MBA/CPDP/performance_irdp_mutated.csv"
)
RANDOM_STATE = 42
TEMP_PATH = "temp2"


save_file = PerformanceMeasureCSV(RESULT_CSV_PATH, is_mutated=True)
dataset_mapping_train = {
    "ant": [
        "ant_1.3_org_java_1.3_ext_2.2_j16_45.csv",
        "ant_1.4_org_java_1.3_ext_2.2_j16_45.csv",
        "ant_1.5_org_java_1.3_ext_2.2_j16_45.csv",
        "ant_1.6_org_java_1.4.1_ext_2.2_j16_45.csv",
        "ant_1.7_org_java_1.5_ext_2.2_j16_45.csv",
    ],
    "jEdit": [
        "jEdit_4.0_org_java_1.4.0_ext_2.2_j16_45.csv",
        "jEdit_4.1_org_java_1.4.0_ext_2.2_j16_45.csv",
        "jEdit_4.2_org_java_1.5_ext_2.2_j16_45.csv",
        "jEdit_4.3_org_java_1.5_ext_2.2_j16_45.csv",
    ],
    "all": [],
    "pBeans": ["pBeans_2.0_org_java_1.5_ext_2.2_j16_45.csv"],
    "poi": ["poi_2.0RC1_org_java_1.4.0_ext_2.2_j16_45.csv"],
    "synapse": ["synapse_1.1_org_java_1.5_ext_2.2_j16_45.csv"],
    "xerces": [
        "xerces_1.2.0_org_java_1.3_ext_2.2_j16_45.csv",
        "xerces_1.3.0_org_java_1.3_ext_2.2_j16_45.csv",
    ],
}  # 19

test_sets_for_all_cases = (
    [
        "lucene_2.0_org_java_1.4.2_ext_2.2_j16_45.csv",
        "lucene_2.2_org_java_1.5_ext_2.2_j16_45.csv",
        "lucene_2.4_org_java_1.5_ext_2.2_j16_45.csv",
    ]
    + [
        "velocity_1.4_org_java_1.4.0_ext_2.2_j16_45.csv",
        "velocity_1.5_org_java_1.4.0_ext_2.2_j16_45.csv",
        "velocity_1.6.1_org_java_1.4.0_ext_2.2_j16_45.csv",
    ]
    + [
        "xalan_2.6_org_java_1.3_ext_2.2_j16_45.csv",
        "xalan_2.7_org_java_1.3_ext_2.2_j16_45.csv",
    ]
)

dataset_mapping_test = {
    "ant": [
        "ant_1.3_org_java_1.3_ext_2.2_j16_45.csv",
        "ant_1.4_org_java_1.3_ext_2.2_j16_45.csv",
        "ant_1.5_org_java_1.3_ext_2.2_j16_45.csv",
        "ant_1.6_org_java_1.4.1_ext_2.2_j16_45.csv",
        "ant_1.7_org_java_1.5_ext_2.2_j16_45.csv",
    ]
    + test_sets_for_all_cases,
    "jEdit": [
        "jEdit_3.2.1_org_java_1.3_ext_2.2_j16_45.csv",
        "jEdit_4.0_org_java_1.4.0_ext_2.2_j16_45.csv",
        "jEdit_4.1_org_java_1.4.0_ext_2.2_j16_45.csv",
        "jEdit_4.2_org_java_1.5_ext_2.2_j16_45.csv",
        "jEdit_4.3_org_java_1.5_ext_2.2_j16_45.csv",
    ]
    + test_sets_for_all_cases,
    "pBeans": [
        "pBeans_1.0_org_java_1.4.0_ext_2.2_j16_45.csv",
        "pBeans_2.0_org_java_1.5_ext_2.2_j16_45.csv",
    ]
    + test_sets_for_all_cases,
    "poi": [
        "poi_1.5_org_java_1.3_ext_2.2_j16_45.csv",
        "poi_2.0RC1_org_java_1.4.0_ext_2.2_j16_45.csv",
        "poi_2.5.1_org_java_1.4.0_ext_2.2_j16_45.csv",
        "poi_3.0_org_java_1.4.0_ext_2.2_j16_45.csv",
    ]
    + test_sets_for_all_cases,
    "synapse": [
        "synapse_1.0_org_java_1.5_ext_2.2_j16_45.csv",
        "synapse_1.1_org_java_1.5_ext_2.2_j16_45.csv",
        "synapse_1.2_org_java_1.5_ext_2.2_j16_45.csv",
    ]
    + test_sets_for_all_cases,
    "all": test_sets_for_all_cases,
    "xerces": [
        "xerces_1.2.0_org_java_1.3_ext_2.2_j16_45.csv",
        "xerces_1.3.0_org_java_1.3_ext_2.2_j16_45.csv",
    ]
    + test_sets_for_all_cases,
}  # 29

max_bug_per_file_mapping = {15: "f1f2f3f4f5f6f7f8f9f10f11f12f13f14f15"}
mutation_version = "v_1_0_15"
RATIOS = [0.3, 0.4, 0.5]

dimensionalityreductions = {
    "None": None
    # 'PCA': PCA(random_state=RANDOM_STATE),
    # 'SVD': TruncatedSVD(random_state=RANDOM_STATE)
    #'LDA': LinearDiscriminantAnalysis(),
    #'tSNE': TSNE(random_state=RANDOM_STATE),
    #'MDS': MDS(random_state=RANDOM_STATE)
}

for RATIO in RATIOS:
    for (
        dimensionalityreduction_name,
        dimensionalityreduction,
    ) in dimensionalityreductions.items():
        for train, versions in dataset_mapping_train.items():
            print(train)
            train_versions_path = []
            train_path = os.path.join(
                TEMP_PATH, train + "_" + str(RATIO) + "_not" + ".csv"
            )
            for mbpf_key, mbpf_val in max_bug_per_file_mapping.items():
                for train_, versions_ in dataset_mapping_train.items():
                    if train == train_:
                        continue
                    else:
                        for version in versions_:
                            train_m = version.replace(
                                "org",
                                mutation_version + "_" + mbpf_val + "_" + str(RATIO),
                            )
                            train_versions_path.append(
                                os.path.join(DATA_FOLDER, train_m)
                            )
                create_merged_data_folder(train_versions_path, train_path)

                for test in dataset_mapping_test[train]:
                    try:
                        out = apply_knn_cross_release(
                            train_path,
                            os.path.join(DATA_FOLDER, test),
                            mbpf_val,
                            mutation_version,
                            RATIO,
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
                            RATIO,
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
                            RATIO,
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
                            RATIO,
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
                            RATIO,
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
                            RATIO,
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
                            RATIO,
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
