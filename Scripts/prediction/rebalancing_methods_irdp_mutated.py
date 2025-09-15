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


DATA_FOLDER = "../../Software Measures Datasets"
RESULT_CSV_PATH = "../../Performance Results/Rebalancing Methods/IRDP/performance_irdp_mutated.csv"
RANDOM_STATE = 42

save_file = PerformanceMeasureCSV(RESULT_CSV_PATH, is_mutated=True)
dataset_mapping = {
    "ant_1.3_org_java_1.3_ext_2.2_j16_45.csv": [
        "ant_1.4_org_java_1.3_ext_2.2_j16_45.csv",
        "ant_1.5_org_java_1.3_ext_2.2_j16_45.csv",
        "ant_1.6_org_java_1.4.1_ext_2.2_j16_45.csv",
        "ant_1.7_org_java_1.5_ext_2.2_j16_45.csv",
    ],
    "ant_1.4_org_java_1.3_ext_2.2_j16_45.csv": [
        "ant_1.5_org_java_1.3_ext_2.2_j16_45.csv",
        "ant_1.6_org_java_1.4.1_ext_2.2_j16_45.csv",
        "ant_1.7_org_java_1.5_ext_2.2_j16_45.csv",
    ],
    "ant_1.5_org_java_1.3_ext_2.2_j16_45.csv": [
        "ant_1.6_org_java_1.4.1_ext_2.2_j16_45.csv",
        "ant_1.7_org_java_1.5_ext_2.2_j16_45.csv",
    ],
    "ant_1.6_org_java_1.4.1_ext_2.2_j16_45.csv": [
        "ant_1.7_org_java_1.5_ext_2.2_j16_45.csv"
    ],

    "jEdit_3.2.1_org_java_1.3_ext_2.2_j16_45.csv": [
        "jEdit_4.0_org_java_1.4.0_ext_2.2_j16_45.csv",
        "jEdit_4.1_org_java_1.4.0_ext_2.2_j16_45.csv",
        "jEdit_4.2_org_java_1.5_ext_2.2_j16_45.csv",
        "jEdit_4.3_org_java_1.5_ext_2.2_j16_45.csv",
    ],
    "jEdit_4.0_org_java_1.4.0_ext_2.2_j16_45.csv": [
        "jEdit_4.1_org_java_1.4.0_ext_2.2_j16_45.csv",
        "jEdit_4.2_org_java_1.5_ext_2.2_j16_45.csv",
        "jEdit_4.3_org_java_1.5_ext_2.2_j16_45.csv",
    ],
    "jEdit_4.1_org_java_1.4.0_ext_2.2_j16_45.csv": [
        "jEdit_4.2_org_java_1.5_ext_2.2_j16_45.csv",
        "jEdit_4.3_org_java_1.5_ext_2.2_j16_45.csv",
    ],
    "jEdit_4.2_org_java_1.5_ext_2.2_j16_45.csv": [
        "jEdit_4.3_org_java_1.5_ext_2.2_j16_45.csv"
    ],

    "lucene_2.0_org_java_1.4.2_ext_2.2_j16_45.csv": [
        "lucene_2.2_org_java_1.5_ext_2.2_j16_45.csv",
        "lucene_2.4_org_java_1.5_ext_2.2_j16_45.csv",
    ],

    "poi_2.0RC1_org_java_1.4.0_ext_2.2_j16_45.csv": [
        "poi_2.5.1_org_java_1.4.0_ext_2.2_j16_45.csv",
        "poi_3.0_org_java_1.4.0_ext_2.2_j16_45.csv",
    ],

    "synapse_1.1_org_java_1.5_ext_2.2_j16_45.csv": [
        "synapse_1.2_org_java_1.5_ext_2.2_j16_45.csv"
    ],

    "xalan_2.6_org_java_1.3_ext_2.2_j16_45.csv": [
        "xalan_2.7_org_java_1.3_ext_2.2_j16_45.csv"
    ],

    "xerces_1.2.0_org_java_1.3_ext_2.2_j16_45.csv": [
        "xerces_1.3.0_org_java_1.3_ext_2.2_j16_45.csv"
    ],
}
ratio_mapping = {
    "ant_1.3_org_java_1.3_ext_2.2_j16_45.csv": [0.5],
    "ant_1.4_org_java_1.3_ext_2.2_j16_45.csv": [0.5],
    "ant_1.5_org_java_1.3_ext_2.2_j16_45.csv": [0.5],
    "ant_1.6_org_java_1.4.1_ext_2.2_j16_45.csv": [0.5],
    "jEdit_3.2.1_org_java_1.3_ext_2.2_j16_45.csv": [0.5],
    "jEdit_4.0_org_java_1.4.0_ext_2.2_j16_45.csv": [0.5],
    "jEdit_4.1_org_java_1.4.0_ext_2.2_j16_45.csv": [0.5],
    "jEdit_4.2_org_java_1.5_ext_2.2_j16_45.csv": [0.5],
    "lucene_2.0_org_java_1.4.2_ext_2.2_j16_45.csv": [0.5],
    "pBeans_1.0_org_java_1.4.0_ext_2.2_j16_45.csv": [0.5],
    "poi_2.0RC1_org_java_1.4.0_ext_2.2_j16_45.csv": [0.5],
    "synapse_1.1_org_java_1.5_ext_2.2_j16_45.csv": [0.5],
    "xalan_2.6_org_java_1.3_ext_2.2_j16_45.csv": [0.5],
    "xerces_1.2.0_org_java_1.3_ext_2.2_j16_45.csv": [0.5],
}
max_bug_per_file_mapping = {
    15: "f1f2f3f4f5f6f7f8f9f10f11f12f13f14f15"
}
mutation_version = "v_1_0_15"

dimensionalityreductions = {
    "None": None
    #'PCA': PCA(random_state=RANDOM_STATE),
    #'SVD': TruncatedSVD(random_state=RANDOM_STATE),
    #'LDA': LinearDiscriminantAnalysis(),
    #'tSNE': TSNE(random_state=RANDOM_STATE),
    #'MDS': MDS(random_state=RANDOM_STATE)
}

for (
    dimensionalityreduction_name,
    dimensionalityreduction,
) in dimensionalityreductions.items():
    for train, tests in dataset_mapping.items():
        for mbpf_key, mbpf_val in max_bug_per_file_mapping.items():
            for ratio in ratio_mapping[train]:
                for test in tests:
                    train_m = train.replace(
                        "org", mutation_version + "_" + mbpf_val + "_" + str(ratio)
                    )
                    print(train_m)
                    try:
                        out = apply_knn_cross_release(
                            os.path.join(DATA_FOLDER, train_m),
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
                            os.path.join(DATA_FOLDER, train_m),
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
                            os.path.join(DATA_FOLDER, train_m),
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
                            os.path.join(DATA_FOLDER, train_m),
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
                            os.path.join(DATA_FOLDER, train_m),
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
                            os.path.join(DATA_FOLDER, train_m),
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
                            os.path.join(DATA_FOLDER, train_m),
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
