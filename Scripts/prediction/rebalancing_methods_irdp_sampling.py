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
from dataloader import DataLoader
from collections import Counter

DATA_FOLDER = "../../Software Measures Datasets"
RESULT_CSV_PATH = "../../Performance Results/Rebalancing Methods/IRDP/performance_irdp_sampling_71.csv"
RANDOM_STATE = 71 # 42, 51, 92, 14, 71

save_file = PerformanceMeasureCSV(RESULT_CSV_PATH)
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
        print(train)
        raw_data = DataLoader(os.path.join(DATA_FOLDER, train))
        counter = Counter(raw_data.y)
        total_file_count = counter[0] + counter[1]
        bug_ratio = counter[1] / total_file_count

        for test in tests:
            ratios = []
            t_ratios = []
            if bug_ratio < 0.5:
                is_over_sample = True
            else:
                is_over_sample = False

            if is_over_sample:
                # if bug_ratio < 0.3:
                #     t_ratios.append(0.3)
                #     ratios.append(3/7)
                # if bug_ratio < 0.4:
                #     t_ratios.append(0.4)
                #     ratios.append(4/6)
                if bug_ratio < 0.5:
                    t_ratios.append(0.5)
                    ratios.append(5 / 5)
            else:
                if bug_ratio > 0.9:
                    t_ratios.append(0.9)
                    ratios.append(1 / 9)
                if bug_ratio > 0.8:
                    t_ratios.append(0.8)
                    ratios.append(2 / 8)
                if bug_ratio > 0.7:
                    t_ratios.append(0.7)
                    ratios.append(3 / 7)
                if bug_ratio > 0.6:
                    t_ratios.append(0.6)
                    ratios.append(4 / 6)
                if bug_ratio > 0.5:
                    t_ratios.append(0.5)
                    ratios.append(5 / 5)

            for idx, ratio in enumerate(ratios):
                t_ratio = t_ratios[idx]
                if is_over_sample:
                    # OVER SAMPLE
                    sampling_type = "over_sample"
                    samplings = {
                        "smote": SMOTE(
                            sampling_strategy=ratio, random_state=RANDOM_STATE
                        ),
                        "random_over_sampler": RandomOverSampler(
                            sampling_strategy=ratio, random_state=RANDOM_STATE
                        ),
                        "smote_n": SMOTEN(
                            sampling_strategy=ratio, random_state=RANDOM_STATE
                        ),
                        "borderline_smote": BorderlineSMOTE(
                            sampling_strategy=ratio, random_state=RANDOM_STATE
                        ),
                        "borderline_smote_svm": SVMSMOTE(
                            sampling_strategy=ratio, random_state=RANDOM_STATE
                        ),
                    }

                else:
                    continue

                for name, sampling in samplings.items():
                    try:
                        out = apply_knn_cross_release(
                            os.path.join(DATA_FOLDER, train),
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
                            os.path.join(DATA_FOLDER, train),
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
                            os.path.join(DATA_FOLDER, train),
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
                            os.path.join(DATA_FOLDER, train),
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
                            os.path.join(DATA_FOLDER, train),
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
                            os.path.join(DATA_FOLDER, train),
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
                            os.path.join(DATA_FOLDER, train),
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
