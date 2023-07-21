from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from dataloader import DataLoader
import os
from performance_measure import calculate_performance_metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from performance_measure import loss_function_auc
from imblearn.pipeline import Pipeline
from sklearn.metrics import make_scorer

ML_METHOD = "knn"
RANDOM_STATE = 42
PARAMS = {
    # 'dimensionalityreduction__n_components':(8,10,12,14,16),
    "classification__n_neighbors": (1, 3, 5, 7, 9, 11, 13, 15)
}


def apply_knn_cross_release(
    dataset_path,
    test_dataset,
    sampling_type,
    sampling_name,
    ratio,
    sampling,
    dimensionalityreduction,
    dimensionalityreduction_name,
    is_sampling=False,
):
    dataloader = DataLoader(dataset_path)
    test_dataloader = DataLoader(test_dataset)
    _, f_test = os.path.split(test_dataset)
    f_test, _ = os.path.splitext(f_test)
    _, f = os.path.split(dataset_path)
    f, _ = os.path.splitext(f)
    f = f + "_" + f_test
    f = f + "_" + ML_METHOD + "_cross"
    f_tokens = f.split("_")
    project = f_tokens[0]
    train_version = f_tokens[1]
    test_version = f_test.split("_")[1]

    X_train = dataloader.X
    y_train = dataloader.y
    X_test = test_dataloader.X
    y_test = test_dataloader.y
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)

    if is_sampling:
        pipeline = Pipeline(
            [
                ("sampling", sampling),
                # ('dimensionalityreduction', dimensionalityreduction),
                ("classification", KNeighborsClassifier()),
            ]
        )
    else:
        pipeline = Pipeline(
            [
                # ('dimensionalityreduction', dimensionalityreduction),
                ("classification", KNeighborsClassifier())
            ]
        )

    custom_scorer = make_scorer(loss_function_auc, greater_is_better=True)
    kf = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)

    folds_train = {}
    folds_val = {}
    output = []
    best_idx = -1
    best_score = 0
    best_params = {}
    for idx, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        folds_train[idx] = (X_train[train_index], y_train[train_index])
        folds_val[idx] = (X_train[val_index], y_train[val_index])
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=PARAMS,
            cv=[(train_index, val_index)],
            verbose=-1,
            scoring=custom_scorer,
            refit=False,
            n_jobs=4,
        )
        grid.fit(X_train, y_train)

        pipeline.set_params(**grid.best_params_)
        pipeline.fit(folds_train[idx][0], folds_train[idx][1])

        y_pred_val = pipeline.predict(folds_val[idx][0])
        y_gt_val = folds_val[idx][1]
        y_pred_test = pipeline.predict(X_test)

        output.append(
            calculate_performance_metrics(
                f,
                pipeline,
                X_train,
                y_train,
                project,
                train_version,
                test_version,
                ML_METHOD,
                sampling_type,
                sampling_name,
                dimensionalityreduction_name,
                ratio,
                idx,
                0,
                str(grid.best_params_),
                y_gt_val,
                y_pred_val,
            )
        )
        if output[-1].auc > best_score:
            best_idx = idx
            best_params = grid.best_params_
            best_score = output[-1].auc

    pipeline.set_params(**best_params)
    pipeline.fit(folds_train[best_idx][0], folds_train[best_idx][1])
    y_pred_test = pipeline.predict(X_test)
    output.append(
        calculate_performance_metrics(
            f,
            pipeline,
            X_train,
            y_train,
            project,
            train_version,
            test_version,
            ML_METHOD,
            sampling_type,
            sampling_name,
            dimensionalityreduction_name,
            ratio,
            best_idx,
            1,
            str(best_params),
            y_test,
            y_pred_test,
        )
    )

    return output


# def apply_knn_mutated(dataset_path, org_dataset_path, ratio):
#     dataloader = DataLoader(dataset_path)
#     _, f = os.path.split(dataset_path)
#     f, _ = os.path.splitext(f)
#     f_tokens = f.split('_')
#     project = f_tokens[0]
#     train_version = f_tokens[1]
#     test_version = f_tokens[1]
#     ### Split train val and test with org dataset get indexes
#     # use mutated elements for train org elements for val and test
#     f = f + '_' + ML_METHOD + '_within'

#     X = dataloader.X
#     y = dataloader.y
#     min_max_scaler = MinMaxScaler()
#     X = min_max_scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

#     pipeline = Pipeline([
#         ('classification', KNeighborsClassifier())
#     ])

#     custom_scorer = make_scorer(loss_function_auc, greater_is_better=True)
#     kf = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)

#     folds_train = {}
#     folds_val = {}
#     output = []
#     best_idx = -1
#     best_score = 0
#     best_params = {}
#     for idx, (train_index, val_index) in enumerate(kf.split(X_train,y_train)):
#         folds_train[idx] = (X_train[train_index], y_train[train_index])
#         folds_val[idx] = (X_train[val_index], y_train[val_index])
#         grid = GridSearchCV(estimator=pipeline, param_grid=PARAMS, cv=[(train_index, val_index)], verbose=-1, scoring=custom_scorer, refit=False, n_jobs=4)
#         grid.fit(X_train, y_train)

#         pipeline.set_params(**grid.best_params_)
#         pipeline.fit(folds_train[idx][0], folds_train[idx][1])

#         y_pred_val = pipeline.predict(folds_val[idx][0])
#         y_gt_val = folds_val[idx][1]
#         y_pred_test = pipeline.predict(X_test)

#         output.append(calculate_performance_metrics(f, project, train_version, test_version, ML_METHOD, sampling_type, sampling_name, ratio, idx, 0, str(grid.best_params_), y_gt_val, y_pred_val))
#         if output[-1].bal > best_score:
#             best_idx = idx
#             best_params = grid.best_params_
#             best_score = output[-1].bal

#     pipeline.set_params(**best_params)
#     pipeline.fit(folds_train[best_idx][0], folds_train[best_idx][1])
#     y_pred_test = pipeline.predict(X_test)
#     output.append(calculate_performance_metrics(f, project, train_version, test_version, ML_METHOD, sampling_type, sampling_name, ratio, best_idx, 1, str(best_params), y_test, y_pred_test))

#     return output
