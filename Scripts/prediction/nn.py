import torch
import numpy as np
from torch import nn
from skorch import NeuralNetClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from dataloader import DataLoader
import os
from performance_measure import calculate_performance_metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from performance_measure import loss_function_mcc
from imblearn.pipeline import Pipeline
from sklearn.metrics import make_scorer


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim=21, dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, X):
        return self.model(X)

class BinaryClassifierSampling(BinaryClassifier):
    def __init__(self, input_dim=21, dropout=0.3):
        super().__init__(input_dim, dropout)
    
    def forward(self, X):
        out = self.model(X)
        return out.squeeze(1) 


ML_METHOD = "neuralnet"
RANDOM_STATE = 42
PARAMS = {
    'classification__module__dropout': [0.1, 0.3, 0.5],
    'classification__lr': [0.001, 0.01],
    'classification__max_epochs': [20, 30],
}


def apply_neuralnet_cross_release(
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
    y_train = y_train.reshape(-1, 1)
    X_test = test_dataloader.X
    y_test = test_dataloader.y

    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')   
    y_train = y_train.astype('float32')
    # y_test = y_test.astype('float32')
    if is_sampling:
        pipeline = Pipeline(
            [
                ("sampling", sampling),
                # ('dimensionalityreduction', dimensionalityreduction),
                ("classification", NeuralNetClassifier(
                                    module=BinaryClassifierSampling,
                                    module__input_dim=20,
                                    module__dropout=0.3,
                                    max_epochs=20,
                                    lr=0.01,
                                    optimizer=torch.optim.Adam,
                                    criterion=nn.BCELoss,
                                    batch_size=32,
                                    iterator_train__shuffle=True,
                                    verbose=0,
                                    device='cpu'
                                )),
            ]
        )
    else:
        pipeline = Pipeline(
            [
                # ('dimensionalityreduction', dimensionalityreduction),
                ("classification", NeuralNetClassifier(
                                    module=BinaryClassifier,
                                    module__input_dim=20,
                                    module__dropout=0.3,
                                    max_epochs=20,
                                    lr=0.01,
                                    optimizer=torch.optim.Adam,
                                    criterion=nn.BCELoss,
                                    batch_size=32,
                                    iterator_train__shuffle=True,
                                    verbose=0,
                                    device='cpu'
                                ))
            ]
        )

    custom_scorer = make_scorer(loss_function_mcc, greater_is_better=True)
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
            verbose=0,
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
        y_pred_val = np.ravel(y_pred_val)
        y_gt_val = np.ravel(y_gt_val)
        y_pred_test = np.ravel(y_pred_test)

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
    y_pred_test = np.ravel(y_pred_test)

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
