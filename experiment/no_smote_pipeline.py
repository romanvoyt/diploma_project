# standard libs
import argparse
from math import sqrt
import random
import os

# default data science libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# modules for data preprocessing
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score

# classification models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# evaluation metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import SVMSMOTE

parser = argparse.ArgumentParser(description='Experiment parameters')
parser.add_argument('--data', dest='data', type=str, help='Path to data')
parser.add_argument('--threshold', dest='threshold', type=float, help='Threshold coefficient to fix imbalance,'
                                                                      ' if no value is passed, '
                                                                      'then it will not use SMOTE or any other '
                                                                      'algorithms to fix imbalance')
parser.add_argument('--output_dir', dest='output_dir', type=str, help='file to save output')

args = parser.parse_args()


def preprocess_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    def set_new_headers(df):
        columns = ['X' + str(i + 1) for i in range(len(df.columns) - 1)]
        columns.append('Y')
        df.columns = columns

    imputed_df = pd.DataFrame(imputer.fit_transform(dataframe))
    set_new_headers(imputed_df)
    imputed_df['Altman'] = 1.2 * imputed_df['X3'] + 1.4 * imputed_df['X6'] \
                           + 3.3 * imputed_df['X7'] + 0.6 * imputed_df['X8'] + imputed_df['X9']
    print(imputed_df.head())

    Y = imputed_df['Y'].values
    imputed_df.drop('Y', axis=1, inplace=True)
    X = imputed_df.values

    return X, Y


def optimize_hyperparams(model, parameters, x_train, y_train):
    nfolds = 10
    cross_val = StratifiedKFold(nfolds)
    grid = GridSearchCV(model, parameters, cv=cross_val, refit=True, verbose=1, n_jobs=4)
    grid.fit(x_train, y_train)
    print(f'Accuracy : {grid.best_score_} with params {grid.best_estimator_}')
    return grid.best_estimator_


def BuildModel(best_alg, x_train, y_train, x_test, kf, ntrain, ntest, nclass, nfolds):
    Xr_train = np.zeros((ntrain, nclass))
    Xr_test = np.zeros((ntest, nclass))
    tr_ind = np.arange(ntrain)
    for i, (ttrain, ttest) in enumerate(kf.split(tr_ind)):
        clf = best_alg
        clf.fit(x_train[ttrain], y_train[ttrain])
        sc = clf.score(x_train[ttest], y_train[ttest])
        print(f'{i} accuracy {sc:.4f}')
        Xr_train[ttest] = clf.predict_proba(x_train[ttest])
        Xr_test += clf.predict_proba(x_test) / nfolds

    return Xr_train, Xr_test


def train_and_test_model(model, x_train, y_train, x_test, y_test, kf, ntrain, ntest, nclass, nfolds, labels):
    pred_train, pred_test = BuildModel(model, x_train, y_train, x_test, kf, ntrain, ntest, nclass,
                                       nfolds)
    thresholds = np.linspace(0.01, 0.9, 100)
    f1_sc = np.array([f1_score(y_train, pred_train[:, 1] > thr) for thr in thresholds])
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, f1_sc, linewidth=4)
    plt.ylabel("F1 score", fontsize=18)
    plt.xlabel("Threshold", fontsize=18)
    best_lr = thresholds[f1_sc.argmax()]
    print(f1_sc.max())
    print(best_lr)
    show_accuracy(pred_train[:, 1], y_train, labels, best_lr, nclass)
    show_accuracy(pred_test[:, 1], y_test, labels, best_lr, nclass)


def show_accuracy(Xr, y, labels, best, nclass):
    pred = []
    for x in Xr:
        if x > best:
            pred.append(1)
        else:
            pred.append(0)
    print(classification_report(y, pred, target_names=labels, digits=4))
    print(confusion_matrix(y, pred, labels=range(nclass)))


def main():
    data = pd.read_excel(args.data)
    X, Y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1337)
    scaler = StandardScaler().fit(X_train)
    scaler_test = StandardScaler().fit(X_test)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler_test.transform(X_test)

    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]
    nclass = 2
    SEED = 1337
    NFOLDS = 10
    kf = KFold(n_splits=NFOLDS, random_state=SEED, shuffle=True)
    labels = ['Normal', 'Bankruptcy']

    # logistic regression
    print(f'{"*" * 20} - Logistic regression- {"*" * 20}')
    log_reg = LogisticRegression(class_weight='balanced', max_iter=15000)
    log_reg_params = {'C': [0.45, 0.5, 0.55],
                      'solver': ['newton-cg']}
    best_log = optimize_hyperparams(log_reg, log_reg_params, X_train_scaled, y_train)
    train_and_test_model(best_log, X_train_scaled, y_train, X_test_scaled, y_test, kf, ntrain, ntest, nclass, NFOLDS, labels)

    print(f'{"*" * 40}')

    # decision tree

    print(f'{"*" * 20} - Decision tree classifier - {"*" * 20}')

    tree_clf = DecisionTreeClassifier(criterion='gini')
    tree_clf_params = {'max_depth': [50, 75, 100],
                       'max_leaf_nodes': [50, 100, 150]}

    best_tree = optimize_hyperparams(tree_clf, tree_clf_params, X_train, y_train)
    train_and_test_model(best_tree, X_train, y_train, X_test, y_test, kf, ntrain, ntest, nclass, NFOLDS, labels)

    print(f'{"*" * 40}')

    # Random Forest classifier

    print(f'{"*" * 20} - Random Forest Classifier - {"*" * 20}')

    rf_clf = RandomForestClassifier(criterion='gini')
    rf_params = {
        'n_estimators': [50, 100, 150],
        # 'max_depth': [25, 50, 100],
        'max_features': [16, 32, 64],
        'max_leaf_nodes': [5,10,15],
        'bootstrap': [True]
        }
    best_rf = optimize_hyperparams(rf_clf, rf_params, X_train, y_train)
    train_and_test_model(best_rf, X_train, y_train, X_test, y_test, kf, ntrain, ntest, nclass, NFOLDS, labels)

    print(f'{"*" * 40}')

    # XGB classifier

    print(f'{"*" * 20} XGB Classifier - {"*" * 20}')

    xgb_clf = xgb.XGBClassifier()
    xgb_params = {
        'n_estimators': [150],
        'booster': ['gbtree', 'gblinear'],
        'learning_rate': [0.1, 0.2, 0.3]
    }
    best_xgb = optimize_hyperparams(xgb_clf, xgb_params, X_train, y_train)
    train_and_test_model(best_xgb, X_train, y_train, X_test, y_test, kf, ntrain, ntest, nclass, NFOLDS, labels)

    print(f'{"*" * 40}')


if __name__ == '__main__':
    main()