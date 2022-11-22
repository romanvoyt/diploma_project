#standard libs
import argparse
from math import sqrt
import random
import os

#default data science libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#modules for data preprocessing
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, KFold
from sklearn.model_selection import RandomizedSearchCV

#classification models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

#evaluation metrics
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


def main():
    data = pd.read_excel(args.data)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    def set_new_headers(df):
        columns = ['X'+str(i+1) for i in range(len(df.columns)-1)]
        columns.append('Y')
        df.columns = columns

    imputed_df = pd.DataFrame(imputer.fit_transform(data))
    set_new_headers(imputed_df)
    imputed_df['Altman'] = 1.2*imputed_df['X3']+1.4*imputed_df['X6']\
                           + 3.3*imputed_df['X7']+0.6*imputed_df['X8']+imputed_df['X9']
    print(imputed_df.tail(100))


if __name__ == '__main__':
    main()

