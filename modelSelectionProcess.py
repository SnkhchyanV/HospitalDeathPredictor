# -*- coding: utf-8 -*-
import pandas as pd
import warnings

import sklearn.metrics

warnings.filterwarnings('ignore')

# !pip install imblearn

df = pd.read_csv('./hospital_deaths_train.csv')

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table # EDIT: see deprecation warnings below

ax = plt.subplot(1,1,1, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis



##Preprocessiong Changing NaN_to_median
##Median
# target = 'In-hospital_death'
# X = df.drop([target,'recordid'],axis=1)
# y = df[target]

target = 'In-hospital_death'
X = df.drop([target], axis=1)
y = df[target]


def fit(data):
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(X)
    X_train = imp.transform(X)

    smote = SMOTE()  # deals with class imbalance
    X_train, y_train = smote.fit_resample(X_train, y)

    sel = SelectKBest(f_classif, k=10)
    sel.fit(X_train, y_train)
    Parameters_for_column_selection = sel.get_params()
    X_train = sel.transform(X_train)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    return X_train, y_train



# q_sam, q_fet = X.shape
# for i in range(q_fet):
#   median = X.iloc[:,i].median(skipna=True)
#   X.iloc[:,i] = X.iloc[:,i].fillna(median)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#
# ##DecisionTree for median
X, y = fit(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#
DTC = DecisionTreeClassifier()
DTC.fit(X_train,y_train)

print("\nDecision Tree Median scores\n")
print(f"Score train: {DTC.score(X_train,y_train)}")
print(f"Score test: {DTC.score(X_test,y_test)}")

tn, fp, fn, tp = confusion_matrix(DTC.predict(X_test), y_test).ravel()

DTCV = [
    sklearn.metrics.accuracy_score(DTC.predict(X_test), y_test),
    tn / (tn+fp),
    tp / (tp+fn),
    sklearn.metrics.roc_auc_score(DTC.predict(X_test), y_test),
    sklearn.metrics.matthews_corrcoef(DTC.predict(X_test), y_test)
]


# # parameters = {
# #     "min_samples_split": [2,3,4,5,6,7,8,9,10],
# # }
# # clf = GridSearchCV(DecisionTreeClassifier(), parameters)
# # clf.fit(X_train, y_train)
# #
# # print("\nGreed Decision Tree Median scores\n")
# # print(f"Best Score for test: {clf.best_score_}\n with for f{clf.best_params_}")
#
# ## RandomForest for median
RFC = RandomForestClassifier(bootstrap=True)
RFC.fit(X_train,y_train)

print("\n\nRandomForest for median scores: \n")
print(f"Score train: {RFC.score(X_train,y_train)}")
print(f"Score test: {RFC.score(X_test,y_test)}")

tn, fp, fn, tp = confusion_matrix(RFC.predict(X_test), y_test).ravel()

RFCV = [
        sklearn.metrics.accuracy_score(RFC.predict(X_test), y_test),
        tn / (tn + fp),
        tp / (tp + fn),
        sklearn.metrics.roc_auc_score(RFC.predict(X_test), y_test),
        sklearn.metrics.matthews_corrcoef(RFC.predict(X_test), y_test)
    ]
#
# # parameters = {
# #     "min_samples_split": [2,3,4,5],
# #     "bootstrap": [True, False],
# #     # "oob_score": [True, False],
# #     # "class_weight": ["balanced", "balanced_subsample"]
# # }
# # clf = GridSearchCV(RandomForestClassifier(), parameters)
# # clf.fit(X_train, y_train)
# #
# # print("\nRandomForest Median scores\n")
# # print(f"Best Score for test: {clf.best_score_}\n with for f{clf.best_params_}")
#
# ## KNN for median
KNNC = KNeighborsClassifier(weights='uniform',algorithm='auto', n_neighbors=11, leaf_size=10, p=1)
KNNC.fit(X_train,y_train)

print("\n\nKNeighborsClassifier for median scores: \n")
print(f"Score train: {KNNC.score(X_train,y_train)}")
print(f"Score test: {KNNC.score(X_test,y_test)}")

tn, fp, fn, tp = confusion_matrix(KNNC.predict(X_test), y_test).ravel()


KNNCV = [
    sklearn.metrics.accuracy_score(KNNC.predict(X_test), y_test),
     tn / (tn + fp),
     tp / (tp + fn),
     sklearn.metrics.roc_auc_score(KNNC.predict(X_test), y_test),
     sklearn.metrics.matthews_corrcoef(KNNC.predict(X_test), y_test)
 ]

#
# # parameters = {
# #     'n_neighbors': [10,11,12,13],
# #     'leaf_size': [10,15,20,25,30,35,40],
# #     'p': [1,2,3],
# #     # 'weights': ['uniform', 'distance'],
# #     # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
# # }
# # clf = GridSearchCV(KNeighborsClassifier(), parameters)
# # clf.fit(X_train, y_train)
# #
# # print("\nKNN Median scores\n")
# # print(f"Best Score for test: {clf.best_score_}\n with for {clf.best_params_}")
#
# ## RBF svm
SVCC = SVC()
SVCC.fit(X_train,y_train)
print("\n\nSVM RBF scores scores: \n")
print(f"Score train: {SVCC.score(X_train,y_train)}")
print(f"Score test: {SVCC.score(X_test,y_test)}")

tn, fp, fn, tp = confusion_matrix(SVCC.predict(X_test), y_test).ravel()


SVCCV = [sklearn.metrics.accuracy_score(SVCC.predict(X_test), y_test),
         tn / (tn + fp),
         tp / (tp + fn),
         sklearn.metrics.roc_auc_score(SVCC.predict(X_test), y_test),
         sklearn.metrics.matthews_corrcoef(SVCC.predict(X_test), y_test)
     ]

#
# parameters = {
#     'C': [0.1,0.2,0.3,0.7,0.9,1.0],
#     # 'gamma': ['scale', 'auto']
# }
# clf = GridSearchCV(SVC(), parameters)
# clf.fit(X_train, y_train)
#
# print("\nSVC Median scores\n")
# print(f"Best Score for test: {clf.best_score_}\n with for {clf.best_params_}")

## AdaBoost RandomForest


ABSVM1 = AdaBoostClassifier(RandomForestClassifier())
ABSVM1.fit(X_train,y_train)
print("\n\nAdaBoost RandomForest scores: \n")
print(f"Score train: {ABSVM1.score(X_train,y_train)}")
print(f"Score test: {ABSVM1.score(X_test,y_test)}")
# s = GridSearchCV(AdaBoostClassifier(RandomForestClassifier()), {
#     'n_estimators': [110,15],
#     'learning_rate': [0.55,0.75]
# })
# s.fit(X_train,y_train)

# print(s.best_params_)
# print(s.best_score_)

tn, fp, fn, tp = confusion_matrix(ABSVM1.predict(X_test), y_test).ravel()

ABSVM1V = [sklearn.metrics.accuracy_score(ABSVM1.predict(X_test), y_test),
         tn / (tn + fp),
         tp / (tp + fn),
         sklearn.metrics.roc_auc_score(ABSVM1.predict(X_test), y_test),
         sklearn.metrics.matthews_corrcoef(ABSVM1.predict(X_test), y_test)
     ]


table(ax, pd.DataFrame({
    'Classifier':
        ['Random forest', 'Naive Bayas', 'KNN', 'SVM', 'Adaboost Random Forest'],
    'Accuracy':
        [DTCV[0], RFCV[0], KNNCV[0], SVCCV[0], ABSVM1V[0]],
    'Sensitivity':
        [DTCV[1], RFCV[1], KNNCV[1], SVCCV[1], ABSVM1V[1]],
    'Specificity':
        [DTCV[2], RFCV[2], KNNCV[2], SVCCV[2], ABSVM1V[2]],
    'AUC':
        [DTCV[3], RFCV[3], KNNCV[3], SVCCV[3], ABSVM1V[3]],
    'MCC':
        [DTCV[4], RFCV[4], KNNCV[4], SVCCV[4], ABSVM1V[4]]
    }))

plt.tight_layout()
plt.savefig('mytable.png', bbox_inches = 'tight', dpi=1000)