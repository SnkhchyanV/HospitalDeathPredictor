import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

class Preprocessor:
    def __init__(self):
        self.imp = None
        self.sel = None
        self.scaler = None
    def fit(self, data):
        # By the way Here is a just fit-transformer
        target = 'In-hospital_death'
        X = data.drop([target], axis=1)
        y = data[target]

        # Replacing all NaN with medians of column, and then saving the imputer to call transform in the future
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        imp.fit(X)
        X_train = imp.transform(X)
        self.imp = imp

        # Fixing class imbalance
        smote = SMOTE()  # deals with class imbalance
        X_train, y_train = smote.fit_resample(X_train, y)

        # Leave the most (top 30) important columns, saving model of selector
        sel = SelectKBest(f_classif, k=30)
        sel.fit(X_train, y_train)
        X_train = sel.transform(X_train)
        self.sel = sel

        # Making a scale, and save the scaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        self.scaler = scaler

        # Return X_train and y_train in case the reusability
        return X_train, y_train
    def transform(self, X):

        # Preprocessing the test data, with the saved models
        X_test = self.imp.transform(X)
        X_test = self.sel.transform(X_test)
        X_test = self.scaler.transform(X_test)
        return X_test



