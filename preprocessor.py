import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

class Preprocessor:
    def __init__(self):
        self.NaN_new_values = None
        self.Parameters_for_column_selection = None
        self.Scaler_parameter = None
    def fit(self, data):
        ## Here is a just fit_transfrom
        target = 'In-hospital_death'
        X = data.drop([target], axis=1)
        y = data[target]
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        imp.fit(X)
        X_train = imp.transform(X)
        self.NaN_new_values = imp.get_params()

        smote = SMOTE()  # deals with class imbalance
        X_train, y_train = smote.fit_resample(X_train, y)

        sel = SelectKBest(f_classif, k=10)
        sel.fit(X_train, y_train)
        self.Parameters_for_column_selection = sel.get_params()
        X_train = sel.transform(X_train)

        scaler = StandardScaler()
        scaler.fit(X_train)
        self.Scaler_parameter = scaler.get_params()
        X_train = scaler.transform(X_train)

        return X_train, y_train
    def transform(self, X):
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        imp.set_params(self.NaN_new_values)
        X_test = imp.transform(X)

        sel = SelectKBest(f_classif, k=10)
        sel.set_params(self.Parameters_for_column_selection)
        X_test = sel.transform(X_test)

        scaler = StandardScaler()
        scaler.set_params(self.Scaler_parameter)
        X_test = scaler.transform(X_test)

        return X_test



