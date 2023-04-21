# HospitalDeathPredictor
It's trained application that predict the probability of death in the hospital using some of information.

Job division:

Vlad - pipeline.py, model.py, model selection process member, preprocessing algorithm selection member, testing
Sona - preprocessing.py, preprocessing algorithm selection member, new test data creation, testing 
Karapet - model selection process member, model best parameter searcher (gread searcher), testing

Our GutHub repository:

https://github.com/SnkhchyanV/HospitalDeathPredictor

Neccessary liberies and modules in pipeline.py, model.py, prepocessing.py
import argparse
import preprocessor
import model
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

Other liberies and modules that we use in model selection:
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
