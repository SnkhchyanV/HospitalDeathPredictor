import argparse
import preprocessor
import model
import pickle
import json
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--inference", type=str, default=None)
args = parser.parse_args()

class Pipeline:
    def __init__(self, preprocessor_obj, model_obj):
        self.preprocessor = preprocessor_obj
        self.model = model_obj

    def Run(self, data_path, inference):
        data = pd.read_csv(data_path)

        # Choosing the mode
        if inference is None:
            result = self._Run_Train(data)  # training on the data, returns X_train, y_train, y_pred
        else:
            result = self._Run_Test(data) # testing the data, returns X_test, y_pred

        # returning the values of functions that are train or test datas, and predictions, in case of after use
        return result

    def _Run_Test(self, data):
        # loading the saved preprocessor
        with open('preprocessor.pkl', 'rb') as preprocessor_file:
            self.preprocessor = pickle.load(preprocessor_file)

        # preprocessing the test data
        X_test = self.preprocessor.transform(data)  # Preprocessing the data, before test

        # Loading the trained model
        with open('model.pkl', 'rb') as model_file:
            self.model = pickle.load(model_file)

        # Setting the threshold and gain the probas
        # threshold is set low because we want to have more false negative (just in case to doctors be more attentive)
        threshold = 0.3
        y_pred_proba = self.model.predict_proba(X_test)

        y_pred = np.zeros(len(y_pred_proba), dtype=int)
        # Make predictions with set threshold
        for i in range(len(y_pred)):
            if y_pred_proba[i][0] >= threshold:
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        # Saving outputs in predictions.json file
        output = {'predict_probes': y_pred_proba.tolist(), 'threshold': threshold}
        with open('predictions.json', 'w') as predictions_file:
            json.dump(output, predictions_file)

        # Returning test data, and predicted values in case of the future use
        # None is here to have the same result such as in the train function
        return X_test, None, y_pred

    def _Run_Train(self, data):
        # Preprocessing
        X_train, y_train = self.preprocessor.fit(data)

        # Fitting the model
        self.model.fit(X_train, y_train)

        # Saving the trained model and preprocessor
        with open('model.pkl', 'wb') as model_file:
            pickle.dump(self.model, model_file)
        with open('preprocessor.pkl', 'wb') as preprocessor_file:
            pickle.dump(self.preprocessor, preprocessor_file)

        # Predictions, just in case to use for threshold decision
        y_pred = self.model.predict(X_train)

        # Returning test data, and predicted values in case of the future use
        return X_train, y_train, y_pred

preprocessor_obj = preprocessor.Preprocessor()
model_obj = model.Model()

pipeline = Pipeline(preprocessor_obj, model_obj)
X, y_real, y_pred = pipeline.Run(args.data_path, args.inference) #y_real could be None if it was testing


### to_delete, just testing
'''
data = pd.read_csv("C:/Users/vsnkh/Downloads/hospital_deaths_train.csv")

target = 'In-hospital_death'
X = data.drop([target], axis=1)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)
X_train[target]=y_train

X_train, y_train, y_pred_train = pipeline._Run_Train(X_train) #
X_test, y_not_real_just_Nones, y_pred_test = pipeline._Run_Test(X_test) #y_test would be None because it was testing

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(accuracy_train)
print(accuracy_test)
'''