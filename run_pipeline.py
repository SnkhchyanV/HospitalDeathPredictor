import argparse
import preprocessor
import model
import pickle
import json
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--inference",type=str, default=None)
args = parser.parse_args()

class Pipeline:
    def __init__(self, preprocessor_obj, model_obj):
        self.preprocessor = preprocessor_obj
        self.model = model_obj

    def Run(self, data_path, inference):
        data = pd.read_csv(data_path)
        # Choosing the mode
        if inference is None:
            self._Run_Train(data)
        else:
            self._Run_Test(data)

    def _Run_Test(self, data):
        X_test = self.preprocessor.transform(data)  # Preprocessing the data, before test

        # Loading the trained model
        with open('model.pkl', 'rb') as model_file:
            self.model = pickle.load(model_file)

        # Making the prediction
        threshold = 0.5
        y_pred_proba = self.model.predict_proba(X_test)
        y_pred = np.zeros(len(y_pred_proba), dtype=int)
        # use a loop to iterate over each probability value
        for i in range(len(y_pred)):
            if y_pred_proba[i] >= threshold:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        # Saving outputs in predictions.json file
        output = {'predict_probes': list(y_pred_proba), 'threshold': threshold}
        with open('predictions.json', 'w') as predictions_file:
            json.dump(output, predictions_file)

        return y_pred

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

        # Predictions
        y_pred = self.model.predict(X_train)
        # Accuracy
        accuracy = np.mean(y_train == y_pred)
        print('Accuracy: ', accuracy)

        return y_pred

preprocessor_obj = preprocessor.Preprocessor()
model_obj = model.Model()

pipeline = Pipeline(preprocessor_obj, model_obj)
pipeline.Run(args.data_path, args.inference)
