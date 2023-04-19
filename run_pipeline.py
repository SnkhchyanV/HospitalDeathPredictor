import argparse
import preprocessor
import model
import pickle
import json
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--inference", default=None)
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
        # Needs modifying
        threshold = 0.5
        y_pred_proba = self.model.predict_proba(X_test, threshold)
        y_pred = None
        #

        # Saving outputs in predictions.json file
        output = {'predict_probes': list(y_pred_proba), 'threshold': threshold}
        with open('predictions.json', 'w') as predictions_file:
            json.dump(output, predictions_file)

        return y_pred

    def _Run_Train(self, data):
        # Splitting data and target
        target = 'In-hospital_death'
        X_train = data.drop([target], axis=1)
        y_train = data[target]

        # Preprocessing
        self.preprocessor.fit(X_train)  # or
        X_train_transformed = self.preprocessor.transform(X_train)

        # Fitting the model
        self.model.fit(X_train_transformed, y_train)

        # Saving the trained model and preprocessor
        with open('model.pkl', 'wb') as model_file:
            pickle.dump(self.model, model_file)
        with open('preprocessor.pkl', 'wb') as preprocessor_file:
            pickle.dump(self.preprocessor, preprocessor_file)

        # Predictions
        y_pred = self.model.predict(X_train_transformed)

        # Accuracy
        accuracy = np.mean(y_train == y_pred)
        print('Accuracy: ', accuracy)

        return y_pred


preprocessor_obj = preprocessor.Preprocessor()
model_obj = model.Model()

pipeline = Pipeline(preprocessor_obj, model_obj)
pipeline.Run(args.data_path, args.inference)
