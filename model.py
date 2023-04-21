from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

class Model:
    def __init__(self):
        self.ABRFC = AdaBoostClassifier(RandomForestClassifier())
    def fit(self, X, y):
        self.ABRFC.fit(X, y)
    def predict(self, X):
        return self.ABRFC.predict(X)
    def predict_proba(self, X):
        return self.ABRFC.predict_proba(X)
