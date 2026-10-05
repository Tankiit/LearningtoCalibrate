"""Frozen fit-fold imputation, scaling, predictor and nonconformity score."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline


class ScoreModel:
    def __init__(self,task="regression",seed=0):
        if task not in ("regression","classification"):
            raise ValueError(task)
        self.task=task
        self.transform=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler())
        self.predictor=Ridge(alpha=1.) if task=="regression" else LogisticRegression(max_iter=2000,random_state=seed)

    def fit(self,X_fit,y_fit):
        self.predictor.fit(self.transform.fit_transform(X_fit),y_fit)
        return self

    def features(self,X):
        return self.transform.transform(X)

    def score(self,X,y):
        Z=self.features(X)
        if self.task=="regression":
            return np.abs(np.asarray(y)-self.predictor.predict(Z))
        probabilities=self.predictor.predict_proba(Z)
        class_index={c:i for i,c in enumerate(self.predictor.classes_)}
        try:
            idx=np.array([class_index[c] for c in y])
        except KeyError as e:
            raise ValueError("Label absent from fit fold") from e
        return 1-probabilities[np.arange(len(y)),idx]

    def efficiency(self,X,thresholds):
        if self.task=="regression":
            return 2*np.asarray(thresholds)
        p=self.predictor.predict_proba(self.features(X))
        return np.sum(1-p<=np.asarray(thresholds)[:,None],axis=1)
