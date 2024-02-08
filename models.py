import pandas as pd
from typing import Optional
from pydantic import BaseModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from utils import plot_side_by_side, demographic_parity, equal_opportunity, equal_error_rates

class Model(BaseModel):
    name: str
    model: object
    params: dict = {}
    grid_search: bool = True # Whether to perform grid search or not

class ModelEvaluator(BaseModel):
    model: Model
    pos_label: str = 'Sí'
    cv: int = 5
    n_jobs: int = -1
    best_params: dict = {}
    accuracy: float = 0.0
    auc: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    equal_opportunitty_gender: float = 0.0
    demographic_parity_gender: float = 0.0
    equal_error_rates_gender: float = 0.0
    equal_opportunitty_nationality: float = 0.0
    demographic_parity_nationality: float = 0.0
    equal_error_rates_nationality: float = 0.0
    equal_odds_gender: float = 0.0
    equal_odds_nationality: float = 0.0

    class Config:
        arbitrary_types_allowed = True

    def fit(self, X_train:pd.DataFrame, y_train:pd.Series):
        if self.model.grid_search:
            self.model.model = GridSearchCV(
                self.model.model, 
                self.model.params, 
                cv=self.cv, 
                n_jobs=self.n_jobs)
            self.model.model.fit(X_train, y_train)
            self.best_params = self.model.model.best_params_
        return self.model.model.best_estimator_

    def predict(self, X_test:pd.DataFrame):
        return self.model.model.predict(X_test)

    def predict_proba(self, X_test:pd.DataFrame):
        return self.model.model.predict_proba(X_test)[:,1]

    def evaluate(
            self,
            X_train:pd.DataFrame,
            y_train:pd.Series,
            X_test:pd.DataFrame,
            y_test:pd.Series):
        self.model.model = self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.auc = roc_auc_score(y_test, y_pred_proba)
        self.precision = precision_score(y_test, y_pred, pos_label=self.pos_label)
        self.recall = recall_score(y_test, y_pred, pos_label=self.pos_label)
        self.equal_error_rates_gender = equal_error_rates(
            X_test, 
            y_test, 
            y_pred_proba, 
            "V1_sexe_Dona", 
            "V1_sexe_Home")
        self.equal_opportunitty_gender = equal_opportunity(
            X_test, 
            y_test, 
            y_pred_proba, 
            "V1_sexe_Dona", 
            "V1_sexe_Home")
        self.demographic_parity_gender = demographic_parity(
            X_test, 
            y_pred_proba, 
            "V1_sexe_Dona", 
            "V1_sexe_Home")
        self.equal_error_rates_nationality = equal_error_rates(
            X_test, 
            y_test, 
            y_pred_proba, 
            "V2_estranger_Espanyol", 
            "V2_estranger_Estranger")
        self.equal_opportunitty_nationality = equal_opportunity(
            X_test, 
            y_test, 
            y_pred_proba, 
            "V2_estranger_Espanyol", 
            "V2_estranger_Estranger")
        self.demographic_parity_nationality = demographic_parity(
            X_test, 
            y_pred_proba, 
            "V2_estranger_Espanyol", 
            "V2_estranger_Estranger")
        self.equal_odds_gender = (
            self.equal_opportunitty_gender + 
            self.equal_error_rates_gender)
        self.equal_odds_nationality = (
            self.equal_opportunitty_nationality + 
            self.equal_error_rates_nationality)
        print("Model: %s" % self.model.name)
        print("Best params: %s" % self.best_params)
        print("Accuracy: %.2f" % 
              self.accuracy)
        print("AUC: %.2f" % 
              self.auc)
        plot_side_by_side(
            y_pred_proba[y_test==self.pos_label], 
            y_pred_proba[y_test!=self.pos_label], 
            self.model.name,
            y_test, 
            y_pred_proba, 
            pos_label=self.pos_label)