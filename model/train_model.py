import sys
sys.path.insert(0, "./")

import logging
from utils.logging_setup import setup_logging

import pickle

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


def classification_model(model, X_train, y_train):
    #Fit the model:
    model.fit(X_train,y_train)
    
    train_predictions = model.predict(X_train)
    precision = precision_score(y_train, train_predictions)
    recall = recall_score(y_train, train_predictions)
    f1 = f1_score(y_train, train_predictions)
    
    logging.getLogger(__name__).info(f"Precision {precision:.3%}")
    logging.getLogger(__name__).info(f"Recall {recall:.3%}")
    logging.getLogger(__name__).info(f"F1 score {f1:.3%}")

    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        
    logging.getLogger(__name__).info("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(cv_score)))
    

def train_model():
    setup_logging()
    data = pd.read_csv("./data/heart.csv")
        
    Xtrain = data.drop('target', axis=1).copy()
    Ytrain = data['target'].copy()

    log_reg = LogisticRegression(solver="liblinear")
    classification_model(log_reg, Xtrain, Ytrain)

    pickle.dump(log_reg, open("./model/trained_model.pkl", 'wb+'))
    

if __name__ == "__main__":
    train_model()