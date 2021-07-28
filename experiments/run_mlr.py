from .MLR import MLRNNRegressor, MLRNNClassifier
from .MLR_architectures import *
import torch
from sklearn.metrics import roc_auc_score as auc

def run_mlr(X_train, X_test, y_train, y_test, method_name, seed, regression = True):
    parameters = eval(method_name+"_parameters")
    MLR = MLRNNRegressor if regression else MLRNNClassifier
    model = MLR(random_state = seed, **parameters).fit(X_train,y_train)
    results = [model.score(X_test, y_test)]
    if not regression: results.append(auc(y_test, model.predict_proba(X_test)[:,1]))
    model.delete_model_weights()
    del model
    torch.cuda.empty_cache()
    return results