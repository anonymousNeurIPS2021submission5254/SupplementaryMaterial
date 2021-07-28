import importlib
if importlib.util.find_spec('torch') is None:
    raise ImportError("MLR is implemented in torch here! => conda install -c pytorch pytorch")
xgboost_available = importlib.util.find_spec('xgboost') is not None #conda install -c conda-forge xgboost
catboost_available = importlib.util.find_spec('catboost') is not None #conda install -c conda-forge catboost
lgbm_available = importlib.util.find_spec('lightgbm') is not None #conda install -c conda-forge lightgbm
mars_available = importlib.util.find_spec('pyearth') is not None #conda install -c conda-forge sklearn-contrib-py-earth
fastai_available = importlib.util.find_spec('fastai') is not None #conda install -c fastai fastai
#else excluded from the benchmark

baseline_name = "Baseline"
lm_name = "GLM"
QDA_name = "QDA"
tree_name = "TREE"
ensemble_name = "RF"
spline_name = "MARS"
svm_name = "SVM"
nn_name = "NN"
xgb_name = "GBDT"
mlr_name = "MLR"

fastai_experiment = "run_fastai"
xgb_experiment = "run_xgb"
sklearn_experiment = "run_sklearn"
mlr_experiment = "run_mlr"

def get_benchmarked_methods(regression = True):
    regressor_methods = [(baseline_name, "Intercept", sklearn_experiment),
                         (lm_name, "Ridge", sklearn_experiment),
                         (lm_name, "Lasso", sklearn_experiment),
                         (lm_name, "Enet", sklearn_experiment),
                         (tree_name, "CART", sklearn_experiment),
                         (ensemble_name, "RF", sklearn_experiment),
                         (ensemble_name, "XRF", sklearn_experiment),
                         (xgb_name, "xgb_sklearn", sklearn_experiment),
                         (svm_name, "Kernel", sklearn_experiment),
                         (svm_name, "NuSVM", sklearn_experiment),
                         (nn_name, "MLP_sklearn", sklearn_experiment),
                         (mlr_name, "MLR3", mlr_experiment),
                         (mlr_name, "MLR4", mlr_experiment)]
    regressor_methods += [(spline_name, "MARS", sklearn_experiment)] * mars_available
    regressor_methods += [(xgb_name, "XGBoost", xgb_experiment)] * xgboost_available
    regressor_methods += [(xgb_name, "CAT", xgb_experiment)] * catboost_available
    regressor_methods += [(xgb_name, "LGBM", xgb_experiment)] * lgbm_available
    regressor_methods += [(nn_name, "fastai", fastai_experiment)] * fastai_available 

    classifier_methods = [(baseline_name, "Intercept", sklearn_experiment),
                         (lm_name, "Ridge", sklearn_experiment),
                         (lm_name, "LinearRidge", sklearn_experiment),
                         (lm_name, "Lasso", sklearn_experiment),
                         (lm_name, "Enet", sklearn_experiment),
                         (QDA_name, "QDA", sklearn_experiment),
                         (tree_name, "CART", sklearn_experiment),
                         (tree_name, "XCART", sklearn_experiment),
                         (ensemble_name, "RF", sklearn_experiment),
                         (ensemble_name, "XRF", sklearn_experiment),
                         (xgb_name, "xgb_sklearn", sklearn_experiment),
                         (xgb_name, "ADABoost", sklearn_experiment),
                         (nn_name, "MLP_sklearn", sklearn_experiment),
                         (mlr_name, "MLR3", mlr_experiment),
                         (mlr_name, "MLR4", mlr_experiment)]
    classifier_methods += [(xgb_name, "XGBoost", xgb_experiment)] * xgboost_available
    classifier_methods += [(xgb_name, "CAT", xgb_experiment)] * catboost_available
    classifier_methods += [(xgb_name, "LGBM", xgb_experiment)] * lgbm_available
    classifier_methods += [(nn_name, "fastai", fastai_experiment)] * fastai_available
    return regressor_methods * regression + classifier_methods * (not regression)