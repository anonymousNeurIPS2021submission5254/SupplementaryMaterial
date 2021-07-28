#xgboost, catboost, lightgbm
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import roc_auc_score as auc

def run_xgb(X_train, X_test, y_train, y_test, method_name, seed, regression = True):
    if regression:
        if method_name == "XGBoost":
            from xgboost import XGBRegressor as XGB
            model = XGB(random_state = seed, objective ='reg:squarederror', verbose = False).fit(X_train,y_train)
            result = model.score(X_test, y_test)

        elif method_name == "CAT":
            from catboost import CatBoostRegressor as CAT
            model = CAT(random_seed=seed, logging_level='Silent').fit(X_train,y_train)
            result = model.score(X_test, y_test)
            
        elif method_name == "LGBM":
            from lightgbm.sklearn import LGBMRegressor as LGBM
            model = LGBM(random_state=seed).fit(X_train,y_train)
            result = r2_score(y_test, model.predict(X_test)) 
            
        del model    
        return [result]
    
    else:
        if method_name == "XGBoost":
            from xgboost import XGBClassifier as XGB
            model = XGB(random_state = seed, verbose = False).fit(X_train,y_train)
            result = model.score(X_test, y_test)
            roc = auc(y_test, model.predict_proba(X_test)[:,1])

        elif method_name == "CAT":
            from catboost import CatBoostClassifier as CAT
            model = CAT(random_seed=seed, logging_level='Silent').fit(X_train,y_train)
            result = model.score(X_test, y_test)
            roc = auc(y_test, model.predict_proba(X_test)[:,1])
            
        elif method_name == "LGBM":
            from lightgbm.sklearn import LGBMClassifier as LGBM
            model = LGBM(random_state=seed).fit(X_train,y_train)
            result = acc(y_test, model.predict(X_test))
            roc = auc(y_test, model.predict_proba(X_test)[:,1])
            
        del model
        return [result, roc]