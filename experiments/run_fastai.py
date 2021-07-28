#fastai
from fastai.tabular.all import *
import pandas as pd
from scipy.special import expit as logistic_func
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import roc_auc_score as auc

def run_fastai(X_train, X_test, y_train, y_test, method_name, seed, regression = True):
    #no simple way to set random state
    #forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628
    df = pd.DataFrame(X_train)
    df["target"] = y_train
    dls = TabularPandas(df, procs=[],
                       cat_names = [],
                       cont_names = range(len(df.columns)-1),
                       y_names='target',
                       splits=RandomSplitter(valid_pct=0.2)(range_of(df))).dataloaders(bs=64)
    if regression:
        learn = tabular_learner(dls, metrics=rmse)
        learn.cbs = [learn.cbs[0]]
        learn.fit_one_cycle(200)
        result = r2_score(y_test, learn.get_preds(dl=learn.dls.test_dl(pd.DataFrame(X_test)))[0].numpy())
        del learn
        return [result]
    else:
        learn = tabular_learner(dls, metrics=accuracy)
        learn.cbs = [learn.cbs[0]]
        learn.fit_one_cycle(200)
        decision = learn.get_preds(dl=learn.dls.test_dl(pd.DataFrame(X_test)))[0].numpy()
        preds = (decision>0).astype(int)
        result, roc = acc(y_test,preds), auc(y_test, logistic_func(decision))
        return [result, roc]