import numpy as np
import pandas as pd

class_dic = {'fastai':"NN",
    "MLP_sklearn":"NN", 
    'MLR1':"MLR", 
    'Bagging_MLR1':"MLR", 
    'MLR2':"MLR", 
    'Bagging_MLR2':"MLR",
    'ensemble':"MLR", 
    'Best-MLR':"MLR", 
    'Top5-MLR':"MLR", 
    'MLR3':"MLR", 
    'MLR4':"MLR",
    'XGBoost':"GBDT", 
    'CAT':"GBDT", 
    'LGBM':"GBDT",
    'Ridge':"LM", 
    'Lasso':"LM", 
    'Enet':"LM",
    'LinearRidge':"LM", 
    'QDA':"QDA", 
    'CART':"TREE", 
    'XCART':"TREE", 
    'RF':"RF", 
    'XRF':"RF", 
    'Bagging':"RF",
    'ADABoost':"GBDT", 
    'xgb_sklearn':"GBDT", 
    'Intercept':"Baseline", 
    'MARS':"MARS", 
    "NuSVM":"SVM",
    "Kernel":"SVM"}


def process_benchmark_results(result_file, metric, reference = "ensemble"):
    q_values = [0.90,0.95,0.98]
    kept_columns = ["class"] + [metric+col for col in ["","_std", "_rank","_rank_std","_PMA","_PMA_std"]]+[metric+"_P"+str(q) for q in q_values]
    
    df2 = pd.read_csv(result_file)
    df2["class"] = [class_dic[method] for method in df2["method"].values]
    
    #Compute PMA
    df_max = df2.groupby(["dataset","seed"]).max()
    df2.set_index(["dataset","seed"],inplace = True)
    df2[metric+"_max"] = df_max[metric]
    df2.reset_index(inplace = True)
    df2[metric+"_PMA"] = df2[metric]/df2[metric+"_max"]

    #Compute P90, P95, P98
    df_p = df2.groupby(["dataset","seed","class"]).max().reset_index()
    for q in q_values:
        df_p[metric+"_P"+str(q)] = (df_p[metric+"_PMA"] > q).astype(int)
    df_p = df_p.groupby('class').mean()

    #Use ensemble(MLR1+MLR2) results accross all seeds as a baseline to mesure standard deviation for all methods
    ensemble_ref = df2[df2["method"]==reference].set_index(["dataset","seed"])
    df2.set_index(["dataset","seed"], inplace = True)
    df2[metric+"_ref"] = ensemble_ref[metric]
    df2[metric+"_std"] = df2[metric].values-df2[metric+"_ref"].values
    df2.reset_index(inplace = True)
    df_mean_seed = df2.groupby(["dataset","method"]).mean().reset_index()
    df_mean_seed["class"] = [class_dic[method] for method in df_mean_seed["method"].values]
    df_mean_seed.sort_values(["dataset","class",metric],inplace = True)
    df_mean_seed_max_class = df_mean_seed.groupby(["dataset","class"]).last().reset_index()
    df_mean_seed_max_class.set_index("method",inplace = True)
    df_mean_seed_max_class[metric + "_std"] = df2.groupby(["method"]).std()[metric+"_std"]
    df_mean_seed_max_class[metric+"_PMA"] = df2.groupby("method").mean()[metric+"_PMA"]
    df_mean_seed_max_class[metric+"_PMA_std"] = df2.groupby("method").std()[metric+"_PMA"]
    df_mean_seed_max_class.reset_index(inplace = True)
    df_mean_seed_max_class_mean_ds = df_mean_seed_max_class.groupby(["class"]).mean().reset_index()
    
    #Compute Friedman Rank
    df_rank = df2.groupby(["dataset","seed","class"]).max().reset_index().sort_values(["dataset","seed",metric],ascending = False)
    df_rank[metric+"_rank"]= np.arange(len(df_rank))%len(df_rank["class"].unique()) +1
    df_mean_seed_max_class_mean_ds.set_index("class", inplace = True)
    df_mean_seed_max_class_mean_ds[metric+"_rank"] = df_rank.groupby("class").mean()[metric+"_rank"]
    df_mean_seed_max_class_mean_ds[metric+"_rank_std"] = df_rank.groupby("class").std()[metric+"_rank"]
    for q in q_values:
        df_mean_seed_max_class_mean_ds[metric+"_P"+str(q)] = df_p[metric+"_P"+str(q)]
    df_mean_seed_max_class_mean_ds.reset_index(inplace = True)
    
    #return results with only usefull columns
    return df_mean_seed_max_class_mean_ds[kept_columns]

def process_ablation_results(result_file, metric, reference = 'Bagging_MLR2'):
    df = pd.read_csv(result_file)
    
    #keep only single and bagging estimators (i.e. no ensemble)
    kept_methods = [prefix + method_name  for method_name in ["NN2", "NN2_Ridge", "NN2_Ridge_Permut", "NN2_Ridge_SD", "MLR2"] for prefix in ["", "Bagging_"]]
    df = df[np.isin(df["method"].values,kept_methods)]
    
    #use MLR2_Bagging as a reference to compute result variation accross all seeds
    df.set_index(["dataset","seed"], inplace=True)
    df["ref"] = df[df["method"] == reference][metric]
    df.reset_index(inplace=True)
    df['std'] = df[metric] - df["ref"]
    
    #Average accross all seeds
    mean_df = df.groupby(["method"]).mean()
    std_df = df.groupby(["method"]).std()
    mean_df["std"] = std_df["std"]
    return mean_df.reset_index()[["method", metric, "std"]]

def process_dependance_results(metrics, result_file): 
    kept_columns = metrics +["time", "best_iter", "valid_max", "lambda_init"]
    return pd.read_csv(result_file).groupby(["parameter_name","dataset","value"]).mean()[kept_columns]
