from .get_dataset import get_dataset
from .write_results import write_results
from .run_fastai import run_fastai
from .run_sklearn import run_sklearn
from .run_xgb import run_xgb
from .run_mlr import run_mlr
import time

def run_experiment(methods, datasets, input_name, input_repository, output_file, output_repository, seeds = 10, regression = True):
    if regression: metrics = ["R2"]
    else: metrics = ["ACC","AUC"]
    for dataset_id in datasets:
        for seed in range(seeds):
            X_train, X_test, y_train, y_test = get_dataset(dataset_id, input_name, input_repository, train_size = 0.8, seed = False)

            for method_category, method_name, function in methods:
                exp_id = str(dataset_id)+'_'+str(seed)+"_"+str(method_category)+"_"+str(method_name)
                start_time = time.time()
                results = eval(function)(X_train, X_test, y_train, y_test, method_name, seed, regression = regression)
                end_time = time.time() - start_time
                
                result_line = [exp_id, dataset_id, seed, method_category, method_name, end_time]+results
                write_results(result_line, output_file, output_repository, metrics = metrics)