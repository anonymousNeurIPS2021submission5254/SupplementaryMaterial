from .get_dataset import get_dataset
from .write_results import write_results
import time
import numpy as np
from copy import deepcopy
from .MLR import MLRNNRegressor as MLR
from .MLR_architectures import *
import torch

def run_dependance_mlr(method_name, parameter_name, values, datasets, input_name, input_repository, output_file, output_repository, seeds = 10):
    method_category = "MLR"
    for dataset_id in datasets:
        for seed in range(seeds):
            X_train, X_test, y_train, y_test = get_dataset(dataset_id, input_name, input_repository, train_size = 0.8, seed = False)    
            for value in values:
                exp_id = str(dataset_id)+'_'+str(seed)+"_"+str(method_category)+"_"+str(method_name)+"_"+str(parameter_name)
                parameters = deepcopy(eval(method_name+"_parameters"))
                parameters.update({parameter_name : value})
                
                start_time = time.time()
                model = MLR(random_state = seed, **parameters).fit(X_train, y_train)
                result = model.score(X_test, y_test)
                best_iter = model.best_iter
                valid_max = model.record["validation"][model.best_iter]
                lambda_init = model.record["lambda"][0]
                model.delete_model_weights()
                del model
                torch.cuda.empty_cache()
                end_time = time.time() - start_time
                
                result_line = [exp_id, dataset_id, seed, method_category, method_name, end_time] + [parameter_name, value, result, best_iter, valid_max, lambda_init]
                write_results(result_line, output_file, output_repository, metrics = ["parameter_name", "value", "R2", "best_iter", "valid_max", "lambda_init"])
                
def run_dependance_batchsize_mlr(method_name, values, datasets, input_name, input_repository, output_file, output_repository, seeds = 10):
    method_category = "MLR"
    parameter_name = "batch_size"
    for dataset_id in datasets:
        for seed in range(seeds):
            X_train, X_test, y_train, y_test = get_dataset(dataset_id, input_name, input_repository, train_size = 0.8, seed = False)    
            n = X_train.shape[0]
            if n < np.max(values):
                dataset_values = [value for value in values if value < n] + [n]
            else: 
                dataset_values = [value for value in values]
            for value in values:
                exp_id = str(dataset_id)+'_'+str(seed)+"_"+str(method_category)+"_"+str(method_name)+"_"+str(parameter_name)
                parameters = deepcopy(eval(method_name+"_parameters"))
                parameters.update({parameter_name : value})
                
                start_time = time.time()
                model = MLR(random_state = seed, **parameters).fit(X_train, y_train)
                result = model.score(X_test, y_test)
                best_iter = model.best_iter
                valid_max = model.record["validation"][model.best_iter]
                lambda_init = model.record["lambda"][0]
                model.delete_model_weights()
                del model
                torch.cuda.empty_cache()
                end_time = time.time() - start_time
                
                result_line = [exp_id, dataset_id, seed, method_category, method_name, end_time] + [parameter_name, value, result, best_iter, valid_max, lambda_init]
                write_results(result_line, output_file, output_repository, metrics = ["parameter_name", "value", "R2", "best_iter", "valid_max", "lambda_init", ])