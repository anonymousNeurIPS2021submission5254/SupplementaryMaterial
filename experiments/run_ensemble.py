from .get_dataset import get_dataset
from .write_results import write_results
import time
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import roc_auc_score as auc
from .MLR import MLRNNRegressor, MLRNNClassifier
from .MLR_architectures import *
import torch

#compute aggregated models simultaneously, based on the same set of predictions
#in our experiments, ensemble_components = ["MLR1","MLR2"]

def get_MLR_prediction(X_train, X_test, y_train, y_test, method_name, seed, regression = True):
    #for one estimator, get the validation score and for each test-set observation the predicted value
    parameters = eval(method_name+"_parameters")
    MLR = MLRNNRegressor if regression else MLRNNClassifier
    model = MLR(random_state = seed, **parameters).fit(X_train,y_train)
    prediction = model.predict(X_test).reshape(-1) if regression else model.predict_proba(X_test)[:,1]        
    validation_performance = np.max(model.record["validation"])
    model.delete_model_weights()
    del model
    torch.cuda.empty_cache()
    return prediction, validation_performance

def evaluate_MLR_prediction(y_test, prediction, exp_id, dataset_id, seed, method_category, method_name, end_time,output_file, output_repository, regression = True):
    #for an aggregated prediction vector, compute score and write results
    metrics = ["R2"] if regression else ["ACC","AUC"]
    results = [r2_score(y_test, prediction)] if regression else [acc(y_test, prediction >= 0.5)]
    if not regression: results.append(auc(y_test, prediction))

    result_line = [exp_id, dataset_id, seed, method_category, method_name, end_time]+results
    write_results(result_line, output_file, output_repository, metrics = metrics)
    
def run_ensemble(ensemble_components, 
                            datasets, 
                            input_name, 
                            input_repository, 
                            output_file, 
                            output_repository, 
                            seeds = 10, 
                            bagging_reps = 10,
                            top_valid_cut = 5,
                            regression = True):
    method_category = "MLR"
    for dataset_id in datasets:
        for seed in range(seeds):
            X_train, X_test, y_train, y_test = get_dataset(dataset_id, input_name, input_repository, seed = seed)
            predictions = {}
            validation_performances = {}
            for method_name in ensemble_components:
                predictions[method_name] = []
                validation_performances[method_name] = []
                exp_id = str(dataset_id)+'_'+str(seed)+"_"+str(method_category)+"_"+str(method_name)

                start_time = time.time()
                for rep in range(bagging_reps):
                    prediction, validation_performance = get_MLR_prediction(X_train, X_test, y_train, y_test, method_name, rep + seed * bagging_reps, regression = True)
                    predictions[method_name].append(prediction)
                    validation_performances[method_name].append(validation_performance)

                #Mean performance accross several models
                if regression:
                    result = np.mean([r2_score(y_test, pred) for pred in predictions[method_name]])
                    end_time = (time.time()-start_time)/bagging_reps
                    result_line = [exp_id, dataset_id, seed, method_category, method_name, end_time]+[result]
                    write_results(result_line, output_file, output_repository, metrics = ["R2"])
                else:
                    result = np.mean([acc(y_test, pred >= 0.5) for pred in predictions[method_name]])
                    roc = np.mean([auc(y_test, pred) for pred in predictions[method_name]])
                    end_time = (time.time()-start_time)/bagging_reps
                    result_line = [exp_id, dataset_id, seed, method_category, method_name, end_time]+[result, roc]
                    write_results(result_line, output_file, output_repository, metrics = ["ACC","AUC"])

                #Bagging aggregation
                prediction = np.mean(predictions[method_name], axis = 0)
                method_name = "Bagging_" + method_name
                end_time = 0
                exp_id = str(dataset_id)+'_'+str(seed)+"_"+str(method_category)+"_"+str(method_name)
                evaluate_MLR_prediction(y_test, prediction, exp_id, dataset_id, seed, method_category, method_name, end_time,output_file, output_repository, regression = regression)

            #Ensemble aggregation
            prediction = np.mean([np.mean(predictions[method_name], axis = 0) for method_name in ensemble_components], axis = 0)
            method_name = "ensemble"
            end_time = 0
            exp_id = str(dataset_id)+'_'+str(seed)+"_"+str(method_category)+"_"+str(method_name)
            evaluate_MLR_prediction(y_test, prediction, exp_id, dataset_id, seed, method_category, method_name, end_time,output_file, output_repository, regression = regression)

            #Valid model selection
            predictions = [prediction for method_name in ensemble_components for prediction in predictions[method_name]]
            validation_performances = [validation_performance for method_name in ensemble_components for validation_performance in validation_performances[method_name]]
            top_valid_performances = np.argsort(validation_performances)[-top_valid_cut:]

            #Best model
            method_name = "Best-MLR"  
            end_time = 0
            exp_id = str(dataset_id)+'_'+str(seed)+"_"+str(method_category)+"_"+str(method_name)
            prediction = predictions[top_valid_performances[-1]]
            evaluate_MLR_prediction(y_test, prediction, exp_id, dataset_id, seed, method_category, method_name, end_time,output_file, output_repository, regression = regression)

            #Top top_valid_cut models
            method_name = "Top"+str(top_valid_cut)+"-MLR"   
            end_time = 0
            exp_id = str(dataset_id)+'_'+str(seed)+"_"+str(method_category)+"_"+str(method_name)
            prediction = np.mean([predictions[index] for index in top_valid_performances], axis = 0)
            evaluate_MLR_prediction(y_test, prediction, exp_id, dataset_id, seed, method_category, method_name, end_time,output_file, output_repository, regression = regression)
