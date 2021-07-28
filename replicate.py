# run benchmark
from experiments import *

ensemble_components = ["MLR1","MLR2"] #architectures used for meta-models            
benchmark_bagging_reps = 10 #number of estimator in each bagging model
benchmark_top_valid_cut = 5 #number of estimators aggregated when sorted by validation-set performance

ablation_architectures = ["NN2", "NN2_Ridge", "NN2_Ridge_Permut", "NN2_Ridge_SD", "MLR2"]
ablation_bagging_reps = 10
ablation_top_valid_cut = 2 #performs useless but costless aggregation between the different architectures (discarded when processing results)
ablation_metric = "R2"

dependance_loops = { "target_rotation_scale" : [0.,1e-1,0.5,1,1.5], #the actual value is twice this (for legacy code reasons)
"n_permut" : [0,1,2,4,16,256,1024],
"ridge_init" : [1e-3,1e-1,1e1,1e3,1e5,1e7,1e9, "max_variation"],
"label_noise_scale" : [0.,1e-2, 1e-2*3, 1e-1,1e-1*3],
"width": [16,64,256,1024,4096]}
batch_size_values = [1,16,32,64,128,256,512,1024,2048,4096,8192, 16384]
dependance_metrics = ["R2"]

def benchmark(datasets, task_name, metrics, input_repository, output_file, output_repository, regression, seeds):
    benchmarked_methods = get_benchmarked_methods(regression = regression)
    run_experiment(benchmarked_methods, datasets, task_name, input_repository, output_file, output_repository, regression = regression, seeds = seeds)
    run_ensemble(  ensemble_components, datasets, task_name, input_repository, output_file, output_repository, regression = regression, seeds = seeds, bagging_reps = benchmark_bagging_reps, top_valid_cut = benchmark_top_valid_cut)

    for metric in metrics: 
        processed_output_file = "processed_"+metric+"_"+output_file
        process_benchmark_results(output_repository+output_file, metric).to_csv(output_repository+processed_output_file)
    pass

def ablation(datasets, task_name, metrics, input_repository, output_file, output_repository, regression, seeds):
    run_ensemble(ablation_architectures, datasets, task_name, input_repository, output_file, output_repository, seeds = seeds, bagging_reps = ablation_bagging_reps, top_valid_cut = ablation_top_valid_cut, regression = regression)
    
    processed_output_file = "processed_" + output_file
    process_ablation_results(output_repository+output_file, ablation_metric).to_csv(output_repository+processed_output_file)
    pass

def dependance(datasets, task_name, metrics, input_repository, output_file, output_repository, regression, seeds):
    method_name = "MLR2"
    for parameter_name, values in dependance_loops.items():
        run_dependance_mlr(method_name, parameter_name, values, datasets, task_name, input_repository, output_file, output_repository, seeds = seeds)
    run_dependance_batchsize_mlr(method_name, batch_size_values, datasets, task_name, input_repository, output_file, output_repository, seeds = seeds)   
    
    processed_output_file = "processed_" + output_file
    process_dependance_results(dependance_metrics, output_repository+output_file).to_csv(output_repository+processed_output_file)
    pass

def main(args=None):
    if args is None:
        import sys
        args = sys.argv[1:]
    args = {opt:arg for i in range(int(len(args)/2)) for opt, arg in [(args[i*2],args[i*2+1])]} #alternate key:value
    args = {**{"-experiment":"benchmark",
            "-task":"reg", 
            "-type":"range", 
            "-datasets":"16", 
            "-seeds":"10", 
            "-input_repo":"./", 
            "-output_repo":"./", 
            "-output_file":"results.csv"}, 
            **args} #fill missing arguments
    
    input_repository = args["-input_repo"]
    output_repository = args["-output_repo"]
    output_file = args["-output_file"]
    
    regression = args["-task"][0] in ["r","R"] #r, reg, regression, R,...
    task_name = "regression" if regression else "classification"
    metrics = ["R2"] if regression else ["ACC", "AUC"]
    
    if args["-type"] == "range":
        datasets = range(int(args["-datasets"]))
    elif args["-type"] == "list":
        datasets = map(int, args["-datasets"].split(","))
    else:
        datasets, metrics = [], []
    seeds = int(args["-seeds"])
    if args["-experiment"] not in ["benchmark", "ablation", "dependance"]: return
    eval(args["-experiment"])(datasets, task_name, metrics, input_repository, output_file, output_repository, regression, seeds)
    pass

if __name__ == '__main__':
    main()