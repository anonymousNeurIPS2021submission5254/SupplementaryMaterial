# This repository contains supplementary materials for the Neurips 2021 submission 5254 : Muddling Label Regularization
It contains:
 - MLRNN.py, a self-contained implementation following the Scikit-learn API.
 - MLRNN.ipynb, a self-contained jupyter notebook with a step-by-step demo on the boston dataset.
 - preprocessed_datasets/, a repository with all the datasets used in the article, in a .npy format.
 - replicate.sh, a script to replicate the results from the article, with associated .py scripts in the experiments/ folder.
 - notebook_version/Benchmark MLR Git.ipynb, a jupyter notebook to interactively replicate these results.
 
 # Remarks
 
 Also note:
 - To use MLR in other projects, MLRNN.py is all you need (and sklearn and torch also). 
 - To replicate all the experiments, a machine with a good gpu needs several days. If you want to do so you should run replicate.sh. 
 - If you only want to check for specific parts of our results, on a subset of datasets, using a subset of the benchmarked methods and architectures, with smaller number of random seeds, you should play with Benchmark MLR Git.ipynb. You could also try to add new datasets (format: sample by (feature+target) .npy matrix) or new methods (format: class with .fit(X,y), .predict(X), .score(X,y) .predict_proba(X)).
 - Please create a git issue if you encounter errors, bugs, anomalies, questions, etc..
 

 

