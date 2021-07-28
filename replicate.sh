#!/bin/bash 
REG="regression"
CLF="classification"
ABL="ablation_study"
DEP="dependance_study"
DS="preprocessed_datasets/"
RES="outputs/"

mkdir -p $RES
python replicate.py -experiment benchmark -task reg -seeds 10 -datasets 16 -type range -input_repo $DS -output_repo $RES -output_file $REG
python replicate.py -experiment benchmark -task clf -seeds 10 -datasets 16 -type range -input_repo $DS -output_repo $RES -output_file $CLF
python replicate.py -experiment ablation   -task reg -seeds 100 -datasets -1,0,1 -type list -input_repo $DS -output_repo $RES -output_file $ABL
python replicate.py -experiment dependance -task reg -seeds 100 -datasets -1,0,1 -type list -input_repo $DS -output_repo $RES -output_file $DEP