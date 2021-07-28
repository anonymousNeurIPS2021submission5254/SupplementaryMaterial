from copy import deepcopy
#set architectures hyper-parameters
common_parameters = {"max_runtime" : 900, "width":1024}
MLR1_parameters = deepcopy(common_parameters)
MLR1_parameters["depth"] = 1
MLR1_parameters["learning_rate"] = 1e-2
MLR1_parameters["max_iter"] = 200

MLR2_parameters = deepcopy(common_parameters)
MLR2_parameters["depth"] = 2
MLR2_parameters["learning_rate"] = 1e-3
MLR2_parameters["max_iter"] = 200

MLR3_parameters = deepcopy(common_parameters)
MLR3_parameters["depth"] = 3
MLR3_parameters["learning_rate"] = 1e-3 /3
MLR3_parameters["max_iter"] = 400

MLR4_parameters = deepcopy(common_parameters)
MLR4_parameters["depth"] = 4
MLR4_parameters["learning_rate"] = 1e-4
MLR4_parameters["max_iter"] = 400

NN2_parameters = deepcopy(MLR2_parameters)
NN2_parameters["ridge_init"] = False
NN2_parameters["n_permut"] = False
NN2_parameters["target_rotation_scale"] = False

NN2_Ridge_parameters = deepcopy(MLR2_parameters)
NN2_Ridge_parameters["ridge_init"] = "max_variation"
NN2_Ridge_parameters["n_permut"] = False
NN2_Ridge_parameters["target_rotation_scale"] = False

NN2_Ridge_SD_parameters = deepcopy(MLR2_parameters)
NN2_Ridge_SD_parameters["ridge_init"] = "max_variation"
NN2_Ridge_SD_parameters["n_permut"] = False
NN2_Ridge_SD_parameters["target_rotation_scale"] = 0.5

NN2_Ridge_Permut_parameters = deepcopy(MLR2_parameters)
NN2_Ridge_Permut_parameters["ridge_init"] = "max_variation"
NN2_Ridge_Permut_parameters["n_permut"] = 16
NN2_Ridge_Permut_parameters["target_rotation_scale"] = False