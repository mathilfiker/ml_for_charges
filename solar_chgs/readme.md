This repository contains another model for predicting ESP partial charges at the PBE0 level of theory.
The architecture is the same as described in the main text, with the difference that atomic descriptors are created using SO3LR (MIT license). 
For installation, refer to the **environment.yaml** file.
For more information regarding SO3LR, please refer to the original paper (https://doi.org/10.1021/jacs.5c09558) and the official repository (https://github.com/general-molecular-simulations/so3lr.git) .
Validation of the model is still ongoing, but the test error is comparable with the one of the model based on MACE-OFF descriptors.
