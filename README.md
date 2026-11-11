# GrIS water routing model

Hydrological model describing the supra-glacial water movement, crevasse formation, filling and their drainage to the bed. This model has been developed by Caroline Clason in MATLAB and has been translated using Claude Pro to Python. The model will be used during a MRes project at the University of St Andrews, and be applied to the Kangerdlugssuaq catchment in East Greenland to investigate surface-to-bed transfer of meltwater.

Identified potential improvements:
- snow depth calculation: move from complicated nested if-else structure to simple linear regression, including uncertainty (95% CI) - however, is this really needed? - I will use the model with downscaled MAR data, potentially not enhance this since it is changed anyways
- 

