# Train 
This is an example Brane package to train a ML model. Import it as follows:

```shell
$ brane import onnovalkering/train
```

The following ENVIRONMENT variables could be set: 

```shell
$export MODEL_NAME='lightgbm' 
EVAL_METRIC='auc' 
MAX_DEPTH=1 
N_ESTIMATORS=30000 
LEARNING_RATE=0.05 
NUM_LEAVES=4095 
COLSAMPLE_BYTREE=0.28 
OBJECTIVE='binary' 
USE_LOCAL=True 
USE_SAMPLED_DATA=True
```
