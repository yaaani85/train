## WSCBS2021 Training

This implementation is based on [bogorodvo's solution](https://www.kaggle.com/bogorodvo/lightgbm-baseline-model-using-sparse-matrix).

This is an example Brane package to train a ML model. Import it as follows:

```shell
$ brane import yaaani85/wscbs2021-training
```

The following environment variables can be set: 

```shell
$ export MODEL_NAME='lightgbm' 
$ export EVAL_METRIC='auc' 
$ export MAX_DEPTH=1 
$ export N_ESTIMATORS=30000 
$ export LEARNING_RATE=0.05 
$ export NUM_LEAVES=4095 
$ export COLSAMPLE_BYTREE=0.28 
$ export OBJECTIVE='binary' 
$ export USE_LOCAL=True 
$ export USE_SAMPLED_DATA=True
```

For an overview of the parameters of the brane package, you can `test` the package
```shell
$ brane --debug test training
```


You also need to push the package to be able to import it in your remote session or jupyterlab notebook:
```shell
brane push training 1.0.0
```
