from brane_train import train

#local testing
def test_preprocess():
  assert train("lightgbm" "auc", 1, 30000, 0.05, 4095, 0.28, "binary", True, True) == "Model saved succesfully"
  
