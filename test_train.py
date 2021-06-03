from brane_train import train

#local testing
def test_preprocess():
  assert train(True, True) == "Model saved succesfully"
  
