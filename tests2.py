from lzma import CHECK_CRC32
from re import X
import numpy as np
from random import shuffle, randint
import os
from datetime import datetime
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from libs.models_creation import SimpleDNN_Model, ResNet_Model, VGG16_Model
import libs.visualization_package as vis
import libs.utils as utils
tf.random.set_seed(1234)


test_params = {
  "model" : "ResNet",
  "dataset" : "CIFAR100",
  "learning_type" : "FBCL",
  "epoch_start_test" : [15,30],
  "batch_size" : 32,
  "epochs" : 50,
}

PROGRAM_PATH = os.getcwd()


if __name__ == '__main__':
  BATCH_SIZE = test_params['batch_size']
  EPOCHS = test_params['epochs']
  MODEL_TYPE = test_params["model"]
  LEARNING_TYPE = test_params["learning_type"]
  DATASET = test_params["dataset"]
  TEST_NUMBER = f"{datetime.now()}_{MODEL_TYPE}{BATCH_SIZE}_{DATASET}"
  DATA_FOLDER_PATH = os.path.join(PROGRAM_PATH, "data_calibrated", DATASET)
  CHECKPOINT_PATH = os.path.join(DATA_FOLDER_PATH, "checkpoints", test_params["model"])
  FIG_SAVE_PATH = os.path.join(PROGRAM_PATH, "tests", TEST_NUMBER,"pictures")
  RESULT_SAVE_PATH = os.path.join(PROGRAM_PATH, "tests", TEST_NUMBER, "results")
  TEST_EPOCH_START = np.array(test_params['epoch_start_test']) - 1
  os.makedirs(CHECKPOINT_PATH,exist_ok=True)
  os.makedirs(FIG_SAVE_PATH,exist_ok=True)
  os.makedirs(RESULT_SAVE_PATH,exist_ok=True)
  
  cascade_df = pd.DataFrame([])
  
  # Wczytywanie danych
  X_train = np.load(os.path.join(DATA_FOLDER_PATH, "X_train.npy"), allow_pickle=True)
  y_train = np.load(os.path.join(DATA_FOLDER_PATH, "y_train.npy"), allow_pickle=True)
  X_valid = np.load(os.path.join(DATA_FOLDER_PATH, "X_valid.npy"), allow_pickle=True)
  y_valid = np.load(os.path.join(DATA_FOLDER_PATH, "y_valid.npy"), allow_pickle=True)
  X_test = np.load(os.path.join(DATA_FOLDER_PATH, "X_test.npy"), allow_pickle=True)
  y_test = np.load(os.path.join(DATA_FOLDER_PATH, "y_test.npy"), allow_pickle=True)
  # skalowanie danych
  X_train = X_train/255
  X_valid = X_valid/255
  X_test = X_test/255
  # Zczytywanie kształtu danych wejściowych i wyjściowych
  input_shape = X_train.shape[1:]
  output_shape = len(np.unique(y_train))
  # Tworzenie modelu
  if MODEL_TYPE == "SimpleDNN":
    model_class = SimpleDNN_Model(input_shape, 10, output_shape)
    model = model_class.create_model()
  elif MODEL_TYPE == "ResNet":
    if X_train.ndim == 3:
      X_valid = utils.make_3_channels(X_valid)
      X_test = utils.make_3_channels(X_test)
      X_train = utils.make_3_channels(X_train)
      # X_train = np.dstack([X_train, X_train, X_train])
      input_shape = X_train.shape[1:]
    model = ResNet_Model(output_shape, input_shape, 5)
  elif MODEL_TYPE == "VGG16":
    if X_train.ndim == 3:
      X_valid = utils.make_3_channels(X_valid)
      X_test = utils.make_3_channels(X_test)
      X_train = utils.make_3_channels(X_train)
      X_train = utils.make_compitable_with_VGG16(X_train)
      X_valid = utils.make_compitable_with_VGG16(X_valid)
      X_test = utils.make_compitable_with_VGG16(X_test)
      input_shape = X_train.shape[1:]
    model = VGG16_Model(output_shape, input_shape, 5)
  # Przygotowywanie danych wyjściowych
  y_train = to_categorical(y_train)
  y_valid = to_categorical(y_valid)

  # X_train, y_train = X_train[:15000], y_train[:15000]
  # X_valid, y_valid = X_valid[:50], y_valid[:50]
  # X_test, y_test = X_test[:50], y_test[:50]
  
  for epoch_start in TEST_EPOCH_START:
    temp_cascade_df = pd.DataFrame([])

    for learning_type in ["FBCL","BFCL", "RSCL"]:
      if MODEL_TYPE == "SimpleDNN":
        model_class = SimpleDNN_Model(input_shape, 10, output_shape)
        model = model_class.create_model()
      elif MODEL_TYPE == "ResNet":
        model = ResNet_Model(output_shape, input_shape, 5)
      elif MODEL_TYPE == "VGG16":
        model = VGG16_Model(output_shape, input_shape, 5)
      model.load_weights(os.path.join(CHECKPOINT_PATH, f"{epoch_start}.ckpt"))
      dense_layers = []
    
      if MODEL_TYPE == "SimpleDNN":
        for i, layer in enumerate(model.layers):
          if "dense" in layer.name:
            layer.trainable = False
            dense_layers.append(i)
      else:
        for i, layer in enumerate(model.layers[0].layers):
          if "dense" in layer.name:
            layer.trainable = False
            dense_layers.append(i)
      if learning_type == "BFCL": 
        dense_layers.reverse()
      elif learning_type == "RSCL":
        shuffle(dense_layers)

      CASCADE_FIG_SAVE_PATH = os.path.join(FIG_SAVE_PATH, learning_type, str(epoch_start+1))
      os.makedirs(CASCADE_FIG_SAVE_PATH, exist_ok=True)
      
      remaining_epochs = (EPOCHS - epoch_start)//len(dense_layers)
      model, history = utils.my_fit_function_cascade_2(model, X_train, y_train, (X_valid, y_valid), dense_layers, BATCH_SIZE, MODEL_TYPE, remaining_epochs)
      vis.visualize_loss(history, "test_loss",CASCADE_FIG_SAVE_PATH, test_epoch_start = epoch_start)
      vis.visualize_accuracy(history, "test_accuracy",CASCADE_FIG_SAVE_PATH, test_epoch_start = epoch_start)
      # prediction = np.argmax(model.predict(X_test), axis=1)
      predictions = np.argmax(model.predict(X_test, batch_size = BATCH_SIZE), axis=1)
      metrics2 = accuracy_score(y_test, predictions)
      history["Test_acc"] = [metrics2]
      history["Epoch_test_start"] = epoch_start
      history["Cascade_type"] = learning_type
      cascade_training_df_temp = pd.DataFrame.from_dict(history)
      if temp_cascade_df.empty:
        temp_cascade_df = cascade_training_df_temp
      else:
        temp_cascade_df = pd.concat([temp_cascade_df, cascade_training_df_temp])
      print(f"""
      Second model is {metrics2*100}% accurate""")
    if cascade_df.empty:
      cascade_df = temp_cascade_df
    else:
      cascade_df = pd.concat([cascade_df, temp_cascade_df])
  cascade_df.to_parquet(os.path.join(RESULT_SAVE_PATH, "cascade_training.parquet"))