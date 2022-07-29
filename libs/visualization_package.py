import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_loss(history, name, save_path, test_epoch_start = 0):
  y_loss = history['loss'][0]
  y_val_loss = history['val_loss'][0]
  #y_loss = np.mean(loss, axis=1)
  x = np.arange(0, y_loss.shape[0],1) + test_epoch_start 
  plt.plot(x,y_loss)
  plt.plot(x,y_val_loss)
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(os.path.join(save_path, f"{name}.png"))
  plt.clf()

def visualize_accuracy(history, name, save_path, test_epoch_start = 0):
  y_acc = history['accuracy'][0]
  y_val_acc = history['val_accuracy'][0]
  #y_acc = np.mean(acc, axis=1)
  x = np.arange(0, y_acc.shape[0],1) + test_epoch_start
  plt.plot(x,y_acc)
  plt.plot(x,y_val_acc)
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(os.path.join(save_path, f"{name}.png"))
  plt.clf()
  

def visualize_accuracy_full(history_normal, history_cascade, name, save_path):
  test_epoch_start = history_cascade["Epoch_test_start"].values[0]
  cascade_epochs = history_cascade["accuracy"].values[0].shape[0]
  learning_types = history_cascade["Cascade_type"].values
  normall_acc = history_normal["accuracy"][0]
  cascade_acc = history_cascade["accuracy"].values
  normal_x = np.arange(0, normall_acc.shape[0],1)
  cascade_x = np.arange(0, cascade_epochs,1) + test_epoch_start + 1
  plt.plot(normal_x, normall_acc, label="Normal training")
  for i in range(cascade_acc.shape[0]):
    plt.plot(cascade_x, cascade_acc[i], label=learning_types[i])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(loc='upper left')
  plt.savefig(os.path.join(save_path, f"{name}_{test_epoch_start}.png"))
  plt.clf()


def visualize_loss_full(history_normal, history_cascade, name, save_path):
  test_epoch_start = history_cascade["Epoch_test_start"].values[0]
  cascade_epochs = history_cascade["loss"].values[0].shape[0]
  learning_types = history_cascade["Cascade_type"].values
  normall_loss = history_normal["loss"][0]
  cascade_loss = history_cascade["loss"].values
  normal_x = np.arange(0, normall_loss.shape[0],1)
  cascade_x = np.arange(0, cascade_epochs,1) + test_epoch_start + 1
  plt.plot(normal_x, normall_loss, label="Normal training")
  for i in range(cascade_loss.shape[0]):
    plt.plot(cascade_x, cascade_loss[i], label=learning_types[i])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(loc='upper left')
  plt.savefig(os.path.join(save_path, f"{name}_{test_epoch_start}.png"))
  plt.clf()