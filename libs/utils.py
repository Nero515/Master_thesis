from http.client import NON_AUTHORITATIVE_INFORMATION
from tracemalloc import start
from tensorflow.keras.losses import CategoricalCrossentropy
from keras.preprocessing.image import img_to_array, array_to_img
import numpy as np
import os, time
import tensorflow as tf


def make_3_channels(X_data):
    X_data_new = np.zeros((X_data.shape[0], X_data.shape[1], X_data.shape[2], 3))
    X_data_new[:,:,:,0] = X_data
    X_data_new[:,:,:,1] = X_data
    X_data_new[:,:,:,2] = X_data
    return X_data_new

def make_compitable_with_VGG16(X_data):
    if X_data.shape[1] < 48:
        X_data_new = np.zeros((X_data.shape[0],48,48,3))
        X_data_new[:,:X_data.shape[1],:X_data.shape[2],:] = X_data
    return X_data_new

def my_fit_function(model, X_train, y_train, val_dataset, epochs, batch_size, save_weights_path, save_weights_num):
    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.Adam()
    # Instantiate a loss function.
    loss_fn =  CategoricalCrossentropy(
                from_logits=False,
                )
        # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    acc_fn = tf.keras.metrics.Accuracy()
    loss_vector = np.zeros((epochs, len(list(train_dataset))))
    val_loss_vector = np.zeros((epochs))
    accuracy_vector = np.zeros((epochs, len(list(train_dataset))))
    val_accuracy_vector = np.zeros((epochs))
    gradient_change = np.zeros(int((len(model.trainable_weights))/2))
    training_time = 0
    for i in range(epochs):
        start_time = time.time()
        print(f"\nStarting epoch {i}")
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

          
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = model(x_batch_train, training=True)  # Logits for this minibatch
                # logits = model.predict(x_batch_train)

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)
                accuracy_value = acc_fn(np.argmax(y_batch_train, axis=1), np.argmax(logits, axis=1))

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            loss_vector[i, step] = loss_value
            accuracy_vector[i, step] = accuracy_value
#            gradient_change += [np.sum(np.abs(grads[i])) + np.sum(np.abs(grads[i+1])) for i in range(0,20,2)]
            # Log every 200 batches.
            if step % 20 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))
        end_time = time.time()
        training_time += end_time - start_time
        #predictions = model(val_dataset[0], training=False)
        predictions = model.predict(val_dataset[0], batch_size = batch_size)
        val_loss = loss_fn(val_dataset[1],predictions)
        val_accuracy = acc_fn(np.argmax(val_dataset[1], axis=1), np.argmax(predictions, axis=1))
        val_loss_vector[i] = val_loss
        val_accuracy_vector[i] = val_accuracy
        print(f"""
                Epoch: {i+1},
                Loss: {loss_vector[i].mean()},
                Val loss: {val_loss_vector[i].mean()},
                Accuracy: {accuracy_vector[i,:].mean()},
                Val accuracy: {val_accuracy_vector[i].mean()}
                """)
        if i in save_weights_num:
            model.save_weights(os.path.join(save_weights_path, f"{i}.ckpt"))
    print(f"""
          Training time: {training_time}
          Mean training epoch time: {(training_time)/epochs}""")
#    gradient_vector = gradient_change/(step*epochs)
    history = {
    "accuracy" : [np.mean(accuracy_vector, axis=1)],
    "val_accuracy" : [val_accuracy_vector],
    "loss" : [np.mean(loss_vector, axis=1)],
    "val_loss" : [val_loss_vector],
#    "gradient" : [gradient_vector],
    "training_time" : [training_time],
    "training_time_epoch" : [(training_time)/epochs],
    "batch_size" : [batch_size]
    }
    return model, history
    
def my_fit_function_cascade(model, X_train, y_train, val_dataset, dense_layers, batch_size, model_type, epochs):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.Adam()
    # Instantiate a loss function.
    loss_fn =  CategoricalCrossentropy(
                from_logits=False,
                )
    acc_fn = tf.keras.metrics.Accuracy()
    loss_vector = np.zeros((len(dense_layers)*epochs, len(list(train_dataset))))
    val_loss_vector = np.zeros((len(dense_layers)*epochs))
    accuracy_vector = np.zeros((len(dense_layers)*epochs, len(list(train_dataset))))
    val_accuracy_vector = np.zeros((len(dense_layers)*epochs))
    trainable_params_vector = np.zeros((len(dense_layers)*epochs))
    non_trainable_params_vector = np.zeros((len(dense_layers)*epochs))
    training_time = 0
    i = 0                          
#    gradient_vector = np.zeros((len(dense_layers)))
    for _ in range(epochs):
        for layer_to_freeze in dense_layers:
            start_time = time.time()
            print(f"\nNow starting to freeze {layer_to_freeze}")
            if model_type == "SimpleDNN":
                model.layers[layer_to_freeze].trainable = True
            else:
                model.layers[0].layers[layer_to_freeze].trainable = True
            gradient_change = 0
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                
                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits = model(x_batch_train, training=True)  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value = loss_fn(y_batch_train, logits)
                    accuracy_value = acc_fn(np.argmax(y_batch_train, axis=1), np.argmax(logits, axis=1))

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                loss_vector[i, step] = loss_value
                accuracy_vector[i, step] = accuracy_value
                gradient_change += np.sum(np.abs(grads[0])) + np.sum(np.abs(grads[1]))
                # Log every 200 batches.
                if step % 20 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * batch_size))
        #    gradient_vector[layer_to_freeze-min(dense_layers)] = gradient_change/step
            end_time = time.time()
            training_time += end_time - start_time
            trainable_params_vector[i] = trainableParams =np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
            non_trainable_params_vector[i] = nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
            # predictions = model(val_dataset[0], training=False)
            predictions = model.predict(val_dataset[0], batch_size = batch_size)
            val_loss = loss_fn(val_dataset[1], predictions)
            val_accuracy = acc_fn(np.argmax(val_dataset[1], axis=1), np.argmax(predictions, axis=1))
            val_loss_vector[i] = val_loss
            val_accuracy_vector[i] = val_accuracy
            if model_type == "SimpleDNN":
                model.layers[layer_to_freeze].trainable = False
            else:
                model.layers[0].layers[layer_to_freeze].trainable = False
            
            print(f"""
                    Epoch: {i+1},
                    Loss: {loss_vector[i].mean()},
                    Val loss: {val_loss_vector[i].mean()},
                    Accuracy: {accuracy_vector[i,:].mean()},
                    Val accuracy: {val_accuracy_vector[i].mean()}
                    """)
            i+=1
            
    print(f"""
          Training time: {training_time}
          Mean training epoch time: {(training_time)/len(dense_layers)}""")
    total_params = trainableParams + nonTrainableParams
    history = {
        "accuracy" : [np.mean(accuracy_vector, axis=1)],
        "val_accuracy" : [val_accuracy_vector],
        "loss" : [np.mean(loss_vector, axis=1)],
        "val_loss" : [val_loss_vector],
#        "gradient" : [gradient_vector],
        "training_time" : [training_time],
        "training_time_epoch" : [(training_time)/len(dense_layers)],
        "batch_size" : [batch_size],
        "trainable_params" : [trainable_params_vector],
        "non_trainable_params" : [non_trainable_params_vector],
        "total_params_vector" : [total_params],
    }
    return model, history

def my_fit_function_cascade_2(model, X_train, y_train, val_dataset, dense_layers, batch_size, model_type, epochs):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.Adam()
    # Instantiate a loss function.
    loss_fn =  CategoricalCrossentropy(
                from_logits=False,
                )
    acc_fn = tf.keras.metrics.Accuracy()
    loss_vector = np.zeros((len(dense_layers)*epochs, len(list(train_dataset))))
    val_loss_vector = np.zeros((len(dense_layers)*epochs))
    accuracy_vector = np.zeros((len(dense_layers)*epochs, len(list(train_dataset))))
    val_accuracy_vector = np.zeros((len(dense_layers)*epochs))
    trainable_params_vector = np.zeros((len(dense_layers)*epochs))
    non_trainable_params_vector = np.zeros((len(dense_layers)*epochs))
    training_time = 0
    i = 0                          
#    gradient_vector = np.zeros((len(dense_layers)))
    for layer_to_freeze in dense_layers:
        for _ in range(epochs):       
            start_time = time.time()
            print(f"\nNow starting to freeze {layer_to_freeze}")
            if model_type == "SimpleDNN":
                model.layers[layer_to_freeze].trainable = True
            else:
                model.layers[0].layers[layer_to_freeze].trainable = True
            gradient_change = 0
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                
                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits = model(x_batch_train, training=True)  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value = loss_fn(y_batch_train, logits)
                    accuracy_value = acc_fn(np.argmax(y_batch_train, axis=1), np.argmax(logits, axis=1))

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                loss_vector[i, step] = loss_value
                accuracy_vector[i, step] = accuracy_value
                gradient_change += np.sum(np.abs(grads[0])) + np.sum(np.abs(grads[1]))
                # Log every 200 batches.
                if step % 20 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * batch_size))
        #    gradient_vector[layer_to_freeze-min(dense_layers)] = gradient_change/step
            end_time = time.time()
            training_time += end_time - start_time
            trainable_params_vector[i] = trainableParams =np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
            non_trainable_params_vector[i] = nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
            # predictions = model(val_dataset[0], training=False)
            predictions = model.predict(val_dataset[0], batch_size = batch_size)
            val_loss = loss_fn(val_dataset[1], predictions)
            val_accuracy = acc_fn(np.argmax(val_dataset[1], axis=1), np.argmax(predictions, axis=1))
            val_loss_vector[i] = val_loss
            val_accuracy_vector[i] = val_accuracy
            if model_type == "SimpleDNN":
                model.layers[layer_to_freeze].trainable = False
            else:
                model.layers[0].layers[layer_to_freeze].trainable = False
            
            print(f"""
                    Epoch: {i+1},
                    Loss: {loss_vector[i].mean()},
                    Val loss: {val_loss_vector[i].mean()},
                    Accuracy: {accuracy_vector[i,:].mean()},
                    Val accuracy: {val_accuracy_vector[i].mean()}
                    """)
            i+=1
            
    print(f"""
          Training time: {training_time}
          Mean training epoch time: {(training_time)/len(dense_layers)}""")
    total_params = trainableParams + nonTrainableParams
    history = {
        "accuracy" : [np.mean(accuracy_vector, axis=1)],
        "val_accuracy" : [val_accuracy_vector],
        "loss" : [np.mean(loss_vector, axis=1)],
        "val_loss" : [val_loss_vector],
#        "gradient" : [gradient_vector],
        "training_time" : [training_time],
        "training_time_epoch" : [(training_time)/len(dense_layers)],
        "batch_size" : [batch_size],
        "trainable_params" : [trainable_params_vector],
        "non_trainable_params" : [non_trainable_params_vector],
        "total_params_vector" : [total_params],
    }
    return model, history