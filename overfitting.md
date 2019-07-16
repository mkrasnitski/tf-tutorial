#Mitigating Overfitting
##1. Weight Regularization

When defining layers for the model, each layer can be regularized via the `kernel_regularizer` argument, e.g:

	tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.001))

####- L1 Regularization - `tf.keras.regularizers.l1(coef)`
- Each weight in the network contributes to the loss function a linear term proportional to the weight's value/

####- L2 Regularization - `tf.keras.regularizers.l2(coef)`
- Each weight in the network contributes to the loss function term proportional to the square of the weight's value.

##2. Dropout - `tf.keras.layers.Dropout(frac)`
- Dropout can be added to any layer in a network by adding a Dropout layer directly after it. Dropout applies itself to the output of the previous layer, and nullifies a portion of all the connections and nodes in the previous layer, according to the fraction parameter given to it, which is usually between 0.2 and 0.5. For example:

		model = keras.models.Sequential([
			keras.layers.Dense(512, activation=keras.activations.relu, input_shape=(784,)),
			keras.layers.Dropout(0.2),	
			keras.layers.Dense(10, activation=keras.activations.softmax)
		])

	In this case, Dropout is applied to the first layer in the list. 80% of the nodes are simply piped on through, and 20% are nullified.

##3. Reduce Model Complexity
- Too many parameters can lead to significant overfitting, because the model may end up simply learning to map training examples to labels, memorizing more than learning. In general, making the model smaller requires the model to make more assumptions that may generalize better, although making the model too small will cause it to lose learning power.

##4. Early Stopping - `tf.keras.callbacks.EarlyStopping`
- When training, the `EarlyStopping` callback can be invoked whenever a chosen metric does not decrease for a given number of epochs.

		early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
		model.fit(train_data, train_labels, epochs=EPOCHS, validation_data=(test_data, test_labels), callbacks=[early_stop])

	Here, the loss function applied to the validation data is monitored, and if no improvement is seen after 10 epochs, training is stopped.

##5. Batch Normalization
- In addition to normalization of input data, the activations in each layer can be normalized so that their distribution stays consistent. Similar to Dropout, Tensorflow allows batch normalization layers to be placed in the network that normalize the previous layer and pipe the normalized activations through.

