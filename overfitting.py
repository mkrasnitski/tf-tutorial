import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

NUM_WORDS = 10000
EPOCHS = 10
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
	results = np.zeros((len(sequences), dimension))
	for i, word_indices in enumerate(sequences):
		results[i, word_indices] = 1.0
	return results

def plot_history(histories, key='binary_crossentropy'):
	plt.figure(figsize=(16,10))

	for name, history in histories:
		val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')
		plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')

	plt.xlabel('Epochs')
	plt.ylabel(key.replace('_',' ').title())
	plt.legend()

	plt.xlim([0,max(history.epoch)])

train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

baseline_model = keras.Sequential([
	keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
	keras.layers.Dense(16, activation=tf.nn.relu),
	keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

baseline_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
# baseline_model.summary()

# smaller_model = keras.Sequential([
# 	keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
# 	keras.layers.Dense(4, activation=tf.nn.relu),
# 	keras.layers.Dense(1, activation=tf.nn.sigmoid),
# ])
# smaller_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

l2_model = keras.models.Sequential([
	keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,),
						   kernel_regularizer=keras.regularizers.l2(0.001)),
	keras.layers.Dense(16, activation=tf.nn.relu,
						   kernel_regularizer=keras.regularizers.l2(0.001)),
	keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

l2_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

dropout_model = keras.models.Sequential([
	keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,),
						   kernel_regularizer=keras.regularizers.l2(0.001)),
	keras.layers.Dropout(0.5),
	keras.layers.Dense(16, activation=tf.nn.relu,
						   kernel_regularizer=keras.regularizers.l2(0.001)),
	keras.layers.Dropout(0.5),
	keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

dropout_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])


baseline_history = baseline_model.fit(train_data, train_labels, epochs=EPOCHS, batch_size=512, validation_data=(test_data, test_labels), verbose=1)
l2_model_history = l2_model.fit(train_data, train_labels, epochs=EPOCHS, batch_size=512, validation_data=(test_data, test_labels), verbose=1)
dropout_model_history = dropout_model.fit(train_data, train_labels, epochs=EPOCHS, batch_size=512, validation_data=(test_data, test_labels), verbose=1)

plot_history([('baseline', baseline_history), ('l2', l2_model_history), ('dropout', dropout_model_history)])
plt.show()