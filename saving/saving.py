import os

import tensorflow as tf
from tensorflow import keras

NUM_IMAGES = 10000

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# train_labels = train_labels[:NUM_IMAGES]
# test_labels = test_labels[:NUM_IMAGES]

# train_images = train_images[:NUM_IMAGES].reshape(-1, 28*28) / 255.0
# test_images = test_images[:NUM_IMAGES].reshape(-1, 28*28) / 255.0
train_images = train_images.reshape(-1, 28*28) / 255.0
test_images = test_images.reshape(-1, 28*28) / 255.0

def create_model():
	model = keras.models.Sequential([
		keras.layers.Dense(512, activation=keras.activations.relu, input_shape=(784,)),
		keras.layers.Dropout(rate=0.2),
		keras.layers.Dense(10, activation=keras.activations.softmax)
	])

	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return model

checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cpt_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=0)

model = create_model()
# model.summary()

model.fit(train_images, train_labels, epochs=10, validation_data = (test_images, test_labels), callbacks=[cpt_callback], verbose=1)
