"""
Basic Image Classification using the MNIST Database of Digits, or Google's modified Fashion MNIST Database, which contains images of clothing.
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import sys

def plot_image(i, predictions_array, true_label, img):
	predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(img, cmap=plt.cm.binary)

	predicted_label = np.argmax(predictions_array)
	color = 'blue' if predicted_label == true_label else 'red'
	plt.xlabel(f'{predicted_label} at {100*np.max(predictions_array):2.0f}% ({true_label})')

def plot_value_array(i, predictions_array, true_label):
	predictions_array, true_label = predictions_array[i], true_label[i]
	plt.grid(False)
	plt.xticks(range(10), range(10))
	plt.yticks([])
	barplot = plt.bar(range(10), predictions_array, color='#777777')
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_array)

	barplot[predicted_label].set_color('red')
	barplot[true_label].set_color('blue')


assert(len(sys.argv) == 3 and sys.argv[1] == '--epochs')


mnist = keras.datasets.mnist
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Greyscale images with pixel values from 0-255, so normalize the np arrays to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# - A Sequential NN is straightforward. 
# - 3 Layers: 1 Input (784 Nodes for 28x28 images), 1 Hidden (128 Nodes), 1 Output (10 Nodes, 0-9)
# - Input layer does not use an activation function, Hidden layer uses relu, 
#   and Output uses softmax to transform raw activations into probabilities
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10, activation=tf.nn.softmax)
])

print('Compiling and fitting')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=int(sys.argv[2]), verbose=1)

print('\nEvaluating on the test set')
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Testing Accuracy: {test_acc}')

predictions = model.predict(test_images)

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(3*num_cols, 1.5*num_rows))
for i in range(num_images):
	plt.subplot(num_rows, 2*num_cols, 2*i+1)
	plot_image(i, predictions, test_labels, test_images)
	plt.subplot(num_rows, 2*num_cols, 2*i+2)
	plot_value_array(i, predictions, test_labels)
plt.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0, 0.4)
plt.show()