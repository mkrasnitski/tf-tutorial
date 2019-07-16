import tensorflow as tf
from tensorflow import keras

import numpy as np

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
word_index = {k:v+3 for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNKNOWN>"] = 2
word_index["<UNUSED>"] = 3

word_lookup = {v:k for k, v in word_index.items()}

def decode_review(text):
	return ' '.join(word_lookup[index] for index in text)

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'], padding='post', maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'], padding='post', maxlen=256)

vocab_size = 10000

model = keras.Sequential([
	keras.layers.Embedding(vocab_size, 16),
	keras.layers.GlobalAveragePooling1D(),
	keras.layers.Dense(16, activation=tf.nn.relu),
	keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
results = model.evaluate(test_data, test_labels)
print(model.predict(test_data[0]), test_labels[0])