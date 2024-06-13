import sys
import os
import time

import numpy as np
import pandas as pd
import sklearn

import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split

# import layers and callbacks we may use (may not be a complete list)
from keras.layers import Input, Flatten, BatchNormalization, Dense, Conv2D, MaxPooling2D, Dropout, Embedding, MultiHeadAttention, GlobalAveragePooling1D, Reshape, SimpleRNN
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

from tensorflow.python.keras.utils import layer_utils
from PIL import Image


from matplotlib import pyplot as plt
from tensorflow.keras.initializers import lecun_uniform

plt.rcParams.update({'font.size': 14})

# Let's check our software versions
print('------------')
print('### Python version: ' + __import__('sys').version)
print('### NumPy version: ' + np.__version__)
print('### Scikit-learn version: ' + sklearn.__version__)
print('### Tensorflow version: ' + tf.__version__)
print('------------')

def var_exists(var_name):
   return (var_name in globals() or var_name in locals())


path_to_data = '/blue/ruogu.fang/rohanshah1/ml/images_processed'
seed = 42
np.random.seed(seed)

def load_and_vectorize_images(image_base_path, genres, avg_dimensions=(336, 218)):
   images = []
   labels = []

   for idx, genre in enumerate(genres):
       genre_path = os.path.join(image_base_path, genre)
       for image_file in os.listdir(genre_path):
           if image_file.endswith('.png'):
               image_path = os.path.join(genre_path, image_file)
               img = Image.open(image_path).convert('L')  # Convert to grayscale
               img_resized = img.resize(avg_dimensions)
               img_vector = np.array(img_resized, dtype=np.float32).reshape(avg_dimensions[0], avg_dimensions[1], 1)  # Add channel dimension
               images.append(img_vector)
               labels.append(idx)  # Genre index

   return np.stack(images, axis=0), np.array(labels)

# Assuming 'path_to_data' is defined and points to the base directory

genres = ["rock", "reggae", "pop", "metal", "jazz", "hiphop", "disco", "country", "classical", "blues"]

# Load and vectorize the images
# images, labels = load_and_vectorize_images(path_to_data, genres)
images, labels = load_and_vectorize_images(path_to_data, genres, (56, 218))

print(f"Loaded {images.shape[0]} images.")
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Permute, GRU
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

# def compile_cnn_model(input_shape, num_classes):
#     model = Sequential([
#         Input(shape=input_shape, name='input_layer'),

#         # First Conv Block
#         Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'),
#         Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
#         MaxPooling2D((2, 2), name='pool1'),
#         BatchNormalization(name='bn1'),

#         # Second Conv Block
#         Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
#         Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
#         MaxPooling2D((2, 2), name='pool2'),
#         BatchNormalization(name='bn2'),

#         # Third Conv Block
#         Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
#         Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
#         MaxPooling2D((2, 2), name='pool3'),
#         BatchNormalization(name='bn3'),
#         Dropout(0.3, name='dropout1'),

#         # Reshape and RNN
#         Permute((2, 1, 3)),  # Permute dimensions for RNN
#         Reshape((-1, 128)),  # Reshape for RNN
#         SimpleRNN(128, return_sequences=True, name='rnn1'),
#         Dropout(0.3, name='dropout2'),
#         SimpleRNN(64, return_sequences=False, name='rnn2'),
       
#         # Output layer
#         Dense(num_classes, activation='softmax', name='output')
#     ])

#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     model.summary()
#     return model

# def compile_cnn_model(input_shape, num_classes):
#     model = Sequential([
#         Input(shape=input_shape, name='input_layer'),
       
#         # Reshape to treat the spectrogram as sequences of columns (flattening along x-axis)
#         Permute((2, 1, 3)),  # Swap the time (x) and frequency (y) axes
#         Reshape((input_shape[1], input_shape[0] * input_shape[2])),  # Flatten the frequency and channel dimensions into features

#         # GRU layers
#         GRU(128, return_sequences=True, name='gru1'),
#         GRU(64, return_sequences=True, name='gru2'),
       
#         # Flattening and final classification
#         Flatten(),
#         Dense(64, activation='relu', name='dense'),
#         Dense(num_classes, activation='softmax', name='output')
#     ])

#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     model.summary()
#     return model

def compile_cnn_model(input_shape, num_outputs):
    model = Sequential(name='CNN-GRU-Model')

    # Convolutional layers
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', 
                     kernel_initializer=lecun_uniform(), kernel_regularizer=l2(0.001)))
    # model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', 
    #                  kernel_initializer=lecun_uniform(), kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', 
                     kernel_initializer=lecun_uniform(), kernel_regularizer=l2(0.001)))
    # model.add(Conv2D(64, (3, 3), strides=(1, 1), padfding='same', activation='relu', 
    #                  kernel_initializer=lecun_uniform(), kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))

    # Prepare for recurrent layer by reshaping the feature map
    model.add(Permute((2, 1, 3)))  # Transpose the height and width dimensions
    # Calculate the correct number of features for reshaping
    model.add(Reshape((-1, 64 * input_shape[0] // 4)))  # This assumes the height and width are both halved twice

    # Recurrent layers
    model.add(GRU(128, return_sequences=True, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.7))
    model.add(GRU(96, return_sequences=False, kernel_regularizer=l2(0.001)))  # Ensures the output is (None, 96)

    # Output layer
    model.add(Dense(num_outputs, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Print model summary
    model.summary()
    return model

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

input_shape = x_train.shape[1:]
num_classes = len(np.unique(labels))

# Build the CNN model
model = compile_cnn_model(input_shape, num_classes)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')


history = model.fit(x_train, y_train,
                   validation_data=(x_val, y_val),
                   epochs=50,
                   batch_size=64,
                   callbacks=[ checkpoint])

#Plotting
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')



plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
plt.savefig("valtest.png")


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predict classes on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=genres, yticklabels=genres)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

plt.savefig("confusion.png")


from keras.models import Model

layer_outputs = [layer.output for layer in model.layers if isinstance(layer, Conv2D)]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(x_train[0:1])

# Plotting the feature maps
for activation_map in activations:
    plt.figure(figsize=(20, 20))
    for i in range(32):  # Assuming 32 filters in the Conv layer
        plt.subplot(6, 6, i+1)
        plt.imshow(activation_map[0, :, :, i], cmap='viridis')
        plt.axis('off')
    plt.show()
plt.savefig("featuremap.png")