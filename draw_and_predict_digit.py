# Import necessary libraries
from keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf
import random as rand
import numpy as np
import os

# Create an output folder
os.makedirs("outputs", exist_ok=True)

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Amount of training data: " + str(x_train.shape[0]))
print("Amount of testing data: " + str(x_test.shape[0]))

# Normalize the data
x_train = x_train / 255
x_test = x_test / 255

# Normalize the data
x_train = x_train / 255
x_test = x_test / 255

# Build the neural network
neural_net = tf.keras.Sequential() # WE have set up a nurel network and now we are telling it we want multi layers
neural_net.add(tf.keras.layers.Flatten(input_shape=(28,28))) # We now create the NN, Flatten gives a 2d array and makes it 1d array.
neural_net.add(tf.keras.layers.Dense(128, activation= "relu")) # These next two layers are the hidden layers that are the thinking layers
neural_net.add(tf.keras.layers.Dense(64, activation= "relu"))
neural_net.add(tf.keras.layers.Dense(10, activation= "softmax")) # Output layer
neural_net.summary()
neural_net.compile('adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Convert labels to categorical one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes = 10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes = 10)
model_history = neural_net.fit(x_train, y_train, validation_split= 0.2 , batch_size = 32, epochs = 10)

# ----------------------------------------------------------------------------------
# Save accuracy graph
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Val Accuracy & Training Accuracy')
plt.xlabel('epochs(iterations)')
plt.ylabel('accuracy')
plt.legend(['Train', 'Validation'])
plt.savefig("outputs/accuracy_plot.png")
plt.close()

# ----------------------------------------------------------------------------------
# Save loss graph
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Val Loss & Training Loss')
plt.xlabel('epochs(iterations)')
plt.ylabel('loss')
plt.legend(['Train', 'Validation'])
plt.savefig("outputs/loss_plot.png")
plt.close()

# ----------------------------------------------------------------------------------
for i in range(40):
    # Make a prediction on a random test image
    randomInt = rand.randint(1,10000)
    print("Random index:", randomInt)
    input_img = x_test[randomInt]

    output_prediction = neural_net.predict(input_img.reshape(1, 28, 28))
    print("Raw output:", output_prediction)
    print("Predicted digit:", np.argmax(output_prediction))

    # Save the random test image + prediction
    plt.imshow(input_img, cmap="gray")
    plt.title(f"Predicted: {np.argmax(output_prediction)}")
    plt.axis("off")
    plt.savefig(f"outputs/random_prediction_{i}_idx{randomInt}.png")
    plt.close()