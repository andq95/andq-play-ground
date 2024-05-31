import tensorflow as tf

print("TensorFlow version:", tf.__version__)


# Load MNIST DATA SET
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Data Shape")
print(x_train.shape)

print("Label shape")
print(y_train.shape)


# Define Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])


predictions = model(x_train[:1]).numpy()


print("Prediction:")
print(predictions.shape)
print(predictions)
print(tf.nn.softmax(predictions).numpy())


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print(y_train[:1])
print("Loss value for this prediction")
print(loss_fn(y_train[:1], predictions).numpy())


# Compile Model

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Train Model

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
