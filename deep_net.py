import mlp_utils
import c3d_utils


## Put sparse features into simple neural networks ##

model_mlp = mlp_utils.get_model(summary=True)
model_mlp.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Generate dummy data
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)



## Put original video into C3D networks ##
model_c3d = c3d_utils.get_model(summary=True)



## Evaluation ##