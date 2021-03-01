import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def init_dataset(data_name, data, ans):
    data = data.astype("float32") / 255
    data = np.expand_dims(data, -1)
    print(f"{data_name} shape:", data.shape)
    print(f"{data_name} samples: ", data.shape[0])
    ans = keras.utils.to_categorical(ans)
    return data, ans


num_classes = 10
input_shape = (28, 28, 1)

(train_data, train_ans), (test_data, test_ans) = keras.datasets.mnist.load_data()

train_data, train_ans = init_dataset(data_name="train", data=train_data, ans=train_ans)
test_data, test_ans = init_dataset(data_name="test", data=test_data, ans=test_ans)

# %% Build Model

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

# %% Train
batch_size = 128
epochs = 50
callbacks = [keras.callbacks.ModelCheckpoint("checkpoint/save_at_{epoch}.h5")]
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(train_data, train_ans, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.save('model')

# %% Evaluate
score = model.evaluate(test_data, test_ans, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])