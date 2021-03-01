import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras


model_number = 50
image_path = input("Input Image Path: ")

from PIL import Image
Image.open(image_path).show()
# image_path = "PetImages/Cat/6779.jpg"

image_size = (100, 100)
model = keras.models.load_model(f'save_at_{model_number}.h5')

img = keras.preprocessing.image.load_img(
    image_path, target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]
print(
    "This image is %.5f%% cat and %.5f%% dog."
    % (100 * (1 - score), 100 * score)
)

if (1-score) > score:
    print("This image is a Cat!")
elif (1-score) < score:
    print("This image is a Dog!")
else:
    print("I don't know")