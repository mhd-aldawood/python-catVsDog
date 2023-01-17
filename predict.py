import tensorflow as tf

from tensorflow import keras

model = tf.keras.models.load_model('model.h5') 

img = keras.preprocessing.image.load_img(
    "PetImages/Cat/6779.jpg", target_size=(200,200)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(predictions[0])

if predictions[0][0] > 0.5:
    print(" is a dog")
else:
    print(" is a cat")
