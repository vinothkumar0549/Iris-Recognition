from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('trained_model_5_classes.h5')

# Define the path to the image you want to make a prediction on
image_path = 'govardhan.jpg'

# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.  # rescale the pixel values to the range [0, 1]

# Make a prediction
prediction = model.predict(img_array)

# Get the predicted class label
predicted_class = np.argmax(prediction)

# Print the predicted class label
print("Predicted class:", predicted_class)
