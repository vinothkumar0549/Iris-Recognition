from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Provide the path to your HDF5 model file
model_path = 'trained_model_5_classes.h5'

# Load the model
model = load_model(model_path)

# Define the directories for train, test, and validate
train_directory = 'split_eye_dataset/train/'
test_directory = 'split_eye_dataset/test/'
validate_directory = 'split_eye_dataset/val/'
# Define parameters for image preprocessing and augmentation
batch_size = 32
image_size = (224, 224)

# Use ImageDataGenerator to load and preprocess images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
validate_datagen = ImageDataGenerator(rescale=1./255)

# Load training and testing and validating datasets


test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'  # Use 'categorical' for multi-class classification
)


# Optionally, you can also load and preprocess train and validation dataset similarly

# Ensure that the number of classes matches the shape of the labels

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)
print('Test Loss:', test_loss)

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate predictions for the test set
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_labels = test_generator.classes

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_classes)
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

