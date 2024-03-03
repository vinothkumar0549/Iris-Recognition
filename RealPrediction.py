from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
import time
import shutil

# Load the pre-trained Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open a connection to the webcam (you may need to change thevino index based on your system configuration)
cap = cv2.VideoCapture(0)

# Create a directory to save the captured images

if os.path.exists('./eye'):
    shutil.rmtree('./eye')

if not os.path.exists('./eye'):
    os.mkdir('./eye')

# Counter for image filenames
img_counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected eyes and save the frames
    for (x, y, w, h) in eyes:
        crop_img = gray[y:y+h, x:x+w]

        crop_img = cv2.resize(crop_img, (224,224))

        #cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Save the frame containing detected eyes
        img_name = f"eye/frame_{img_counter}.jpg"
        cv2.imwrite(img_name, crop_img)

        print(f"{img_name} written!")
        img_counter += 1

    if img_counter > 9:
        break
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Load the trained model
model = load_model('trained_model_5_classes.h5')

i=9
classes = []

while i:
    img_path = f"./eye/frame_{i}.jpg"
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # rescale the pixel values to the range [0, 1]

    # Make a prediction
    prediction = model.predict(img_array)

    # Get the predicted class label
    predicted_class = np.argmax(prediction)
    i=i-1
    classes.append(predicted_class)
    

# Print the predicted class label
print(classes)
#print("Predicted class:", predicted_class)

