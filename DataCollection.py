import cv2
import os
import time

name = input("Enter the Name: ")

# Load the pre-trained Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open a connection to the webcam (you may need to change thevino index based on your system configuration)
cap = cv2.VideoCapture(0)

# Create a directory to save the captured images

if not os.path.exists('./eye_dataset/'+name):
    os.makedirs('./eye_dataset/'+name)

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
        img_name = f"eye_dataset/{name}/frame_{img_counter}.jpg"
        cv2.imwrite(img_name, crop_img)

        print(f"{img_name} written!")
        img_counter += 1

    if img_counter > 99:
        break
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

import splitfolders

splitfolders.ratio("eye_dataset", # The location of dataset
                   output="split_eye_dataset", # The output location
                   seed=42, # The number of seed
                   ratio=(.7, .2, .1), # The ratio of splited dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
                   )
