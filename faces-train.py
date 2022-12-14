import os
import numpy as np
from PIL import Image
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "photos")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_train = []
x_train = []

# creates tuple of path, folders and files
for root, dirs, files in os.walk(image_dir):
    # we only loop through the files found
    for file in files:
        # check if current file is either a png or jpg image
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            # gets the folder name where the image is in
            # handles incorrectly labeled directory names
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()

            # if the label is not in our directory of labels, we add it
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            # set id_ as the label name to be matched to the current image to avoid mismatch
            id_ = label_ids[label]
            gray_image = Image.open(path).convert("L")  # convert to grayscale
            image_array = np.array(gray_image, "uint8")  # create an image array from the grayscale image
            # see if a face can be detected from the image_array
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=6)
            # gets the region of the face on the image and appends it to the two arrays
            for (x, y, w, h) in faces:
                region_of_interest = image_array[y:y + h, x:x + w]
                x_train.append(region_of_interest)
                y_train.append(id_)

# writes labels to labels.pickle, so it can be accessed in the livestreaming.py file
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

# trains the recognizer with the faces found in the images and assigns the labels associated with those names
recognizer.train(x_train, np.array(y_train))
# saves file to trainer.yml
recognizer.save("trainer.yml")
