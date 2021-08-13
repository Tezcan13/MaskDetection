import numpy as np
import imutils
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

#For detecting the faces
proto_txt_path = 'deploy.prototxt'
model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
face_detector = cv2.dnn.readNet(proto_txt_path, model_path)

mask_detector = load_model('mask.model') #My trained model

cap = VideoStream(src=0).start() #Start stream

while True:
    frame = cap.read() #Read the frame
    frame = imutils.resize(frame, width=720) #Set the width as 720
    (h, w) = frame.shape[:2] #We have height and width
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123)) #Creates 4-dimensional blob from image (https://docs.opencv.org/master/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7)

    #Next 2 lines help us to detect where are the all faces on camera
    face_detector.setInput(blob)
    detections = face_detector.forward()

    facelist = [] #It will keep the list of all faces detected
    rect = [] #We will draw rectangles to the faces and this will keep the list of all rectangles
    results = [] #This is for the result whether with mask or without mask

    for i in range(0, detections.shape[2]): #Loop over detections
        probability = detections[0, 0, i, 2] #Probability associated with detection

        if probability > 0.5: #No need to check weak probabilities
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) #Calculate the coordinates of the rectangles
            (startX, startY, endX, endY) = box.astype("int")

            #Take the face, convert it from BGR to RGB, resize it, convert to array
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            #face = np.expand_dims(face, axis=0) #Check line 52

            facelist.append(face) #Add the faces to facelist
            rect.append((startX, startY, endX, endY)) #Add the boxes to rect list

    #Check whether the model determines if the face has mask or not
    if len(facelist) > 0:
        facelist = np.array(facelist) #If you want to read just one face, delete this line and use line 45th
        results = mask_detector.predict(facelist)

    for (face_box, result) in zip(rect, results):
        (startX, startY, endX, endY) = face_box
        (mask, unmask) = result

        label = "Mask" if mask > unmask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255) #BGR
        label = "{}: {:.2f}%".format(label, max(mask,unmask) * 100)

        #Display the label and rectangle on the output frame
        cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break