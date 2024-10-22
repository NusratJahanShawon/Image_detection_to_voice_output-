import os
import cv2
import pyttsx3  # Importing pyttsx3 for text-to-speech
import time

thres = 0.45  # Threshold to detect object
detection_interval = 5  # Time interval between detections in seconds (reduced from 10 to 5)

# Use camera index 0 for the laptop's built-in camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Path to coco.names file
classFile = os.path.join(current_dir, 'coco.names')
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize pyttsx3 engine for speech output
engine = pyttsx3.init()

detected_objects = set()  # Set to keep track of detected objects
last_detection_time = time.time() - detection_interval  # Initialize the last detection time with a time that will trigger immediate detection

while True:
    current_time = time.time()  # Get the current time

    # If enough time has passed since the last detection
    if current_time - last_detection_time >= detection_interval:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                className = classNames[classId - 1].lower()
                if className not in detected_objects:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    # Define spoken sentences based on detected object
                    spoken_sentence = ""
                    if className == 'person':
                        spoken_sentence = "Humans (Homo sapiens) or modern humans are the most common and widespread species of primate, and the last surviving species of the genus Homo. They are great apes characterized by their hairlessness, bipedalism, and high intelligence. "
                    elif className == 'keyboard':
                        spoken_sentence = "A keyboard is detected. Keyboards are input devices used to type characters into computers."
                    elif className == 'cell phone':
                        spoken_sentence = "A cell phone is detected. Cell phones are portable telecommunication devices."
                    elif className == 'bicycle':
                        spoken_sentence = "A bicycle is detected. A bicycle, also called a bike, is a human-powered or motor-powered vehicle with two wheels."
                    elif className == 'car':
                        spoken_sentence = "A car is detected. A car, also known as an automobile, is a wheeled motor vehicle used for transportation."
                    elif className == 'motorcycle':
                        spoken_sentence = "A motorcycle is detected. A motorcycle, often called a motorbike, bike, or cycle, is a two or three-wheeled motor vehicle."
                    elif className == 'airplane':
                        spoken_sentence = "An airplane is detected. An airplane, also known as a plane, is a powered, fixed-wing aircraft that is propelled forward by thrust from a jet engine or propeller."

                    # Output spoken sentence to terminal
                    print("Spoken Sentence:", spoken_sentence)

                    # Output class name through speech
                    engine.say(spoken_sentence)
                    engine.runAndWait()

                    # Add detected object to the set
                    detected_objects.add(className)

                    # Update the last detection time
                    last_detection_time = time.time()

                    # You can add more information about each detected object here if needed

                    # Break the loop to ignore other detections
                    break

        cv2.imshow("Output", img)

    # Check if 'q' key is pressed, then break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
