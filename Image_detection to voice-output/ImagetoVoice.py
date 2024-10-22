import os
import cv2
import pyttsx3  # Importing pyttsx3 for text-to-speech

thres = 0.45  # Threshold to detect object

# Load the image
img = cv2.imread("Girl with a Pearl Earring.png")

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

classIds, confs, bbox = net.detect(img, confThreshold=thres)

if len(classIds) != 0:
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        className = classNames[classId - 1]
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

        # Adjust text position to prevent overlap with bounding box
        text_width, text_height = cv2.getTextSize(className.upper(), cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]
        cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 10, box[1] + 30 + text_height + 5),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Output class name to terminal
        print(className)

        # Output class name through speech
        engine.say(className)

        # If scissors are detected, provide additional information
        if className.lower() == 'scissors':
            additional_info = "Girl with a Pearl Earring is a masterpiece painted by the Dutch artist Johannes Vermeer around 1665. It's a captivating portrait of a young woman, often speculated to be a maid or a model, wearing a turban and a large pearl earring. The painting is renowned for its exquisite use of light and shadow, and the enigmatic expression on the girl's face has captured the imagination of viewers for centuries. The paintings simplicity and the girls mysterious gaze make it one of the most iconic works of art in history"
            engine.say(additional_info)
            print(additional_info)  # Print additional info to terminal

        engine.runAndWait()

cv2.imshow("Output", img)
cv2.waitKey(0)  # Wait indefinitely until any key is pressed
cv2.destroyAllWindows()
