import cv2
import imutils
import numpy as np
import time
import os
import telegram
import asyncio
import requests
import streamlit
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream


TOKEN = '5699239343:AAEeTVVrbhU-HwQ7FVcfIkXt7imapC0ZUaE'
# CHAT_GRUP_ID = '-1001638069126'
CHAT_GRUP_ID = '-999983064'

bot = telegram.Bot(TOKEN)

# path folder save image
path = 'D:\TA-04'  # Replace your path directory
img_path ='D:\TA-04'
MAX_ENROLL = 10

# object detection
config_file = f"{path}/pretrained/mobile-net/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = f"{path}/pretrained/mobile-net/frozen_inference_graph.pb"
labels = f"{path}/pretrained/mobile-net/labels.txt"

async def sendPhotoTelegram(img_path, caption):
    asyncio.run(sendPhotoTelegram("guest-detect.jpg", caption))
    asyncio.run(sendPhotoTelegram("theft-detect.jpg", caption))

    # send to telegram
    await bot.sendPhoto(chat_id=CHAT_GRUP_ID, photo=img_path,
                        caption=caption)

    print(f"Send Photo {img_path} to telegram")

# Load the Haar cascade classifier for full body detection
bodyCascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')

# Load the guest and thief detection models from disk
guestNet = load_model("guest_detection.model")
thiefNet = load_model("thief_detection.model")

# Initialize the video stream
print("[INFO] Starting video stream...")
#vs = VideoStream(src='http://192.168.122.54/mjpeg/1').start()
vs = VideoStream(src=0).start()

# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it
    # to have a maximum width of 600 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bodies in the grayscale frame using the Haar cascade classifier
    bodies = bodyCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Loop over the detected bodies
    for (x, y, w, h) in bodies:
        # Extract the body ROI, resize it to 224x224, and preprocess it
        body_roi = frame[y:y + h, x:x + w]
        body_roi = cv2.cvtColor(body_roi, cv2.COLOR_BGR2RGB)
        body_roi = cv2.resize(body_roi, (224, 224))
        body_roi = img_to_array(body_roi)
        body_roi = preprocess_input(body_roi)

        # Expand dimensions for batch prediction
        body_roi = np.expand_dims(body_roi, axis=0)

        # Make predictions on the body for guests and thieves
        guest_pred = guestNet.predict(body_roi)
        thief_pred = thiefNet.predict(body_roi)

        # Determine the class label and color
        guest_prob = guest_pred[0][0]
        thief_prob = thief_pred[0][0]

        if guest_prob > thief_prob:
            label = "Guest"
            color = (0, 255, 0)  # Green
        else:
            label = "Thief"
            color = (0, 0, 255)  # Red

        # Include the probability in the label
        format_label = "{}: {:.2f}%".format(label, max(guest_prob, thief_prob) * 100)

        # Display the label and bounding box rectangle on the output frame
        cv2.putText(frame, format_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Save and send the image if guest or thief is detected
        async def sendPhotoTelegram(img_path, caption):
            url = f'https://api.telegram.org/bot{TOKEN}/sendPhoto'
            data = {
                'chat_id': CHAT_GRUP_ID,
                'caption': caption
            }
            with open(img_path, 'rb') as photo:
                files = {'photo': photo}
                response = requests.post(url, data=data, files=files)
            if response.status_code == 200:
                print('Foto berhasil dikirim!')
            else:
                print('Gagal mengirim foto.')

        # Mengirim foto ke grup Telegram
        if (label == "Guest"):
            cv2.imwrite("guest-detect.jpg", frame)
            caption = "Guest detected"
            print("label-guest")
            asyncio.get_event_loop().run_until_complete(sendPhotoTelegram("guest-detect.jpg", caption))
        elif (label == "Thief"):
            cv2.imwrite("thief-detect.jpg", frame)
            caption = "Thief detected"
            print("label-thief")
            asyncio.get_event_loop().run_until_complete(sendPhotoTelegram("thief-detect.jpg", caption))

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()