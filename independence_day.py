import datetime
import csv
import json

from gtts import gTTS

import pandas as pd
import face_recognition
import cv2
import pygame
import requests

#Mahatma Gandhi tts
#using the wikipedia API to retrieve Mahatma Gandhi's wikipedia page
r = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/Gandhi")
page = r.json()
gandhi_desc = page["extract"]

#Imaad tts
#I don't have a wikipedia page, so I have to write my message here
imaad_tts ='इमाद खान उन सभी को सलाम करता है जिन्होंने भारत की आजादी के लिए अपना जीवन निर्धारित किया।'

def get_tts(tts_text, language):
    tts = gTTS(text=tts_text, lang=language)
    filename = '/tmp/temp.mp3'
    tts.save(filename)
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play() 


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

gandhi_image = face_recognition.load_image_file("gandhi.jpeg")
gandhi_face_encoding = face_recognition.face_encodings(gandhi_image)[0]
gandhi_flag = 0
imaad_image = face_recognition.load_image_file("imaad.jpg")
imaad_face_encoding = face_recognition.face_encodings(imaad_image)[0]
imaad_flag = 0


# Create arrays of known face encodings and their names
known_face_encodings = [
    gandhi_face_encoding,imaad_face_encoding
   ]
known_face_names = [
    "Mahatma Gandhi", "Imaad Khan"
   ]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
name_of_face = {}



while True:

    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,0.5)
            name = "Not recognised"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)
          
    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    for name in face_names:
        if imaad_flag == 1:
            break
        if name == 'Imaad Khan': 
            get_tts(imaad_tts, 'hi')
            imaad_flag = 1
        else:
            break

    for name in face_names:
        if gandhi_flag == 1:
            break
        if name == 'Mahatma Gandhi':
            get_tts(gandhi_desc, 'en')
            gandhi_flag = 1
        else:
            break
        

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()


print(name_of_face) 
