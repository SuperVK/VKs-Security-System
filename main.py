import face_recognition
import cv2
import time
import datetime
from configparser import ConfigParser
import requests
import os
import sys

#conf file needs a sections, but I don't want
conf_file = ConfigParser()
conf_file.read('./config.conf')
config = conf_file['Config']

video_capture = cv2.VideoCapture(0)


files = os.listdir(config['models'])

known_face_encodings = []
known_face_names = []

for file in files:
    image = face_recognition.load_image_file("models/" + file)
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) == 0:
        print('Found no person in {0}'.format(file))
        continue
    if len(face_encodings) >= 2:
        print('Found multiple people in {0}, only took the first one!'.format(file))
    known_face_encodings.append(face_encodings[0])
    known_face_names.append(file.split('.')[0])


try:
    while True:
        #30 frames a second
        print(1/int(config['frame_rate']))
        time.sleep(1/int(config['frame_rate']))

        unknown_encoding = None
        ret, frame = video_capture.read()
        if not ret:
            print('It seems like your camera is not connected correctly! Connect it, and restart.')
            sys.exit()
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) > 0:
            unknown_encoding = face_recognition.face_encodings(rgb_frame)[0]
        else:
            continue


        results = face_recognition.compare_faces(known_face_encodings, unknown_encoding)

        found_person = "Unkown person"

        for i, isPerson in enumerate(results):
            if isPerson:
                found_person = known_face_names[i]

        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
        if config['picture_logs'] != 'None':
            cv2.imwrite(config['picture_logs'] + st + '_' + found_person +'.jpeg', frame)

        if config['webhook_link'] != 'None':
            requests.post(config['webhook_link'], data={"content": "{0} has been spotted!".format(found_person)})
        
        time.sleep(int(config['timeout']))
except KeyboardInterrupt:
    print('Script stopped')
    video_capture.release()



        




