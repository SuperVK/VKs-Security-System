import face_recognition
import cv2
# Python 3 server example
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import datetime
from configparser import ConfigParser
import requests
import os

#conf file needs a sections, but I don't want
conf_file = ConfigParser()
conf_file.read('./config.conf')
config = conf_file['Config']

video_capture = cv2.VideoCapture(0)

hostName = "localhost"
serverPort = 8073

files = os.listdir(config['models'])

known_face_encodings = []
known_face_names = []

for file in files:
    image = face_recognition.load_image_file("models/" + file)
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) == 0:
        print(f'Found no person in {file}')
        continue
    if len(face_encodings) >= 2:
        print(f'Found multiple people in {file}, only took the first one!')
    known_face_encodings.append(face_encodings[0])
    known_face_names.append(file.split('.')[0])

print(known_face_names)

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        face_found = False
    
        unknown_encoding = None

        while not face_found:
            time.sleep(0.5)
            # Grab a single frame of video
            ret, frame = video_capture.read()
            if not ret:
                print('It seems like your camera is not connected correctly! Connect it, and restart.')
                return              
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = frame[:, :, ::-1]

            # Find all the faces and face enqcodings in the frame of video
            face_locations = face_recognition.face_locations(rgb_frame)
            if len(face_locations) > 0:
                unknown_encoding = face_recognition.face_encodings(rgb_frame)[0]
                face_found = True

        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
        if config['picture_logs'] != 'None':
            cv2.imwrite(config['picture_logs'] + st + '.jpeg', frame)

        results = face_recognition.compare_faces(known_face_encodings, unknown_encoding)

        found_person = "Unkown person"

        for i, isPerson in enumerate(results):
            if isPerson:
                found_person = known_face_names[i]

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes(found_person, "utf-8"))
        if config['webhook_link'] != 'None':
            requests.post(config['webhook_link'], data={"content": "{0} has been spotted!".format(found_person)})
        




       
webServer = HTTPServer((hostName, serverPort), MyServer)

print("Server started http://%s:%s" % (hostName, serverPort))
try:    
    webServer.serve_forever()
    
except KeyboardInterrupt:
    pass

webServer.server_close()
print("Server stopped.")



