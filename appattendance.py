import face_recognition
from flask import Flask,render_template,Response
import cv2
import os
import numpy as np
from datetime import datetime
app=Flask(__name__)
camera =cv2.VideoCapture(0)

path = 'abc'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'n{name},{dtString}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')


def gen_img():
    while True:
        success,img = camera.read()
        if not success:
            break
        else:
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                #print(faceDis)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    print(name)
                    y1,x2,y2,x1 = faceLoc
                    y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
                    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
                    #markAttendance(name)
            ret,buffer = cv2.imdecode('.jpegimg')
            img=buffer.tobytes()
            yield(b'--img\r\n'
                   b'Content-type:image/jpeg\r\n\r\n'+ img + b'\r\n')
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_img(), mimetype='multipart/x-mixed-replace; boundary=img')
if __name__=='__main__':
    app.run(debug=True)
