import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime #for attendence
# First reading the images one by one from the folder using os and then for loop
path = 'Imageattendence'
images = []
classnames = []
mylist = os.listdir(path)

for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])

print(classnames)
print(images)
# encoding all the images one by one
def imgencod(images):
    encodelist=[]
    for img in images:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encod = face_recognition.face_encodings(img)[0]
        encodelist.append(encod)
    return encodelist        
    encodelistknown = imgencod(images)
print('Encode Complete')
# Function to mark the attendence
def markattendence(name):
    with open('attendence.csv','r+') as f:
        mydatalist = f.readlines()
        print(mydatalist)
        namelist=[] # to enter the attendees names
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist: # Checking whether a name is already present or not because in that case a double attendence can't be given
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')
    
    cap = cv2.VideoCapture(0)
# Using the webcam to capture a new face for attendence
while True:
    caps, frame = cap.read()
    imgS = cv2.resize(frame, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    faceincurframe = face_recognition.face_locations(imgS)
    encodscurframe = face_recognition.face_encodings(imgS,faceincurframe)
    
    for encodface, faceloc in zip(encodscurframe, faceincurframe):
        matches = face_recognition.compare_faces(encodelistknown,encodface)
        facedis = face_recognition.face_distance(encodelistknown,encodface)
        print(facedis)
        
        matchindex = np.argmin(facedis) # Taking the index with minimum face distance
        
        # displaying the classname if a match is found
        if matches[matchindex]:
            name = classnames[matchindex].upper()
            print(name)
            # Building a rectangle around the face using face location
            y1,x2,y2,x1 = faceloc #Getting the coordinates
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            # Putting the text around the box using opencv
            cv2.putText(frame, name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)
            
            # Finally calling the function to enter the attendence of this detected person in a csv file
            markattendence(name)
        
    # showing the matched image using opencv
    cv2.imshow('webcam',frame)
    cv2.waitKey(1)
        
    