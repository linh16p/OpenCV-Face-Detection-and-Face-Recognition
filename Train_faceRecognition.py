import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']
DIR = r'ImgTest\train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

feartures = []
labels= []

def create_train():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            face_rect = haar_cascade.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors = 4)

            for(x,y,w,h) in face_rect:
                #vung tim thay
                faces_roi = gray[y:y+h,x:x+w]
                feartures.append(faces_roi)
                labels.append(label)

create_train()
print('Training Done ------------------------------')
# print(f'length of the features = {len(feartures)}')
# print(f'length of the label = {len(labels)}')

feartures = np.array(feartures,dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the regconizer on the feature list and the labels list 
face_recognizer.train(feartures,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy',feartures)
np.save('labels.npy',labels)