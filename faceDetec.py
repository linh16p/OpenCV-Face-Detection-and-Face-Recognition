import cv2 as cv
import numpy as np

#img = cv.imread('ImgTest/lena.png')
img = cv.imread('ImgTest/TeamNasa.jpg')
#img = cv.imread('ImgTest/2peo.png')
#img = cv.imread('ImgTest/ZoomFace.png')
#img = cv.imread('ImgTest/60TH5.jpg')
cv.imshow('demo image',img)

#60th5
# crop = img[310:600,100:980]
# cv.imshow('cop',crop)

#resize
# img = cv.resize(crop,(int(1400*1.5),int(720*1.5)),interpolation=cv.INTER_LINEAR)
# cv.imshow('res',img)

#team NASA
img = cv.resize(img,(1300,1000),interpolation=cv.INTER_LINEAR)
cv.imshow('res',img)

# bilateral
# bilateral = cv.bilateralFilter(img, 10, 15, 20)
# cv.imshow('bilateral', bilateral)

# kernel = np.array([[0, -1, 0],
#                    [-1, 5,-1],
#                    [0, -1, 0]])

kernel = np.array([[-1, -1, -1],
                   [-1, 9,-1],
                   [-1, -1, -1]])

image_sharp = cv.filter2D(src=img, ddepth=-1, kernel=kernel)

# Quá trình nhận diện sẽ được thực hiện trên ảnh xám (Đen/Trắng)
gray = cv.cvtColor(image_sharp,cv.COLOR_BGR2GRAY) # Chuyển ảnh màu sang ảnh xám
cv.imshow('gray face',gray)

# ve co ban van di tim bien cua phan khuon mat
# truoc tien la phan phai co 1 buc hinh xam

#doc file haar_face, noi chua tai nguyen anh 
# Tạo bộ nhận diện khuôn mặt
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Thực thi Face Detection
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 14,minSize=(30, 30))

#60Th5
#faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3,minSize=(2, 2))
# tra ra 1 list các tọa độ hình chữ nhật của khuôn mặt
print(f'Number of face found = {len(faces_rect)}')

# Vẽ một hình tứ giác xung quanh những khuôn mặt phát hiện được. Vẽ trên ảnh màu.
for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('Detective faces',img)
cv.waitKey(0)