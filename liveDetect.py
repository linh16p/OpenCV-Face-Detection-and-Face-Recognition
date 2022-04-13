import cv2

# Tạo bộ nhận diện khuôn mặt
faceCascade = cv2.CascadeClassifier("haar_face.xml")
cap = cv2.VideoCapture(0) # Truy cập vào thiết bị Camera

while True:
    # Chụp lại từng khung hình
    ret, frame = cap.read()

    # Quá trình nhận diện sẽ được thực hiện trên ảnh xám (Đen/Trắng)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Chuyển ảnh màu sang ảnh xám

    # Thực thi Face Detection
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print("Tìm thấy {0} khuôn mặt!".format(len(faces)))

    # Vẽ một hình tứ giác xung quanh các khuôn mặt phát hiện được. Vẽ trên ảnh màu.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame) # Hiển thị kết quả ra màn hình
    if cv2.waitKey(1) & 0xFF == ord('q'): # Nhấn phím q để dừng
        break

# Trước khi kết thúc chương trình, ta phải giải phóng tài nguyên camera
cap.release()
cv2.destroyAllWindows()


