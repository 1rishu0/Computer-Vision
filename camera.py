import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

while True:

    # Capture Frame by Frame
    ret, frame = video_capture.read()

    image_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # minSize = (100,100)/(200,200) , minNeighbors = 10
    detections = face_detector.detectMultiScale(image_gray,minSize=(200,200),minNeighbors=10)

    for(x,y,w,h) in detections:
        print(w,h)

        # Draw a Rectangle the resulting frame
        cv2.rectangle(frame,(x,y),(x+w,y+h),color = (0,255,255),thickness = 2)

    # Display the resulting frame
    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# when everything is done , release the capture
video_capture.release()
cv2.destroyAllWindows()