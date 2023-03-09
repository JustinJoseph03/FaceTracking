import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("Videos/1.mp4")
prevTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id,detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            boundboxClass = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            boundBox = int(boundboxClass.xmin * iw), int(boundboxClass.ymin * ih), \
                       int(boundboxClass.width * iw), int(boundboxClass.height * ih)
            cv2.rectangle(img,boundBox, (255,0,255), 2)
            #Face detection confidence level
            cv2.putText(img, f'{int(detection.score[0]*100)}%',
                        (boundBox[0], boundBox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime
    cv2.putText(img,f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN,
                3, (0,255,0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)