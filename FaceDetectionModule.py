import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectCon = 0.5):
        self.minDetectCon = minDetectCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectCon)

    def findFaces(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        #print(self.results)
        boundBoxes = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                boundboxClass = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                boundBox = int(boundboxClass.xmin * iw), int(boundboxClass.ymin * ih), \
                           int(boundboxClass.width * iw), int(boundboxClass.height * ih)
                boundBoxes.append([id, boundBox, detection.score])
                if draw:
                    self.fancyDraw(img, boundBox)

                    #Face detection confidence level
                    cv2.putText(img, f'{int(detection.score[0]*100)}%',
                            (boundBox[0], boundBox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)
        return img, boundBoxes

    def fancyDraw(self, img, boundBox, length = 30, thickness = 5, rectThick = 1):
        x,y,w,h = boundBox
        x1, y1 = x+w, y+h

        cv2.rectangle(img, boundBox, (255, 0, 255), rectThick)

        #Top left x,y
        cv2.line(img, (x,y), (x+length, y), (255, 0, 255), thickness)
        cv2.line(img, (x, y), (x, y+length), (255, 0, 255), thickness)
        # Top right x1,y
        cv2.line(img, (x1, y), (x1 - length, y), (255, 0, 255), thickness)
        cv2.line(img, (x1, y), (x1, y + length), (255, 0, 255), thickness)
        # Bottom left x,y1
        cv2.line(img, (x, y1), (x + length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x, y1), (x, y1 - length), (255, 0, 255), thickness)
        # Bottom right x1,y1
        cv2.line(img, (x1, y1), (x1 - length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x1, y1), (x1, y1 - length), (255, 0, 255), thickness)

        return img

def main():
    cap = cv2.VideoCapture("Videos/6.mp4")
    prevTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, boundBoxes = detector.findFaces(img)
        print(boundBoxes)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
