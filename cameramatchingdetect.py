import cv2
import numpy as np
import os
import glob



cam = cv2.VideoCapture(0)

train_images_dir = 'D:\\Data Mining\\Project Detector\\data'

img_data = glob.glob(os.path.join(train_images_dir, '*.jpg'))
learnKPs = []
descsLearn = []


detector = cv2.SIFT_create()  # or any other detector
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))


image_names = [os.path.basename(image_path) for image_path in img_data]

for image_path in img_data:
    learnImg = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    learnKP, descLearn = detector.detectAndCompute(learnImg, None)
    learnKPs.append(learnKP)
    descsLearn .append(descLearn)

MIN_MATCH_COUNT = 30

while True:
    ret, QueryImgBGR = cam.read()
    if not ret:
        break

    loadIMD = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
    tray, desc = detector.detectAndCompute(loadIMD, None)

    for i, trainDesc in enumerate(descsLearn ):
        check = flann.knnMatch(desc, trainDesc, k=2)
        identify = [m for m, n in check if m.distance < 0.7 * n.distance]

        if len(identify) > MIN_MATCH_COUNT:
            tt = np.float32([learnKPs[i][m.trainIdx].pt for m in identify])
            qq = np.float32([tray[m.queryIdx].pt for m in identify])
            H, status = cv2.findHomography(tt, qq, cv2.RANSAC, 3.0)
            h, w = cv2.imread(img_data[i], cv2.IMREAD_GRAYSCALE).shape
            trainBorder = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            queryBorder = cv2.perspectiveTransform(trainBorder, H)
            cv2.polylines(QueryImgBGR, [np.int32(queryBorder)], True, (0, 255, 0), 5)
            # Display the name of the picture
            cv2.putText(QueryImgBGR, f"Detected: {image_names[i]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        2, cv2.LINE_AA)
        else:
            print("Data matching - %d/%d" % (len(identify), MIN_MATCH_COUNT))

    cv2.imshow(' Matching Detection', QueryImgBGR)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
