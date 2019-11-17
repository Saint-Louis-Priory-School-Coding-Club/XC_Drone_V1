# pip install opencv-python
# pip install numpy
import cv2
import numpy as np

# definitely not code copied from github that shows the webcam in cv2
cam = cv2.VideoCapture(0)
mirror = False


def create_blob_detector(roi_size=(512, 512), blob_min_area=100,
                         blob_min_int=.5, blob_max_int=.95, blob_th_step=10):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = blob_min_area
    params.maxArea = roi_size[0] * roi_size[1]
    params.filterByCircularity = False
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False
    # blob detection only works with "uint8" images.
    # params.minThreshold = int(blob_min_int * 255)
    # params.maxThreshold = int(blob_max_int * 255)
    params.thresholdStep = blob_th_step
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        return cv2.SimpleBlobDetector(params)
    else:
        return cv2.SimpleBlobDetector_create(params)


color = [255, 0, 0]
tolerance = 100
lower = np.array([color[2] - tolerance, color[1] - tolerance, color[0] - tolerance])
upper = np.array([color[2] + tolerance, color[1] + tolerance, color[0] + tolerance])
print(lower, upper)

detector = create_blob_detector()
kernel = np.ones((3, 3), np.uint8)
while True:
    ret_val, img = cam.read()
    if mirror:
        img = cv2.flip(img, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.inRange(img, lower, upper)

    mask = cv2.dilate(mask, kernel, iterations=3)
    res = cv2.bitwise_and(img, img, mask=mask)

    keypoints = detector.detect(mask)
    im_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (255, 0, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if len(keypoints) > 0:
        keyp = keypoints[0]
        for keypoint in keypoints:
            if keypoint.size > keyp.size:
                keyp = keypoint

        x1 = max(0, round(keyp.pt[0] - keyp.size / 2))
        x2 = max(0, round(keyp.pt[0] + keyp.size / 2))
        y1 = max(0, round(keyp.pt[1] - keyp.size / 2))
        y2 = max(0, round(keyp.pt[1] + keyp.size / 2))
        crop_img = res[y1:y2, x1:x2]
        # crop_img = res[1:100, 1:100]
    else:
        crop_img = res
    BLUE = [255, 255, 255]
    crop_img = cv2.copyMakeBorder(crop_img.copy(), 0, 100, 0, 100, cv2.BORDER_CONSTANT, value=BLUE)
    cv2.imshow("cropped", crop_img)
    # cv2.imshow('frame', img)
    cv2.imshow('mask', mask)
    # cv2.imshow('res', res)
    cv2.imshow('blobs', im_with_keypoints)
    # cv2.imshow('blob', im_with_keypoints)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
cam.release()
