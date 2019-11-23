# pip install opencv-python
# pip install numpy
import cv2
import numpy as np

write_to_file = False  # if this is true, it writes any frames detected with a marker to an avi for training
# filename
name = "out"  # .avi
# definitely not code copied from github that shows the webcam in cv2
cam = cv2.VideoCapture(0)
mirror = False


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


def create_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 15 ** 2
    # params.maxArea = roi_size[0] * roi_size[1]
    params.maxArea = 500 ** 2
    params.filterByCircularity = False
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False
    # blob detection only works with "uint8" images.
    # params.minThreshold = int(blob_min_int * 255)
    # params.maxThreshold = int(blob_max_int * 255)
    params.thresholdStep = 10
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        return cv2.SimpleBlobDetector(params)
    else:
        return cv2.SimpleBlobDetector_create(params)


hue = 0
hue *= 2
hue_tolerance = 5  # hue plus or minus this value (hue is 0-180)
# lower = np.array([hue - hue_tolerance, 128, 128])
# upper = np.array([hue + hue_tolerance, 256, 256])
lower1 = np.array([0, 128, 128])
upper1 = np.array([5, 256, 256])
lower2 = np.array([175, 128, 128])
upper2 = np.array([180, 256, 256])

name += '.avi'
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
if write_to_file:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(name, fourcc, 60, (100, 100))

detector = create_blob_detector()
kernel = np.ones((3, 3), np.uint8)
while True:
    ret_val, img = cam.read()
    if mirror:
        img = cv2.flip(img, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(img, lower, upper)
    mask1 = cv2.inRange(img, lower1, upper1)
    mask2 = cv2.inRange(img, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    invmask = (255 - mask)
    # mask = cv2.dilate(mask, kernel, iterations=3)
    res = cv2.bitwise_and(img, img, mask=mask)

    nored = cv2.bitwise_and(img, img, mask=invmask)

    keypoints = detector.detect(mask)

    if len(keypoints) > 0:
        keyp = keypoints[0]
        for keypoint in keypoints:
            if keypoint.size > keyp.size:
                keyp = keypoint

        x1 = max(0, round(keyp.pt[0] - keyp.size / 2))
        x2 = max(0, round(keyp.pt[0] + keyp.size / 2))
        y1 = max(0, round(keyp.pt[1] - keyp.size / 2))
        y2 = max(0, round(keyp.pt[1] + keyp.size / 2))
        crop_img = img.copy()[y1:y2, x1:x2]
        nored = nored.copy()[y1:y2, x1:x2]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        height, width, channels = crop_img.shape
        crop_imghsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        hues = create_blank(height, width)
        printline = ""
        state = "nocode"
        awaitingstate = "nocode"
        timeinstate = 0
        timewaiting = 0
        for pixel in range(height):
            pixelraw = crop_imghsv[pixel][width // 2]
            hue = pixelraw[0] * 2
            if pixelraw[1] > 50 and pixelraw[2] > 50:
                linehue = (int(hue / 2), 255, 255)
                if awaitingstate == "white":
                    awaitingstate = "nocode"
                    timewaiting = 0
                if 30 < hue < 70:

                    if state == "nocode" and awaitingstate == "nocode":
                        awaitingstate = "yellow"
                        timewaiting = 0
                    if awaitingstate == "yellow" and timewaiting > 2:
                        state = "yellow"
                        timeinstate = 0
                        timewaiting = 0
                        printline += "code: "
                # if 230 < hue < 260:

            # print(hue)
            else:
                linehue = (0, 0, 255)

                if awaitingstate is not "white":
                    awaitingstate = "white"
                    timewaiting = 0
                if timewaiting > 5:
                    state = "white"
            # linehue = (int(crop_imghsv[pixel][width // 2][0] * 2), 100, 100)
            # linehue = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # linehue = colorsys.rgb_to_hsv(linehue, 50, 50)
            # linehue = (linehue, 50, 50)
            timeinstate += 1
            timewaiting += 1
            cv2.rectangle(hues, (0, int(pixel)), (int(width), int(pixel)), linehue, 1)
        print(printline)
        hues = cv2.cvtColor(hues, cv2.COLOR_HSV2BGR)
        # crop_img = res[1:100, 1:100]
    else:
        crop_img = create_blank(100, 100)
        nored = create_blank(100, 100)
        hues = create_blank(100, 100)
    img = cv2.drawKeypoints(img, keypoints, np.array([]), (255, 255, 0),
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    BLUE = [255, 255, 255]
    crop_img = cv2.resize(crop_img, (100, 100), interpolation=cv2.INTER_AREA)
    if write_to_file and len(keypoints) > 0:
        # frame = cv2.flip(crop_img, 0)
        out.write(crop_img)
    nored = cv2.resize(nored, (100, 100), interpolation=cv2.INTER_AREA)
    # hues = cv2.resize(hues, (100, 100), interpolation=cv2.INTER_AREA)
    '''colorcode = nored.mean(axis=0).mean(axis=0)
    hsvcoder = colorsys.rgb_to_hsv(colorcode[2] / 255, colorcode[1] / 255, colorcode[0] / 255)
    hsvcode = [round(hsvcoder[0] * 360), round(hsvcoder[1] * 100), round(hsvcoder[2] * 100)]
    # print(hsvcode)
    if hsvcode[1] > 30 and hsvcode[2] > 30:
        print(hsvcode[0])
    else:
        print("no code detected")
    colorimg = create_blank(200, 200, colorcode)
    colorimg = cv2.cvtColor(colorimg, cv2.COLOR_RGB2BGR)
    cv2.imshow("colorimg", colorimg)
    '''
    cv2.imshow("cropped", crop_img)
    cv2.imshow('frame', img)
    # cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow("nored", nored)
    # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    cv2.imshow("ohno", hues)
    # cv2.imshow('blobs', im_with_keypoints)
    # cv2.imshow('blob', im_with_keypoints)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
cam.release()
if write_to_file:
    out.release()
