import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    return cap

def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        exit()
    return frame

def remove_noise(frame, kernel_size):
    return cv2.morphologyEx(frame, cv2.MORPH_OPEN, np.ones((kernel_size,kernel_size), np.uint8))

def nothing(x):
    pass

img_cnt = 0

if __name__ == "__main__":

    # Create camera
    cap = get_camera()

    # Create a window
    cv2.namedWindow('image')

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    # Set default values for HSV trackbars
    # (0, 30, 136), (56, 255, 255)
    cv2.setTrackbarPos('HMin', 'image', 0)
    cv2.setTrackbarPos('SMin', 'image', 30)
    cv2.setTrackbarPos('VMin', 'image', 136)
    cv2.setTrackbarPos('HMax', 'image', 56)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize HSV min/max values
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    pause = False

    while True:

        # Capture frame-by-frame
        if not pause:
            image = get_frame(cap)

        # while(1):
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display result image
        cv2.imshow('image', result)
        key = cv2.waitKey(1)
        if key == ord('q'): # quit
            cap.release()
            cv2.destroyAllWindows()
            break
        if key == ord('p'): # pause
            pause = True

        # cv2.destroyAllWindows()

        # key = cv2.waitKey(1)
        # if key == ord('q'): # quit
        #     cap.release()
        #     cv2.destroyAllWindows()
        #     break


def nothing(x):
    pass

# Load image
image = cv2.imread('1.jpg')