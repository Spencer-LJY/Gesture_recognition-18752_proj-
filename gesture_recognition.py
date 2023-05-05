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

img_cnt = 0

if __name__ == "__main__":

    background = None
    cap = get_camera()

    while True:

        # Capture frame-by-frame
        frame = get_frame(cap)

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_result = cv2.inRange(hsv, (0, 15, 0), (17,170,255)) # (0, 50, 0), (17, 175, 255)

        YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        YCrCb_result = cv2.inRange(YCrCb, (0, 135, 85), (255,180,135)) # (0, 135, 85), (255, 180, 135)

        # Merge HSV and YCrCb results
        global_result = cv2.bitwise_and(YCrCb_result, YCrCb_result)
        global_result = cv2.medianBlur(global_result, 3)
        
        # finally, erode the small particles
        K = 8
        reduced_result = cv2.erode(global_result, np.ones((K,K)), iterations=1)
        # result = np.concatenate((global_result, reduced_result), axis=0)
        result = reduced_result

        # Display the results
        cv2.imshow('Camera Window', result)

        key = cv2.waitKey(1)
        if key == ord('s'): # save
            cv2.imwrite('img_{}.png'.format(img_cnt), result)
            img_cnt += 1
        if key == ord('q'): # quit
            cap.release()
            cv2.destroyAllWindows()
            break
