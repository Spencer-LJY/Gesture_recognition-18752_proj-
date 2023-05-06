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
        bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold by HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # hsv_result = cv2.inRange(hsv, (0, 30, 136), (56, 255, 255))
        hsv_result = cv2.inRange(hsv, (0, 0, 136), (56, 255, 255))

        # Blur HSV result
        global_result = cv2.medianBlur(hsv_result, 3)
        
        # finally, remove the small particles
        K = 5
        reduced_result = cv2.morphologyEx(global_result, cv2.MORPH_OPEN, np.ones((K,K)))
        reduced_result = cv2.morphologyEx(reduced_result, cv2.MORPH_CLOSE, np.ones((K,K)))
        # result = np.concatenate((hsv_result, global_result, reduced_result), axis=0)

        # Pick out all the contours in the image
        segmented_output = cv2.cvtColor(reduced_result, cv2.COLOR_GRAY2RGB)
        contours, hierarchy = cv2.findContours(reduced_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:

            # find the location of the biggest contour
            c = max(contours, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)

            # For the segmented_output...
            cv2.drawContours(segmented_output, contours, -1, 255, 3)    # draw all contours
            cv2.rectangle(segmented_output,(x,y),(x+w,y+h),(0,255,0),2) # draw biggest contour

            # For the binary image...
            # rescaled = # resize to 100 x 120 image

        # Display the results
        cv2.imshow('Camera Window', segmented_output)

        key = cv2.waitKey(1)
        if key == ord('s'): # save
            cv2.imwrite('img_{}.png'.format(img_cnt), result)
            img_cnt += 1
        if key == ord('q'): # quit
            cap.release()
            cv2.destroyAllWindows()
            break
