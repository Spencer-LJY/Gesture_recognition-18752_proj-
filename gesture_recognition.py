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

def KMeans(frame):
    pass
    return frame

if __name__ == "__main__":

    cap = get_camera()

    while True:

        # Capture frame-by-frame
        frame = get_frame(cap)

        # Perform operations here
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = KMeans(frame)

        # Display the results
        cv2.imshow('Camera Window', result)

        # Quit w/ 'q'
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
