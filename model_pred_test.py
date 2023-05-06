import numpy as np
import matplotlib.pyplot as plt
import keras
import time
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

def larger_window(x,y,w,h, ratio=1.1):
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    new_x = int(x - (new_w - w) / 2)
    new_y = int(y - (new_h - h) / 2)

    return new_x, new_y, new_w, new_h

if __name__ == "__main__":

    # Create camera
    cap = get_camera()
    img_cnt = 0

    # Label mappings
    model = keras.models.load_model("nn_model")
    LABELS =  ['call_me', 'fingers_crossed', 'okay', 'paper', 'peace', 'rock', 'rock_on', 'scissor', 'thumbs', 'up']
    LABELS_TO_INDEX = {k: v for v, k in enumerate(LABELS)}
    INDEX_TO_LABELS = {k: v for k, v in enumerate(LABELS)}
    

    while True:

        # Capture frame-by-frame
        frame = get_frame(cap)
        bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold by HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_result = cv2.inRange(hsv, (0, 0, 136), (56, 255, 255))

        # Blur HSV result
        global_result = cv2.medianBlur(hsv_result, 3)
        
        # finally, remove the small particles
        K = 5
        reduced_result = cv2.morphologyEx(global_result, cv2.MORPH_OPEN, np.ones((K,K)))
        reduced_result = cv2.morphologyEx(reduced_result, cv2.MORPH_CLOSE, np.ones((K,K)))
        cv2.imshow('Camera Window', reduced_result)

        # PASS INTO MODEL!
        cropped = reduced_result
        rescaled = cv2.resize(cropped, (100, 120))

        rescaled = rescaled.reshape(1, 100, 120, 1)
        pred = model.predict(rescaled, verbose = 0)
        y = np.argmax(pred)
        label = INDEX_TO_LABELS[y]
        print(label)

        key = cv2.waitKey(1)
        if key == ord('q'): # quit
            cap.release()
            cv2.destroyAllWindows()
            break

        time.sleep(0.05)