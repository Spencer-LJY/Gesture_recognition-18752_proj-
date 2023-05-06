import numpy as np
import matplotlib.pyplot as plt
# import keras
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
    # model = keras.models.load_model("nn_model")
    LABELS =  ['call_me', 'fingers_crossed', 'okay', 'paper', 'peace', 'rock', 'rock_on', 'scissor', 'thumbs', 'up']
    LABELS_TO_INDEX = {k: v for v, k in enumerate(LABELS)}
    INDEX_TO_LABELS = {k: v for k, v in enumerate(LABELS)}
    

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
            x,y,w,h = larger_window(x,y,w,h, ratio=1.05)

            # if bounding box is outside of the image, then skip
            # if (x < 0) or (y < 0) or (x+w > reduced_result.shape[1]) or (y+h > reduced_result.shape[0]):
            #     continue

            # Pass the binary image to the model
            cropped = reduced_result[x:x+w, y:y+h]
            # print(cropped.shape)
            # print(x,y,w,h)
            if 0 in cropped.shape:
                continue
            # rescaled = cv2.resize(cropped, (100, 120)) # resize to 100 x 120 image
            # pred = model.predict(rescaled)
            # y = np.argmax(pred)
            # label = INDEX_TO_LABELS[y]

            # For the segmented_output...
            cv2.drawContours(segmented_output, contours, -1, 255, 3)    # draw all contours
            cv2.rectangle(segmented_output,(x,y),(x+w,y+h),(0,255,0),2) # draw biggest contour
            
            # segmented_output = cv2.putText(segmented_output,
            #                                label,
            #                                org=(x, y),
            #                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #                                fontScale=1,
            #                                color=(255, 0, 0),
            #                                thickness=2,
            #                                lineType=cv2.LINE_AA)

        # Display the results
        cv2.imshow('Camera Window', segmented_output)

        key = cv2.waitKey(1)
        if key == ord('s'): # save
            cv2.imwrite('img_{}.png'.format(img_cnt), segmented_output)
            img_cnt += 1
        if key == ord('q'): # quit
            cap.release()
            cv2.destroyAllWindows()
            break
