#importing libraries
#numpy: For number operations.
#cv2: For capturing webcam feed and image processing.
#keras: To load and use the trained traffic sign detection model.

import numpy as np
import cv2
from tensorflow import keras

# loading the model
threshold = 0.75
#font: Font used for labeling the sign on the screen.
font = cv2.FONT_HERSHEY_SIMPLEX
model = keras.models.load_model('traffif_sign_model.h5')

#Loads a pre-trained model that knows how to identify traffic signs.
#You trained this model earlier using traffic sign images.


# function to preprocess the image
def preprocess_img(imgBGR, erode_dilate=True):
    
#Converts image to HSV.
#Filters for red and blue colors (common in traffic signs).
#Applies erosion & dilation to reduce noise.
#Useful for creating a binary mask to locate potential signs.
    
    rows, cols, _ = imgBGR.shape
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    Bmin = np.array([100, 43, 46])
    Bmax = np.array([124, 255, 255])
    img_Bbin = cv2.inRange(imgHSV, Bmin, Bmax)

    Rmin1 = np.array([0, 43, 46])
    Rmax1 = np.array([10, 255, 255])
    img_Rbin1 = cv2.inRange(imgHSV, Rmin1, Rmax1)

    Rmin2 = np.array([156, 43, 46])
    Rmax2 = np.array([180, 255, 255])
    img_Rbin2 = cv2.inRange(imgHSV, Rmin2, Rmax2)
    img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
    img_bin = np.maximum(img_Bbin, img_Rbin)


    if erode_dilate:
        kernelErosion = np.ones((3, 3), np.uint8)
        kernelDilation = np.ones((3, 3), np.uint8)
        img_bin = cv2.erode(img_bin, kernelErosion, iterations=2)
        img_bin = cv2.dilate(img_bin, kernelDilation, iterations=2)

    return img_bin

# Counting the number of signs in the image
def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=2.0):

#Finds shapes (contours) in the binary image.
#Filters out shapes that are too small or not square-ish.
#Outputs bounding rectangles around each potential sign.
    rects = []
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return rects

    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])
    return rects

# Preprocessing functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

# Define labels
def getClassName(classNo):
    classes = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection',
        'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
        'No entry', 'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work',
        'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow',
        'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left',
        'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
    ]
    return classes[classNo] if classNo < len(classes) else "Unknown"

# Main function
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    #Opens the webcam feed.
    #Starts reading frames in a loop.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduce resolution
    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, img = cap.read()
        if not ret:
            break
        img_bin = preprocess_img(img, False)
        min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
        rects = contour_detect(img_bin, min_area=min_area)
        
        # Track the highest probability and associated rectangle
        highest_prob = threshold
        best_rect = None
        best_class = None
        
        for rect in rects:
            xc = int(rect[0] + rect[2] / 2)
            yc = int(rect[1] + rect[3] / 2)
            size = max(rect[2], rect[3])
            x1 = max(0, int(xc - size / 2))
            y1 = max(0, int(yc - size / 2))
            x2 = min(cols, int(xc + size / 2))
            y2 = min(rows, int(yc + size / 2))

            if rect[2] > 100 and rect[3] > 100:
                crop_img = np.asarray(img[y1:y2, x1:x2])
                crop_img = cv2.resize(crop_img, (32, 32))
                crop_img = preprocessing(crop_img)
                crop_img = crop_img.reshape(1, 32, 32, 1)
                predictions = model.predict(crop_img)
                classIndex = np.argmax(predictions, axis=-1)[0]
                probabilityValue = np.amax(predictions)

                if probabilityValue > highest_prob:
                    highest_prob = probabilityValue
                    best_rect = rect
                    best_class = classIndex

        if best_rect and best_class is not None:
            cv2.rectangle(img, (best_rect[0], best_rect[1]), 
                          (best_rect[0] + best_rect[2], best_rect[1] + best_rect[3]), 
                          (0, 0, 255), 2)
            cv2.putText(img, getClassName(best_class), 
                        (best_rect[0], best_rect[1] - 10), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"{round(highest_prob * 100, 2)}%", 
                        (best_rect[0], best_rect[1] - 40), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Output", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
