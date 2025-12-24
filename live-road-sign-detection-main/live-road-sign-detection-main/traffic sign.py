#importing libraries
import numpy as np
import cv2
from tensorflow import keras

# Load the trained traffic sign model
threshold = 0.90  # Increased threshold to reduce false positives
font = cv2.FONT_HERSHEY_SIMPLEX
model = keras.models.load_model('traffif_sign_model.h5')

# Preprocess the image to extract red and blue regions
def preprocess_img(imgBGR, erode_dilate=True):
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
        kernel = np.ones((3, 3), np.uint8)
        img_bin = cv2.erode(img_bin, kernel, iterations=2)
        img_bin = cv2.dilate(img_bin, kernel, iterations=2)

    return img_bin

# Detect contours and filter based on shape and area
def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=1.2):
    rects = []
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= wh_ratio:
                approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
                if len(approx) >= 4:  # basic shape filtering
                    rects.append([x, y, w, h])
    return rects

# Preprocessing steps for the model
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

# Class name lookup
def getClassName(classNo):
    classes = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
        'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
        'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    return classes[classNo] if classNo < len(classes) else "Unknown"

# Main detection loop
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img_bin = preprocess_img(img, erode_dilate=True)
        min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
        rects = contour_detect(img_bin, min_area=min_area)

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

                predictions = model.predict(crop_img, verbose=0)
                classIndex = np.argmax(predictions, axis=-1)[0]
                probabilityValue = np.amax(predictions)

                if probabilityValue > highest_prob:
                    highest_prob = probabilityValue
                    best_rect = rect
                    best_class = classIndex

        if best_rect and best_class is not None:
            cv2.rectangle(img, (best_rect[0], best_rect[1]),
                          (best_rect[0] + best_rect[2], best_rect[1] + best_rect[3]),
                          (0, 255, 0), 2)
            cv2.putText(img, getClassName(best_class),
                        (best_rect[0], best_rect[1] - 10), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"{round(highest_prob * 100, 2)}%",
                        (best_rect[0], best_rect[1] - 35), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Traffic Sign Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
