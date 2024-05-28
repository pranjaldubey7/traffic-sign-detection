import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
#############################################
 
frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################
 
# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL
model=load_model("model.h5")  ## rb = READ BYTE
 
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getCalssName(classNo):
    if np.any(classNo == 0): return 'Speed Limit 20 km/h'
    elif np.any(classNo == 1): return 'Speed Limit 30 km/h'
    elif np.any(classNo == 2): return 'Speed Limit 50 km/h'
    elif np.any(classNo == 3): return 'Speed Limit 60 km/h'
    elif np.any(classNo == 4): return 'Speed Limit 70 km/h'
    elif np.any(classNo == 5): return 'Speed Limit 80 km/h'
    elif np.any(classNo == 6): return 'End of Speed Limit 80 km/h'
    elif np.any(classNo == 7): return 'Speed Limit 100 km/h'
    elif np.any(classNo == 8): return 'Speed Limit 120 km/h'
    elif np.any(classNo == 9): return 'No passing'
    elif np.any(classNo == 10): return 'No passing for vehicles over 3.5 metric tons'
    elif np.any(classNo == 11): return 'Right-of-way at the next intersection'
    elif np.any(classNo == 12): return 'Priority road'
    elif np.any(classNo == 13): return 'Yield'
    elif np.any(classNo == 14): return 'Stop'
    elif np.any(classNo == 15): return 'No vehicles'
    elif np.any(classNo == 16): return 'Vehicles over 3.5 metric tons prohibited'
    elif np.any(classNo == 17): return 'No entry'
    elif np.any(classNo == 18): return 'General caution'
    elif np.any(classNo == 19): return 'Dangerous curve to the left'
    elif np.any(classNo == 20): return 'Dangerous curve to the right'
    elif np.any(classNo == 21): return 'Double curve'
    elif np.any(classNo == 22): return 'Bumpy road'
    elif np.any(classNo == 23): return 'Slippery road'
    elif np.any(classNo == 24): return 'Road narrows on the right'
    elif np.any(classNo == 25): return 'Road work'
    elif np.any(classNo == 26): return 'Traffic signals'
    elif np.any(classNo == 27): return 'Pedestrians'
    elif np.any(classNo == 28): return 'Children crossing'
    elif np.any(classNo == 29): return 'Bicycles crossing'
    elif np.any(classNo == 30): return 'Beware of ice/snow'
    elif np.any(classNo == 31): return 'Wild animals crossing'
    elif np.any(classNo == 32): return 'End of all speed and passing limits'
    elif np.any(classNo == 33): return 'Turn right ahead'
    elif np.any(classNo == 34): return 'Turn left ahead'
    elif np.any(classNo == 35): return 'Ahead only'
    elif np.any(classNo == 36): return 'Go straight or right'
    elif np.any(classNo == 37): return 'Go straight or left'
    elif np.any(classNo == 38): return 'Keep right'
    elif np.any(classNo == 39): return 'Keep left'
    elif np.any(classNo == 40): return 'Roundabout mandatory'
    elif np.any(classNo == 41): return 'End of no passing'
    elif np.any(classNo == 42): return 'End of no passing by vehicles over 3.5 metric tons'
    else: return 'Unknown'
 
while True:
    # READ IMAGE
    success, imgOrignal = cap.read()
    
    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = model.predict(img)
    probabilityValue =np.amax(predictions)
    cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)

    k=cv2.waitKey(1) 
    if k== ord('q'):
        break

cv2.destroyAllWindows()
cap.release()