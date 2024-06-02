from facenet_pytorch import MTCNN
from PIL import Image
from matplotlib import pyplot  as plt
import numpy as np
import math
import cv2

mtcnn = MTCNN(image_size=160,
              margin=0,
              min_face_size=20,
              thresholds=[0.6, 0.7, 0.7], # MTCNN thresholds
              factor=0.709,
              post_process=True,
              device='cpu' # If you don't have GPU
        )
lineColor = (255, 255, 0)

frontal = False
spoofing = False

# Landmaeks: [Left Eye], [Right eye], [nose], [left mouth], [right mouth]
def npAngle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b) 
    
    cosine_angle = np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def visualizeCV2(image, landmarks_, angle_R_, angle_L_, pred_, fontScale=2, fontThickness=3):
    if landmarks_ is None or angle_R_ is None or angle_L_ is None or pred_ is None:
        return
    
    for landmarks, angle_R, angle_L, pred in zip(landmarks_, angle_R_, angle_L_, pred_):
        if pred == 'Frontal':
            color = (0, 0, 0)
        elif pred == 'Right Profile':
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
            
        point1 = [int(landmarks[0][0]), int(landmarks[1][0])]
        point2 = [int(landmarks[0][1]), int(landmarks[1][1])]

        point3 = [int(landmarks[2][0]), int(landmarks[0][0])]
        point4 = [int(landmarks[2][1]), int(landmarks[0][1])]

        point5 = [int(landmarks[2][0]), int(landmarks[1][0])]
        point6 = [int(landmarks[2][1]), int(landmarks[1][1])]
        print(landmarks.shape)
        for land in landmarks:
            cv2.circle(image, (int(land[0]), int(land[1])), radius=5, color=(0, 255, 255), thickness=-1)
        cv2.line(image, (int(landmarks[0][0]), int(landmarks[0][1])), (int(landmarks[1][0]), int(landmarks[1][1])), lineColor, 3)
        cv2.line(image, (int(landmarks[0][0]), int(landmarks[0][1])), (int(landmarks[2][0]), int(landmarks[2][1])), lineColor, 3)
        cv2.line(image, (int(landmarks[1][0]), int(landmarks[1][1])), (int(landmarks[2][0]), int(landmarks[2][1])), lineColor, 3)
        
        text_sizeR, _ = cv2.getTextSize(pred, cv2.FONT_HERSHEY_PLAIN, fontScale, 4)
        text_wR, text_hR = text_sizeR
        
        cv2.putText(image, pred, (point1[0], point2[0]), cv2.FONT_HERSHEY_PLAIN, fontScale, color, fontThickness, cv2.LINE_AA)

def predFacePoseCV2(frame):
    
    bbox_, prob_, landmarks_ = mtcnn.detect(frame, landmarks=True) # The detection part producing bounding box, probability of the detected face, and the facial landmarks
    angle_R_List = []
    angle_L_List = []
    predLabelList = []
    
    if bbox_ is None or prob_ is None or landmarks_ is None:
        return None, None, None, None
    
    for bbox, landmarks, prob in zip(bbox_, landmarks_, prob_):
        if bbox is not None: # To check if we detect a face in the image
            if prob > 0.7: # To check if the detected face has probability more than 90%, to avoid 
                angR = npAngle(landmarks[0], landmarks[1], landmarks[2]) # Calculate the right eye angle
                angL = npAngle(landmarks[1], landmarks[0], landmarks[2])# Calculate the left eye angle
                angle_R_List.append(angR)
                angle_L_List.append(angL)
                if ((int(angR) in range(35, 57)) and (int(angL) in range(35, 58))):
                    predLabel='Frontal'
                    predLabelList.append(predLabel)
                else: 
                    if angR < angL:
                        predLabel='Left Profile'
                    else:
                        predLabel='Right Profile'
                    predLabelList.append(predLabel)
            else:
                print('The detected face is Less then the detection threshold')
        else:
            print('No face detected in the image')
    return landmarks_, angle_R_List, angle_L_List, predLabelList
