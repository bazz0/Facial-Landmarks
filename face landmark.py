# -*- coding: utf-8 -*-


import dlib
import cv2

g = input("Enter image name with location : ")



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img = cv2.imread(g, 1)

from google.colab.patches import cv2_imshow

#cv2_imshow(img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces_in_image = detector(img_gray, 0)





for face in faces_in_image:


    landmarks = predictor(img_gray, face)
  
	
    landmarks_list = []
    for i in range(0, landmarks.num_parts):
        landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

    jawline_list = []
    for i in range(0, 17):
		    jawline_list.append((landmarks.part(i).x, landmarks.part(i).y))
    
    
    for l in range(1, len(jawline_list)):
        ptA = tuple(jawline_list[l - 1])
        ptB = tuple(jawline_list[l])
        cv2.line(img, ptA, ptB, (255,0,0), 2)


landmarks_list.to_csv('landmark.csv')		
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

