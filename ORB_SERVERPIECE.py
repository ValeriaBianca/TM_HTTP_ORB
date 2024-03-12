import cv2
import numpy as np
import pandas as pd
import os
import io

nu = 0.75
from matplotlib import pyplot as plt
from tabulate import tabulate
from PIL import Image 


trainim = cv2.imread(r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\TM_HTTP_2024\mosaic_ref\Train\Verde.jpg")
DX = r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\TM_HTTP_2024\mosaic_ref\Scene\Scene1.jpg"
    
SX = r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\upload\image_SX.jpg"

eps = 700 #pixel

imgD = cv2.imread(DX)
imgS = cv2.imread(SX)


def UndistAndRect(img, stereomap_x, stereomap_y):
    frame = cv2.remap(img, stereomap_x, stereomap_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    return frame

def find_depth(right_point, left_point, frame_right, frame_left, baseline, focal, alpha):
    heightR, widthR, depthR = frame_right.shape
    heightL, widthL, depthL = frame_left.shape

    if widthR == widthL:
        #f_pixel = (widthR*0.5)/np.tan(alpha*0.5*np.pi/180) # I already have the info in pixel and I am using the same camera so it should be fxS = fxD and fyS = fyD
        f_pixel = focal
    else:
        print("Left and right images do not have the same pixel width")
    
    # I am using only coordinate x because if the images are undistorted and rectified then
    # I can check the disparity only on the x axis
    xR = right_point[0] 
    xL = left_point[0]

    #-- Calculate disparity
    disparity = xL-xR

    #-- Evaluate depth Z
    zDepth = (baseline*f_pixel)/disparity
    return abs(zDepth)

#-- Undistort and rectify images before giving them to the ORB algorithm to process
#-- Use camera parameters PREVIOUSLY evaluated by stereo calibration script
cv_file = cv2.FileStorage()
cv_file.open(r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\stereoMap.xml", cv2.FILE_STORAGE_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
cv_file.release()

#imgD = cv2.cvtColor(imgD, cv2.COLOR_RGB2GRAY)
#imgS = cv2.cvtColor(imgS, cv2.COLOR_RGB2GRAY) #plt uses RGB

imgD = UndistAndRect(imgD, stereoMapR_x,stereoMapR_y)
imgS = UndistAndRect(imgS, stereoMapL_x,stereoMapL_y)

#plt.imshow(imgD),plt.show()
#plt.imshow(imgS),plt.show()


#outputimg.save(os.path.join(r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\upload", "output_undist"))
#print('File saved succesfully')

#----------- first for SX image
outputimg = imgS.copy()

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1 = orb.detect(trainim,None)
kp2 = orb.detect(imgS,None)

kp1, des1 = orb.compute(trainim, kp1)
kp2, des2 = orb.compute(imgS, kp2)

# create BFMatcher object
bf = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=False)

print(np.size(des1), np.size(des2))
# Match descriptors, knn method.
matches = bf.knnMatch(des1,des2,k=2)

# I can also mask the keypoints by "filtering" only the best 
# ones; a.k.a. the keypoints whose descriptor have low distance

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x: x[:][1].distance)

good = []
for m,n in matches:
    if m.distance < nu*n.distance:
        good.append([m])

Matched = cv2.drawMatchesKnn(trainim,kp1,imgS,kp2,
                        good,outImg=None,matchColor=(0, 155, 0),singlePointColor=(0, 255, 255),matchesMask=None,flags=0)

kp3 = [] #creating an empty keypoint object

for i in range(len(good)):
    a = good[i][0].trainIdx 
    idx=kp2[a].pt
    key = cv2.KeyPoint(idx[0],idx[1],1)
    kp3.append(key)


output_img = cv2.drawKeypoints(outputimg, kp3 ,0,(255,0,0),flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS) 
#plt.imshow(output_img),plt.show()


#===========Getting coordinates for box=====================================================================================================
#-- Localize object
obj = np.empty((len(good),2), dtype = np.float32)
scene = np.empty((len(good),2), dtype = np.float32)
output = np.empty((len(good),2), dtype=np.float32)

for i in range(len(good)):
    obj[i,0] = kp1[good[i][0].queryIdx].pt[0] #coordinata x del keypoint delle train image che corrisponde 
    # all'i esimo keypoint nella query image- sfrutto il match object che mappa le corrispondenze tra i due set di kypoints
    # nell' obj metto quindi le coordinate in pixel della train image con corrispondenza alla query
    # nell'oggetto scene faccio l'opposto: salvo le coodinate in pixel della query image che corrispondono ai keypoint della train image
    obj[i,1] = kp1[good[i][0].queryIdx].pt[1]
    scene[i,0] = kp2[good[i][0].trainIdx].pt[0]
    scene[i,1] = kp2[good[i][0].trainIdx].pt[1]
# questo kp3 serve semplicemente a creare l'immmagine di output singola da quella dove ho l'immagine di train e di query affiancate
for i in range(len(kp3)):    
    output[i,0] = kp3[i].pt[0] 
    output[i,1] = kp3[i].pt[1] 

try:
    H, _ = cv2.findHomography(obj,scene,cv2.RANSAC)
    H2, _ = cv2.findHomography(obj, output, cv2.RANSAC)
except:
    print("Not enough matches or zero matches found!")
    result = {            
        "message": "Not enough matches or zero matches found!",
        "result": None            
    }
#return result #with this return the server can respond to more post/get requests
#-- Get the corners from the image_1 ( the object to be "detected" )
obj_corners = np.empty((4,1,2), dtype=np.float32)
obj_corners[0,0,0] = 0
obj_corners[0,0,1] = 0

obj_corners[1,0,0] = trainim.shape[1]
obj_corners[1,0,1] = 0

obj_corners[2,0,0] = trainim.shape[1]
obj_corners[2,0,1] = trainim.shape[0]

obj_corners[3,0,0] = 0
obj_corners[3,0,1] = trainim.shape[0]

#-- Here I get the corners of the train object "mapped" on to the query image coordinates through the homography matrix
# previously evaluated
scene_corners = cv2.perspectiveTransform(obj_corners, H)
output_corners = cv2.perspectiveTransform(obj_corners, H2) 

#-- Draw lines between the corners (the mapped object in the scene - image_2 )
cv2.line(Matched, 
    (int(scene_corners[0,0,0] + trainim.shape[1]), int(scene_corners[0,0,1])),\
    (int(scene_corners[1,0,0] + trainim.shape[1]), int(scene_corners[1,0,1])),
    (0,255,0), 4)
cv2.line(Matched, (int(scene_corners[1,0,0] + trainim.shape[1]), int(scene_corners[1,0,1])),\
(int(scene_corners[2,0,0] + trainim.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
cv2.line(Matched, (int(scene_corners[2,0,0] + trainim.shape[1]), int(scene_corners[2,0,1])),\
(int(scene_corners[3,0,0] + trainim.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
cv2.line(Matched, (int(scene_corners[3,0,0] + trainim.shape[1]), int(scene_corners[3,0,1])),\
(int(scene_corners[0,0,0] + trainim.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)

cv2.line(output_img, 
(int(output_corners[0,0,0] ), int(output_corners[0,0,1])),\
(int(output_corners[1,0,0] ), int(output_corners[1,0,1])),
(0,255,0), 4) 
# (coordinata x 0,0,0 , coordinata y 0,0,1); le altre due coordinate dopo il backslash sono x,y del punto successivo
cv2.line(output_img, (int(output_corners[1,0,0] ), int(output_corners[1,0,1])),\
(int(output_corners[2,0,0] ), int(output_corners[2,0,1])), (0,255,0), 4)
cv2.line(output_img, (int(output_corners[2,0,0] ), int(output_corners[2,0,1])),\
(int(output_corners[3,0,0] ), int(output_corners[3,0,1])), (0,255,0), 4)
cv2.line(output_img, (int(output_corners[3,0,0] ), int(output_corners[3,0,1])),\
(int(output_corners[0,0,0]), int(output_corners[0,0,1])), (0,255,0), 4)

#plt.imshow(output_img),plt.show()
#-- Get coordinates, height and width of square box
cx = (output_corners[0,0,0]+output_corners[1,0,0]+output_corners[2,0,0]+output_corners[3,0,0])/4
cy = (output_corners[0,0,1]+output_corners[1,0,1]+output_corners[2,0,1]+output_corners[3,0,1])/4
box_h = np.sqrt(np.square(output_corners[0,0,0]-output_corners[3,0,0])+np.square(output_corners[0,0,1]-output_corners[3,0,1]))
box_w = np.sqrt(np.square(output_corners[0,0,0]-output_corners[1,0,0])+np.square(output_corners[0,0,1]-output_corners[1,0,1]))

#-- Get rotation of square box
# acos((tr(R)-1)/2), Rodrigues formula inverted
#print((np.trace(H2)-1)/2)
#theta = np.arccos((np.trace(H2)-1)/2) #answer is in radians
theta = - np.arctan2(H2[0,1], H2[0,0]) 
theta = np.rad2deg(theta)


result = {
            "message":"success",
            "annotations":[
                {
                    "box_cx": float(str(cx)),
                    "box_cy": float(str(cy)),
                    "box_w": float(str(box_w)),
                    "box_h": float(str(box_h)),
                    "label": "SX",
                    "score": float(str(1.000)),
                    "rotation": float(str(theta))

                }
            ],
            "result": "ImageSX" 
        }

table = [["label",result["annotations"][0]["label"]],["box_cx", result["annotations"][0]["box_cx"]],
            ["box_cy",result["annotations"][0]["box_cy"]],["box_w",result["annotations"][0]["box_w"]],
            ["box_h",result["annotations"][0]["box_h"]],["rotation", result["annotations"][0]["rotation"]]]

title = "label values"          
with open(r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\jsonsx.txt", 'a') as f:
    f.write(tabulate(table))
    f.close()

#----------- same thing but for DX image
outputimg = imgD.copy()
# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1 = orb.detect(trainim,None)
kp2 = orb.detect(imgD,None)

kp1, des1 = orb.compute(trainim, kp1)
kp2, des2 = orb.compute(imgD, kp2)

# create BFMatcher object
bf = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=False)

# Match descriptors, knn method.
matches = bf.knnMatch(des1,des2,k=2)

# I can also mask the keypoints by "filtering" only the best 
# ones; a.k.a. the keypoints whose descriptor have low distance

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x: x[:][1].distance)

good = []
for m,n in matches:
    if m.distance < nu*n.distance:
        good.append([m])

Matched = cv2.drawMatchesKnn(trainim,kp1,imgS,kp2,
                        good,outImg=None,matchColor=(0, 155, 0),singlePointColor=(0, 255, 255),matchesMask=None,flags=0)

kp3 = [] #creating an empty keypoint object

for i in range(len(good)):
    a = good[i][0].trainIdx 
    idx=kp2[a].pt
    key = cv2.KeyPoint(idx[0],idx[1],1)
    kp3.append(key)


output_img = cv2.drawKeypoints(outputimg, kp3 ,0,(255,0,0),flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS) 

#plt.imshow(output_img),plt.show()

#===========Getting coordinates for box=====================================================================================================
#-- Localize object
obj = np.empty((len(good),2), dtype = np.float32)
scene = np.empty((len(good),2), dtype = np.float32)
output = np.empty((len(good),2), dtype=np.float32)

for i in range(len(good)):
    obj[i,0] = kp1[good[i][0].queryIdx].pt[0] #coordinata x del keypoint delle train image che corrisponde 
    # all'i esimo keypoint nella query image- sfrutto il match object che mappa le corrispondenze tra i due set di kypoints
    # nell' obj metto quindi le coordinate in pixel della train image con corrispondenza alla query
    # nell'oggetto scene faccio l'opposto: salvo le coodinate in pixel della query image che corrispondono ai keypoint della train image
    obj[i,1] = kp1[good[i][0].queryIdx].pt[1]
    scene[i,0] = kp2[good[i][0].trainIdx].pt[0]
    scene[i,1] = kp2[good[i][0].trainIdx].pt[1]
# questo kp3 serve semplicemente a creare l'immmagine di output singola da quella dove ho l'immagine di train e di query affiancate
for i in range(len(kp3)):    
    output[i,0] = kp3[i].pt[0] 
    output[i,1] = kp3[i].pt[1] 

try:
    H, _ = cv2.findHomography(obj,scene,cv2.RANSAC)
    H2, _ = cv2.findHomography(obj, output, cv2.RANSAC)
except:
    print("Not enough matches or zero matches found!")
    result = {            
        "message": "Not enough matches or zero matches found!",
        "result": None            
    }
#return result #with this return the server can respond to more post/get requests
#-- Get the corners from the image_1 ( the object to be "detected" )
obj_corners = np.empty((4,1,2), dtype=np.float32)
obj_corners[0,0,0] = 0
obj_corners[0,0,1] = 0

obj_corners[1,0,0] = trainim.shape[1]
obj_corners[1,0,1] = 0

obj_corners[2,0,0] = trainim.shape[1]
obj_corners[2,0,1] = trainim.shape[0]

obj_corners[3,0,0] = 0
obj_corners[3,0,1] = trainim.shape[0]

#-- Here I get the corners of the train object "mapped" on to the query image coordinates through the homography matrix
# previously evaluated
scene_corners = cv2.perspectiveTransform(obj_corners, H)
output_corners = cv2.perspectiveTransform(obj_corners, H2) 

#-- Draw lines between the corners (the mapped object in the scene - image_2 )
cv2.line(Matched, 
    (int(scene_corners[0,0,0] + trainim.shape[1]), int(scene_corners[0,0,1])),\
    (int(scene_corners[1,0,0] + trainim.shape[1]), int(scene_corners[1,0,1])),
    (0,255,0), 4)
cv2.line(Matched, (int(scene_corners[1,0,0] + trainim.shape[1]), int(scene_corners[1,0,1])),\
(int(scene_corners[2,0,0] + trainim.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
cv2.line(Matched, (int(scene_corners[2,0,0] + trainim.shape[1]), int(scene_corners[2,0,1])),\
(int(scene_corners[3,0,0] + trainim.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
cv2.line(Matched, (int(scene_corners[3,0,0] + trainim.shape[1]), int(scene_corners[3,0,1])),\
(int(scene_corners[0,0,0] + trainim.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)

cv2.line(output_img, 
(int(output_corners[0,0,0] ), int(output_corners[0,0,1])),\
(int(output_corners[1,0,0] ), int(output_corners[1,0,1])),
(0,255,0), 4) 
# (coordinata x 0,0,0 , coordinata y 0,0,1); le altre due coordinate dopo il backslash sono x,y del punto successivo
cv2.line(output_img, (int(output_corners[1,0,0] ), int(output_corners[1,0,1])),\
(int(output_corners[2,0,0] ), int(output_corners[2,0,1])), (0,255,0), 4)
cv2.line(output_img, (int(output_corners[2,0,0] ), int(output_corners[2,0,1])),\
(int(output_corners[3,0,0] ), int(output_corners[3,0,1])), (0,255,0), 4)
cv2.line(output_img, (int(output_corners[3,0,0] ), int(output_corners[3,0,1])),\
(int(output_corners[0,0,0]), int(output_corners[0,0,1])), (0,255,0), 4)

#plt.imshow(output_img),plt.show()
#-- Get coordinates, height and width of square box
cx = (output_corners[0,0,0]+output_corners[1,0,0]+output_corners[2,0,0]+output_corners[3,0,0])/4
cy = (output_corners[0,0,1]+output_corners[1,0,1]+output_corners[2,0,1]+output_corners[3,0,1])/4
box_h = np.sqrt(np.square(output_corners[0,0,0]-output_corners[3,0,0])+np.square(output_corners[0,0,1]-output_corners[3,0,1]))
box_w = np.sqrt(np.square(output_corners[0,0,0]-output_corners[1,0,0])+np.square(output_corners[0,0,1]-output_corners[1,0,1]))

#-- Get rotation of square box
# acos((tr(R)-1)/2), Rodrigues formula inverted
#print((np.trace(H2)-1)/2)
#theta = np.arccos((np.trace(H2)-1)/2) #answer is in radians
theta = - np.arctan2(H2[0,1], H2[0,0]) 
theta = np.rad2deg(theta)

result = {
            "message":"success",
            "annotations":[
                {
                    "box_cx": float(str(cx)),
                    "box_cy": float(str(cy)),
                    "box_w": float(str(box_w)),
                    "box_h": float(str(box_h)),
                    "label": "DX",
                    "score": float(str(1.000)),
                    "rotation": float(str(theta))

                }
            ],
            "result": "ImageDX" 
        }

table = [["label",result["annotations"][0]["label"]],["box_cx", result["annotations"][0]["box_cx"]],
            ["box_cy",result["annotations"][0]["box_cy"]],["box_w",result["annotations"][0]["box_w"]],
            ["box_h",result["annotations"][0]["box_h"]],["rotation", result["annotations"][0]["rotation"]]]
        #print(table)
title = "label values"          
with open(r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\jsondx.txt", 'a') as f:
    f.write('\n')
    f.write(str(title))
    f.write('\n')
    f.write(tabulate(table))
    f.close()


