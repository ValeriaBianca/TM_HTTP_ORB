import numpy as np
import cv2 as cv
import glob
import pandas as pd

from tabulate import tabulate

chessboardSize=(9,6) 


frameSize = (2592,1944)
#-- Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#-- Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) 

#-- Size of each chessboard square in world coordinates and in mm
objp = objp*21
 
#-- Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

#-- Routes to left and right images
imagesLeft = glob.glob('C:\\Users\\bianc\\TMvision_TmHttp_server_sample_code\\python_example\\images\\calib_trial_400\\Left\\*.jpg')
imagesRight = glob.glob('C:\\Users\\bianc\\TMvision_TmHttp_server_sample_code\\python_example\\images\\calib_trial_400\\Right\\*.jpg')

#-- For loop
for imL,imR in zip(imagesLeft,imagesRight):
    #-- Pick pair of left and right images
    imgL = cv.imread(imL)
    imgR = cv.imread(imR)

    #-- Convert to gray both images
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    #-- Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)
    
    #-- If chessboard is found, add object points, image points (after refining them)
    if retL and retR == True:
        objpoints.append(objp)

        #-- This function iteratively refines the corner locations untill the termination criteria is reached
        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)
        
        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)
        
        #-- Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.namedWindow("outputL", cv.WINDOW_NORMAL) 
        cv.imshow("outputL", imgL)
        cv.resizeWindow("outputL", 960, 540)

        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.namedWindow("outputR", cv.WINDOW_NORMAL)
        cv.imshow("outputR", imgR)
        cv.resizeWindow("outputR", 960, 540) 

        cv.waitKey(1000)
    else:
        print("No chessboard found")

cv.destroyAllWindows()

#-- Calibration for each camera
retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, frameSize, 1, frameSize)

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, frameSize, 1, frameSize)

#-- Stereo Vision Calibration
flags = 0
flags = cv.CALIB_FIX_INTRINSIC #with this flag intrinsic matrices are fixed so that only
# R,T, fundamental and essential matrices are evaluated

criteria_stereo = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30 , 0.001)

retStereo, newCameraMatrixL,distL,newCameraMatrixR, distR, rot, trans, essentialMat, fundamentalMat =  cv.stereoCalibrate(objpoints,
        imgpointsL,imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, frameSize) 

#-- Stero Rectification
rectifyScale = 1
rectL, rectR, projMatL, projMatR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL,
        distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL,distL, rectL, projMatL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR,distR, rectR, projMatR, grayR.shape[::-1], cv.CV_16SC2)

#-- Saving parameters
print("Saving parameters") 
cv_file = cv.FileStorage('C:\\Users\\bianc\\TMvision_TmHttp_server_sample_code\\python_example\\stereoMap.xml', 
        cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x', stereoMapL[0])
cv_file.write('stereoMapL_y', stereoMapL[1])
cv_file.write('stereoMapR_x', stereoMapR[0])
cv_file.write('stereoMapR_y', stereoMapR[1])

cv_file.release()

CamSX = np.array([newCameraMatrixL[0][0],newCameraMatrixL[1][1],newCameraMatrixL[0][2],newCameraMatrixL[1][2]])
CamDX = np.array([newCameraMatrixR[0][0],newCameraMatrixR[1][1],newCameraMatrixR[0][2],newCameraMatrixR[1][2]])
data = {'SX': CamSX , 'DX': CamDX}
tableCam = pd.DataFrame( data, index = ['fx','fy','cx','cy'])

tableCam = tabulate(tableCam, headers = ['SX','DX'])

with open('C:\\Users\\bianc\\TMvision_TmHttp_server_sample_code\\python_example\\StereoMatrices.txt','a') as f:
    f.write(tableCam)
    f.close()