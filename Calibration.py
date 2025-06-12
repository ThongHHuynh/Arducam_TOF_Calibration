import cv2 as cv
import numpy as np
import glob

pattern_size = (8,6)
square_size = 0.03
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

images = glob.glob('calib_images/*.png')


for fname in images:
    img = cv.imread(fname, cv.IMREAD_GRAYSCALE)
 
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(img, pattern_size, None)
 
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(img,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Draw and display the corners
        cv.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(100)
    else:
        print(f"Failed to detect corners in {fname}")
        key = cv.waitKey(100)
        if key == ord('q'):
            break
 
cv.destroyAllWindows()

ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, img.shape[::-1], None, None
)

#print result
print("Calibration complete")
print("Camera Matrix (Intrinsics):\n", cameraMatrix)
print("\nDistortion Coefficients:\n", distCoeffs.ravel())

#extrinsic
#for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
#    R, _ = cv.Rodrigues(rvec)
#    print(f"\nImage {i}:")
#    print("Rotation Matrix:\n", R)
#    print("Translation Vector:\n", tvec.ravel())

# save to file
np.savez("camera_calib_full.npz",
         cameraMatrix=cameraMatrix,
         distCoeffs=distCoeffs,
         rvecs=rvecs,
         tvecs=tvecs)

print("result released")
