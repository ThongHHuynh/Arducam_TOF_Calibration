import cv2
import os
import numpy as np
from ArducamDepthCamera import ArducamCamera, Connection, DeviceType, FrameType, Control
print("Available FrameTypes:", dir(FrameType))

# Create folder to save images
save_dir = "calib_images"
os.makedirs(save_dir, exist_ok=True)

# Initialize camera
print("Opening Arducam ToF camera...")
tof = ArducamCamera()
ret = tof.open(Connection.CSI, 0)
if ret != 0:
    print("Failed to open camera. Error code:", ret)
    exit(1)

ret = tof.start(FrameType.DEPTH)
if ret != 0:
    print("Failed to start camera. Error code1:", ret)
    tof.close()
    exit(1)

info = tof.getCameraInfo()
width, height = info.width, info.height
print(f"Camera started: width={width}, height={height}")

# Set range to 4 meters 
tof.setControl(Control.RANGE, 4)

# Start capture loop
confidence_threshold = 30
i = 0

cv2.namedWindow("confidence", cv2.WINDOW_AUTOSIZE)
print("Press 's' to save frame, 'q' to quit")
while True:
    frame = tof.requestFrame(200)
    if frame is None:
        print("Failed to get frame")
        continue
    confidence = np.array(frame.confidence_data, dtype=np.float32).reshape(height, width)
    
    confidence_8u = cv2.convertScaleAbs(confidence, alpha=(255.0 / 1024.0))

    
    preview = cv2.resize(confidence_8u, (width*2,height *2), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("confidence", preview)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = os.path.join(save_dir, f"img_{i:02d}.png")
        cv2.imwrite(filename, preview)
        print(f"Saved: {filename}")
        i += 1

    tof.releaseFrame(frame)

# Cleanup
tof.stop()
tof.close()
cv2.destroyAllWindows()
print("Camera closed.")
