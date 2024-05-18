import pyzed.sl as sl
import numpy as np
import cv2

def main():
    # Create a ZED camera object
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Choose the depth mode
    init_params.camera_resolution = sl.RESOLUTION.VGA  # Choose the camera resolution
    init_params.coordinate_units = sl.UNIT.METER  # Choose the depth unit

    zed = sl.Camera()
    if not zed.is_opened():
        print("Opening ZED Camera...")
        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()

    runtime_parameters = sl.RuntimeParameters()

    # Create an empty matrix to store depth data
    depth_image = sl.Mat()

    # Main loop
    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve depth map
            zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)

            # Convert depth map to numpy array
            depth_data = depth_image.get_data()

            # Normalize the depth values for visualization
            depth_data_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Display the depth map
            cv2.imshow("Depth Map", depth_data_normalized)

            key = cv2.waitKey(10)
            if key == 27:
                break

    # Close the ZED camera
    zed.close()

if __name__ == "__main__":
    main()
