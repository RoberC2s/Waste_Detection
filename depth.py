import pyzed.sl as sl
import numpy as np
import cv2

def main():
    # Create a ZED camera object
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Choose the depth mode
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Choose the camera resolution
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

            # Get the dimensions of the depth image
            height, width = depth_image.get_height(), depth_image.get_width()
            midpoint = ((319), 195)

            # Retrieve depth value at the midpoint
            depth_value = depth_image.get_value(midpoint[0], midpoint[1])[1]

            if np.isfinite(depth_value):  # Check if the depth value is valid
                print(f"Distance at midpoint: {depth_value:.2f} meters")
            else:
                print("Invalid depth value at midpoint")

            # Convert depth map to numpy array
            depth_data = depth_image.get_data()

            # Normalize the depth values for visualization
            depth_data_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Convert single channel grayscale to BGR
            depth_data_normalized_color = cv2.cvtColor(depth_data_normalized, cv2.COLOR_GRAY2BGR)

            # Draw a red dot at the midpoint
            cv2.circle(depth_data_normalized_color, midpoint, 5, (0, 0, 255), -1)

            # Display the depth map
            cv2.imshow("Depth Map", depth_data_normalized_color)

            key = cv2.waitKey(10)
            if key == 27:  # ESC key to exit
                break

    # Release resources
    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
