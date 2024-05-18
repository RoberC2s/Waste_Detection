import pyzed.sl as sl
import open3d as o3d
import numpy as np

def convert_zed_point_cloud(zed_point_cloud):
    width = zed_point_cloud.get_width()
    height = zed_point_cloud.get_height()
    
    # Create a numpy array from the ZED point cloud data
    point_cloud_np = zed_point_cloud.get_data(sl.MEM.CPU).reshape((height, width, 4))
    
    # Reshape and filter valid points
    valid_points = point_cloud_np[~np.isnan(point_cloud_np).any(axis=2) & (point_cloud_np[..., 2] != 0)]
    
    # Split into xyz coordinates and color
    points = valid_points[:, :3]
    colors = valid_points[:, 3]
    
    # Normalize color values to [0, 1]
    colors = np.stack((colors, colors, colors), axis=1) / 255.0
    
    print(f"Number of valid points: {len(points)}")
    return points, colors

def main():
    # Create a ZED camera object
    zed = sl.Camera()

    # Create initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use a more detailed depth mode

    # Open the camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Error opening ZED camera")
        return

    # Create runtime parameters
    runtime_parameters = sl.RuntimeParameters(enable_fill_mode = True)


    # Capture point cloud data
    point_cloud = sl.Mat()
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

    # Convert the ZED point cloud to Open3D format
    points, colors = convert_zed_point_cloud(point_cloud)

    # Create an Open3D point cloud object
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d_point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Print some debug information
    print(f"First 5 points: {points[:5]}")
    print(f"First 5 colors: {colors[:5]}")

    # Visualize the point cloud using Open3D
    o3d.visualization.draw_geometries([o3d_point_cloud])

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
