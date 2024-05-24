import cv2
import numpy as np
import torch
import os
from time import time
from ultralytics import YOLO
import argparse
import math
import random
from publisher import Publisher
from publisher import get_object_data
import pyzed.sl as sl

from time import sleep
from supervision.draw.color import ColorPalette
from supervision import Detections, BoxAnnotator


##Test function 
def object_data2pub(objects, pub):
    print("É suposto publicar ", len(objects), "mensanges")
    print(objects)
    for i in range(len(objects)):
        message = objects[i].to_dict()
        pub.publish(message)
    #pub.publish(objects)
    return True

def XYZaxis_from_OXY(frame_origin_3d, frame_x_axis_3d, frame_y_axis_3d):
    # Calculate frame XYZ axis
    frame_x_axis = frame_x_axis_3d - frame_origin_3d
    frame_y_axis = frame_y_axis_3d - frame_origin_3d
    frame_z_axis = np.cross(frame_x_axis, frame_y_axis)
    # Make sure XYZ orthogonality
    frame_y_axis = np.cross(frame_z_axis, frame_x_axis)
    # Normalize XYZ axis
    frame_x_axis = frame_x_axis / np.linalg.norm(frame_x_axis)
    frame_y_axis = frame_y_axis / np.linalg.norm(frame_y_axis)
    frame_z_axis = frame_z_axis / np.linalg.norm(frame_z_axis)
    return frame_x_axis, frame_y_axis, frame_z_axis



def calculate_transform_matrix(x_axis, y_axis, z_axis, origin):
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    print(rotation_matrix)
    translation_vector = np.array(origin)
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)
    inv_translation_vector = -np.dot(inv_rotation_matrix, translation_vector)
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = inv_rotation_matrix
    transform_matrix[:3, 3] = inv_translation_vector
    return transform_matrix

def transform_point(point, transform_matrix):
    point_homogeneous = np.hstack((point, 1))
    transformed_point_homogeneous = np.dot(transform_matrix, point_homogeneous)
    transformed_point = transformed_point_homogeneous[:3] / transformed_point_homogeneous[3]
    return transformed_point


def orthogonalize(v1, v2):
    # Gram-Schmidt process
    u1 = v1
    u2 = -v2 + np.dot(v2, u1) / np.dot(u1, u1) * u1
    return u1, u2

def normalize_vector(v):

    magnitude = np.linalg.norm(v)
    if magnitude == 0:
        return v
    return v / magnitude


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument("--webcam-index", default=0, type=int, help="Index of the webcam")
    args = parser.parse_args()
    return args



class ArUco:
    ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }
    def __init__(self):
        # Init ArUco
        #parameters = cv2.aruco.DetectorParameters_create()
        self.aruco_type = "DICT_4X4_100"


    def aruco_display(corners, ids, rejected, image):
        if len(corners) > 0:
            ids = ids.flatten()
            for (markerCorner, markerID) in zip(corners, ids):
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                
                cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                print("[Inference] ArUco marker ID: {}".format(markerID))
        return image


    

    def ref_estimation(self, frame, aruco_dict_type):
        font = cv2.FONT_HERSHEY_PLAIN
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        matrix_coefficients = np.array(((1406.57, 0, 980.817),(0,1407.67, 618.202),(0,0,1))) 
        distortion_coefficients = np.array((-0.16597,0.01935,0,0,0.0035))



        parameters =  cv2.aruco.DetectorParameters()
        Adetector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected_img_points = Adetector.detectMarkers(gray)

        if len(corners) < 3:
            print("There are missing some arucos")
            return False, frame, -1, -1, -1, -1

        # Calculate reference frame using centers of detected markers
        if len(corners) >= 3:
            
            marker_centers = [np.mean(marker_corners[0], axis=0) for marker_corners in corners]
            frame_origin_index = np.where(ids == 0)[0][0]
            marker_centers[0][0] = int(np.mean(corners[frame_origin_index][0,:,0]))
            marker_centers[0][1] = int(np.mean(corners[frame_origin_index][0,:,1]))
            frame_x_index = np.where(ids == 1)[0][0]
            marker_centers[1][0] = int(np.mean(corners[frame_x_index][0,:,0]))
            marker_centers[1][1] = int(np.mean(corners[frame_x_index][0,:,1]))
            frame_y_index = np.where(ids == 2)[0][0]
            marker_centers[2][0] = int(np.mean(corners[frame_y_index][0,:,0]))
            marker_centers[2][1] = int(np.mean(corners[frame_y_index][0,:,1]))

            vx_2d = marker_centers[1] - marker_centers[0]
            vy_2d = marker_centers[1]- marker_centers[2]
            point1_x_axis = zed_point_cloud.get_value(int(marker_centers[0][0]), int(marker_centers[0][1]))[1][:3]
            point2_x_axis = zed_point_cloud.get_value(int(marker_centers[1][0]), int(marker_centers[1][1]))[1][:3]
            point_y_axis = zed_point_cloud.get_value(int(marker_centers[2][0]), int(marker_centers[2][1]))[1][:3]


            vx_2d, vy_2d = orthogonalize(vx_2d, vy_2d)  # Ensure v1 and v2 are orthogonal
            # Compute the center of the reference frame
            mk_ctr = [0,0]
            m1 = vx_2d[1] / vx_2d[0] if vx_2d[0] != 0 else 100000
            m2 = vy_2d[1] / vy_2d[0] if vy_2d[0] != 0 else 100000

            b1 = marker_centers[0][1] - m1 * marker_centers[0][0]
            b2 = marker_centers[2][1] - m2 * marker_centers[2][0]
            mk_ctr[0] =(b2- b1)/(m1 - m2)
            mk_ctr[1] = m1 * mk_ctr[0] + b1
            print(mk_ctr)
            center_3d = zed_point_cloud.get_value(int(mk_ctr[0]), int(mk_ctr[1]))[1][:3]

            frame_x_axis, frame_y_axis, frame_z_axis =  XYZaxis_from_OXY(center_3d, point1_x_axis, point_y_axis)
            normal_vector_2d = np.cross(vx_2d, vy_2d)
            normal_vector_2d = normalize_vector(normal_vector_2d)
            vx_2d = normalize_vector(vx_2d)
            vy_2d = normalize_vector(vy_2d)


            # Draw reference frame
            length = 50
            end_point_x = tuple(np.round(mk_ctr + length * vx_2d).astype(int))
            end_point_y = tuple(np.round(mk_ctr + length * vy_2d).astype(int))
            #end_point_z = tuple(np.round(mk_ctr + length * normal_vector_2d).astype(int))

            cv2.arrowedLine(frame, tuple(np.round(mk_ctr).astype(int)), end_point_x, (0, 0, 255), 2)
            cv2.arrowedLine(frame, tuple(np.round(mk_ctr).astype(int)), end_point_y, (0, 255, 0), 2)
            #cv2.arrowedLine(frame, tuple(np.round(mk_ctr).astype(int)), end_point_z, (0, 0, 255), 2)

        # Draw detected markers

        return True, frame,frame_x_axis, frame_y_axis, frame_z_axis, center_3d


class Object:
    def __init__(self, x, y, z, confidence, class_id, tracker_id=None):
        self.x = x
        self.y = y
        self.z = z
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = tracker_id

    def to_dict(self):
        return [{
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "tracker_id": self.tracker_id
        }]
    
    def __repr__(self):
        return f"Object(x={self.x}, y={self.y}, z={self.z}, confidence={self.confidence}, class_id={self.class_id}, tracker_id={self.tracker_id})"


class ObjectDetection:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device:", self.device)
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

        ##There is the attribute color here, but i let in on default
        self.box_annotator = BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.8)

    def load_model(self):
        model_path = os.path.join('.','Data_seg','train49', 'weights', 'best.pt')
        assert os.path.exists(model_path), "Model file does not exist at {}".format(model_path)
        model = YOLO(model_path)  # load a custom model
        if model is not None:
            print("Model created successfully.")
        else:
            print("Error: Failed to create model.")
        model.fuse()
        return model

    def detect(self, frame, point_cloud, tranform_matrix):
        self.results = self.model.track(frame, persist = True, conf = 0.75)
        detected_objects = []
        for result in self.results:
            result.cpu().numpy()
            num_detected_objects = result.__len__()
            if num_detected_objects == 0:
                print('WARNING: No objects detected!')
  
            grasp_obj_index = 0
            if num_detected_objects > 0:
                obj_size = []
                obj_depth = []
                # Check for closest object
                for i in range(num_detected_objects):
                    print("ENTREI NO RESULT", i)
                    good_points = 0
                    # Get object centroid
                    mask_pixels = np.array(result.masks.xy[i], dtype=np.int32)
                    img_mask = np.zeros_like(frame, dtype=np.uint8)
                    cv2.fillPoly(img_mask, [mask_pixels], 255)
                    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
                    M = cv2.moments(mask_pixels)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
                    # Calculate object mean surface point XYZ
                    depth = 0
                    z = 0
                    for j in range(-2, 2):
                        for k in range(-2, 2): 
                            point_cloud_value =  point_cloud.get_value(cX + j, cY + k)[1][:3]
                            zed2aruco_point = transform_point(point_cloud_value, tranform_matrix)
                            #print("o ponto em relação ao aruco está em: ", zed2aruco_point)
                            if math.isfinite(point_cloud_value[2]):
                                depth += zed2aruco_point[2]
                                z = depth
                                good_points += 1
                    # Check DEPTH
                    if good_points > 0:
                        depth /= good_points
                        z /= good_points
                        obj_depth.append(depth)
                    else:
                        z = 100000
                        obj_depth.append(10000)
                    
                    # Extract bounding box and other attributes here
                    confidence = result.boxes.conf[i].item() if result.boxes.conf[i].item() is not None else None
                    class_id = int(result.boxes.cls[i].item()) if result.boxes.cls[i].item()is not None else None
                    tracker_id = result.boxes.id[i].item() if result.boxes.id[i].item() is not None else None
                    detected_objects.append(Object(zed2aruco_point[0], zed2aruco_point[1], z, confidence, class_id, tracker_id))
        return detected_objects
    
    


    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []
        # Extract detections for person class
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            
            result.cpu().numpy()

        # Setup detections for visualization
        detections = Detections( 
                    #x1 y1 x2 y2
                    xyxy=results.boxes.xyxy.cpu().numpy(),
                    confidence=results.boxes.conf.cpu().numpy(),
                    class_id=results.boxes.cls.cpu().numpy().astype(int),
                    )


        # Format custom labels
        self.labels = [f"{tracker_id}{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"                    
        for  _,_,confidence, class_id, tracker_id
        in detections]
        # Annotate and display frame
        '''
        for bbox in detections.xyxy:
            mask = self.apply_grabcut(frame, bbox)
            frame = cv2.add(frame, mask)
        '''
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        return frame

    def get_grasp_object(self, image, zed, zed_point_cloud, tranform_m):
        for result in self.results:
            result.cpu().numpy()
            detected_objects = result.__len__()
            grasp_obj_index = 0
            if detected_objects > 0:
                obj_size = []
                obj_depth = []
                # Check for closest object
                for i in range(detected_objects):
                    good_points = 0
                    # Get object centroid
                    mask_pixels = np.array(result.masks.xy[i], dtype=np.int32)
                    img_mask = np.zeros_like(image, dtype=np.uint8)
                    cv2.fillPoly(img_mask, [mask_pixels], 255)
                    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
                    M = cv2.moments(mask_pixels)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])  
                    # Calculate object mean surface point XYZ
                    depth = 0
                    x = 0
                    y = 0
                    z = 0
                    for j in range(-2, 2):
                        for k in range(-2, 2): 
                            point_cloud_value =  zed_point_cloud.get_value(cX + j, cY + k)[1][:3]
                            zed2aruco_point = transform_point(point_cloud_value, tranform_m)
                            #print("o ponto em relação ao aruco está em: ", zed2aruco_point)
                            if math.isfinite(point_cloud_value[2]):
                                x += zed2aruco_point[0]
                                y += zed2aruco_point[1]
                                depth += point_cloud_value[2]
                                z = depth
                                good_points += 1
                    # Check DEPTH
                    if good_points > 0:
                        #depth /= good_points
                        z /= good_points
                        y /= good_points
                        x /= good_points
                        depth = math.sqrt(x * x + y * y )
                        print("Para a deteção ", i, " tenho a seguinte pointcloud:")
                        print("X: ", x, " Y: ", y, " Z: ", z)
                        obj_depth.append(depth)
                    else:
                        obj_depth.append(10000)
                    # Check SIZE
                    obj_size.append(cv2.countNonZero(img_mask))
                
                # Determine the object to grasp
                obj_weight = []
                for i in range(detected_objects):
                    # Normalize object size
                    obj_size[i] /= max(obj_size)  
                    # Calculate total object weight
                    obj_weight.append( obj_depth[i] - obj_size[i]*100 )       # CHANGE SIZE WEIGHT HERE
                print("O vetor das distancias é: ", obj_depth)
                grasp_obj_index = obj_depth.index(min(obj_depth))

                # Get closest object Mask
                mask_pixels = np.array(result.masks.xy[grasp_obj_index], dtype=np.int32)
                obj_mask = np.zeros_like(image)
                cv2.fillPoly(obj_mask, [mask_pixels], (0, 0, 255))

                # Find centroid by moments
                M = cv2.moments(mask_pixels)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])   
                # Draw centroid
                cv2.circle(image, (cX, cY), 5, (0, 255, 0), 5)

                # Find best fit rectangle
                rect = cv2.minAreaRect(mask_pixels)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                # Draw rectangle
                cv2.drawContours(image, [box], 0, (0, 255, 0), 3) 

                # Get closest object detectyion donfidence
                conf = result.boxes.conf[grasp_obj_index].cpu().numpy() * 100
                for i in range(len(obj_depth)):
                    obj_id = result.boxes.id[i] if result.boxes.id is not None else 'N/A'
                    confidence = result.boxes.conf[i] * 100 if result.boxes.conf is not None else 'N/A'

                    if obj_id != 'N/A':
                        obj_id = obj_id.cpu().numpy()
                    if confidence != 'N/A':
                        confidence = confidence.cpu().numpy()

                    print(f"Object ID: {obj_id}, Distance: {obj_depth[i]}, Confidence: {confidence}")


                return cX, cY, obj_mask, box, conf





class ZED:

    def __init__(self):
        self.zed = sl.Camera()
        depth_for_display = sl.Mat()


        # Set configuration parameters
        self.init = sl.InitParameters()

        self.init.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init.camera_resolution = sl.RESOLUTION.HD720
        self.init.coordinate_units = sl.UNIT.CENTIMETER
        #self.init.depth_stabilization = 50

        self.status = self.zed.open(self.init)

        if(self.status != sl.ERROR_CODE.SUCCESS):
            print('ZED ERROR: {}.'.format(repr(self.status)))
            return -1
        self.runtime = sl.RuntimeParameters()

    def close(self):
        self.zed.close()

    def capture(self, zed_image: sl.Mat, point_cloud: sl.Mat, normal_map: sl.Mat, depth_image: sl.Mat):
        if(self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS):
            self.zed.retrieve_image(zed_image, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            self.zed.retrieve_measure(normal_map, sl.MEASURE.NORMALS)
            self.zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
            return True
        else:
            print("Error in image acquisition")
            return False


    def open(self):
        err = self.zed.open(self.init)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            zed.close()
            exit(1)




if __name__ == "__main__":


    args = parse_arguments()
    detector = ObjectDetection(capture_index=args.webcam_index)
    aruco = ArUco()
    zed = ZED()    

        # Prepare new image size to retrieve half-resolution images
    image_size = zed.zed.get_camera_information().camera_configuration.resolution
    image_size.width = image_size.width #/ 2
    image_size.height = image_size.height # / 2

    image_zed = sl.Mat(image_size.width, image_size.height)  
    zed_point_cloud = sl.Mat(image_size.width, image_size.height)
    zed_normal_map = sl.Mat()
    depth_image = sl.Mat(image_size.width, image_size.height)

    broker = 'broker.hivemq.com'  # Public broker for demonstration
    port = 1883
    topic = "robotic_arm/object_detection"

    pub = Publisher(broker, port, topic)

    arucoDict = cv2.aruco.getPredefinedDictionary(aruco.ARUCO_DICT[aruco.aruco_type])
    while True:
        start_time = time()
        
    
        if zed.capture(image_zed, zed_point_cloud, zed_normal_map, depth_image) :

            # Retrieve the left image
            frame = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_RGBA2RGB)
            success, frame, frame_x_axis, frame_y_axis, frame_z_axis, center_3d = aruco.ref_estimation(frame, aruco.ARUCO_DICT[aruco.aruco_type])
            if success:
                transform_M = calculate_transform_matrix(frame_x_axis, frame_y_axis, frame_z_axis, center_3d)

                #results = detector.model.track(frame, persist = True, conf = 0.75)      
                objects = detector.detect(frame, zed_point_cloud, transform_M)
                #detector.results  = results
                '''
                for result in results:
                    result.cpu().numpy()
                    detected_objects = result.__len__()
                    if detected_objects == 0:
                        print('WARNING: No objects detected!')
                '''
                detected_objects = len(objects)

                frame = detector.results[0].plot()
                print("objetos detetados: ", detected_objects)
                #if(detected_objects > 0 and success):
                    #cX, cY, obj_mask, box, conf = detector.get_grasp_object(frame, zed, zed_point_cloud, transform_M)

                    # Highlight the object to grasp in the original image
                    #frame = cv2.add(frame, obj_mask)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 3)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Detection', frame)
            object_data = object_data2pub(objects, pub)
        else:
            break



    cv2.destroyAllWindows()
