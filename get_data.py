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
import pyzed.sl as sl
from scipy.signal import savgol_filter
from collections import defaultdict
from datetime import datetime

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

def round_vector(vector, decimals=1):
    return [round(num, decimals) for num in vector]

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
    u2 = v2 - np.dot(v2, u1) / np.dot(u1, u1) * u1
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

def update_speed(detected_objects, tracker_id, new_speed_x, new_speed_y, new_speed_z):
    for obj in detected_objects:
        if obj.tracker_id == tracker_id:
            obj.speed_x = new_speed_x
            obj.speed_y = new_speed_y
            obj.speed_z = new_speed_z
            break

def update_point(detected_objects, tracker_id, new_key, new_x, new_y, new_z):
    for obj in detected_objects:
        if obj.tracker_id == tracker_id:
            obj.tracker_id = new_key
            obj.x = new_x
            obj.y = new_y
            obj.z = new_z
            break

class Kalman_filter:
    def __init__(self):
        # Initialize the Kalman Filter with 6 dynamic parameters and 3 measurement parameters
        self.kf = cv2.KalmanFilter(6, 3)
        
        # Measurement matrix (maps the measured values to the state variables)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0, 0]], np.float32)
        
        # Transition matrix (describes the evolution of the state)
        self.kf.transitionMatrix = np.array([[1, 0, 0, 1, 0, 0],
                                             [0, 1, 0, 0, 1, 0],
                                             [0, 0, 1, 0, 0, 1],
                                             [0, 0, 0, 1, 0, 0],
                                             [0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 1]], np.float32)
        
        # Process noise covariance matrix
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.07

        # Measurement noise covariance matr

        self.kf.measurementNoiseCov = np.array([[0.066767, -0.001617, 0.135032],
                                                [-0.001617,  0.013646,  0.040181],
                                                [0.135032,  0.040181,  0.499050]], dtype=np.float32)
        #self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1.5

    def correct(self, x, y, z):
        # Create the measurement matrix from the input 3D point
        measured = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)]])
        
        # Correct the state of the Kalman filter with the measured values
        self.kf.correct(measured)
        
        # Extract the corrected position (x, y, z)
        corrected = self.kf.statePost
        x, y, z = corrected[0], corrected[1], corrected[2]
        
        return float(x), float(y), float(z)

    def predict(self, x, y, z):
        n = 1
        measured = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)]])
        c = self.kf.correct(measured)
        # Predict the next state
        for i in range(n):
            
            predicted = self.kf.predict()
            print(i, ' e o predict é', predicted)

        print("predicted = ", predicted)
        # Extract the predicted position (x, y, z)
        x, y, z = predicted[0], predicted[1], predicted[2]
        
        return float(c[0]), float(c[1]), float(c[2]), float(x), float(y), float(z)



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

        # Define camera matrix and distortion coefficients
        matrix_coefficients = np.array(((1406.57, 0, 980.817), (0, 1407.67, 618.202), (0, 0, 1)))
        distortion_coefficients = np.array((-0.16597, 0.01935, 0, 0, 0.0035))

        # Create detector parameters
        parameters = cv2.aruco.DetectorParameters_create()

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        

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
            vy_2d = marker_centers[2]- marker_centers[0]
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

            frame_x_axis, frame_y_axis, frame_z_axis =  XYZaxis_from_OXY(center_3d, point2_x_axis, point_y_axis)
            normal_vector_2d = np.cross(vx_2d, vy_2d)
            normal_vector_2d = normalize_vector(normal_vector_2d)
            vx_2d = normalize_vector(vx_2d)
            vy_2d = normalize_vector(vy_2d)


            # Draw reference frame
            length = 250
            end_point_x = tuple(np.round(mk_ctr + length * vx_2d).astype(int))
            end_point_y = tuple(np.round(mk_ctr + length * vy_2d).astype(int))
            #end_point_z = tuple(np.round(mk_ctr + length * normal_vector_2d).astype(int))

            cv2.arrowedLine(frame, tuple(np.round(mk_ctr).astype(int)), end_point_x, (0, 0, 255), 2)
            cv2.arrowedLine(frame, tuple(np.round(mk_ctr).astype(int)), end_point_y, (0, 255, 0), 2 )



        # Draw detected markers

        return True, frame,frame_x_axis, frame_y_axis, frame_z_axis, center_3d


class Object:
    def __init__(self, x, y, z, confidence, class_id, tracker_id, timestamp):
        #x_list = []
        self.x = x#_list.append(x)
        y_list = []
        self.y = y#_list.append(y)
        z_list = []
        self.z = z#_list.append(z)
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = tracker_id
        self.speed_x = None
        self.speed_y = None
        self.speed_z = None
        self.idf_time = timestamp

    def to_dict(self):
        return [{
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "tracker_id": self.tracker_id,
            "speed_x": self.speed_x,
            "speed_y": self.speed_y,
            "speed_z": self.speed_z,
            "idf_time":self.idf_time,
        }]
    
    def __repr__(self):
        return f"Object(x={self.x}, y={self.y}, z={self.z}, confidence={self.confidence}, class_id={self.class_id}, tracker_id={self.tracker_id}, speed_x={self.speed_x}, speed_y={self.speed_y}, speed_z={self.speed_z} idf_time={self.idf_time})"


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
        model_path = os.path.join('.','Data_seg','train50', 'weights', 'best.pt')
        assert os.path.exists(model_path), "Model file does not exist at {}".format(model_path)
        model = YOLO(model_path)  # load a custom model
        if model is not None:
            print("Model created successfully.")
        else:
            print("Error: Failed to create model.")
        model.fuse()
        return model

    def detect(self, frame, point_cloud, tranform_matrix, track_history, track_history_pixels):

        self.results = self.model.track(frame, persist = True, conf = 0.55)
        # Get the boxes and track IDs

        detected_objects = []
        detected_objects_to_fitler01 = []

        detected_objects_to_fitler02 = []

        detected_objects_to_fitler03 = []
        detected_objects_to_fitler04 = []
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
                    good_points = 0
                    # Get object centroid
                    mask_pixels = np.array(result.masks.xy[i], dtype=np.int32)
                    img_mask = np.zeros_like(frame, dtype=np.uint8)
                    cv2.fillPoly(img_mask, [mask_pixels], 255)
                    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
                    M = cv2.moments(mask_pixels)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cX, cY), 4, (255, 0,0), -1)
                    # Calculate object mean surface point XYZ
                    depth = 0
                    x = 0
                    y = 0
                    z = 0
                    for j in range(-2, 2):
                        for k in range(-2, 2): 
                            point_cloud_value =  point_cloud.get_value(cX + j, cY + k)[1][:3]
                            zed2aruco_point = transform_point(point_cloud_value, tranform_matrix)

                            #print("o ponto em relação ao aruco está em: ", zed2aruco_point)
                            if math.isfinite(point_cloud_value[2]):
                                depth += point_cloud_value[2]
                                x += zed2aruco_point[0]
                                y += zed2aruco_point[1]
                                z += zed2aruco_point[2]
                                good_points += 1
                    # Check DEPTH
                    if good_points > 0:
                        depth /= good_points
                        x /= good_points
                        y /= good_points
                        z /= good_points
                        obj_depth.append(depth)
                    else:
                        z = 100000
                        obj_depth.append(10000)
                    
                    # Extract bounding box and other attributes here
                    confidence = result.boxes.conf[i].item() if result.boxes.conf[i].item() is not None else None
                    class_id = int(result.boxes.cls[i].item()) if result.boxes.cls[i].item()is not None else None# Ensure result.boxes.id is not None and contains the index i
                    if result.boxes.id is not None and result.boxes.id[i] is not None:
                        tracker_id = result.boxes.id[i].item()
                    else:
                            tracker_id = None

                    if tracker_id is not None:
                        track = track_history[tracker_id]
                        track.append((x, y, z))
                        track_pixels = track_history_pixels[tracker_id]
                        track_pixels.append((float(cX),float(cY)))  # x, y center point pixels
                        if len(track_pixels) > 30:  # retain 90 tracks for 90 frames
                            track_pixels.pop(0)
                        if len(track) > 30:  # retain 90 tracks for 90 frames
                            track.pop(0)
                        points = np.array(track_pixels)[:, :2].astype(np.int32).reshape((-1, 1, 2))       
                        for i in range(1, len(points)):
                            cv2.line(frame, tuple(points[i - 1][0]), tuple(points[i][0]), (0, 255, 0), 2)   
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        detected_objects.append(Object(x, y, z, confidence, class_id, tracker_id, timestamp))
                        detected_objects_to_fitler01.append(Object(x, y, z, confidence, class_id, tracker_id, timestamp))

                        detected_objects_to_fitler02.append(Object(x, y, z, confidence, class_id, tracker_id, timestamp))

                        detected_objects_to_fitler03.append(Object(x, y, z, confidence, class_id, tracker_id, timestamp))

                        detected_objects_to_fitler04.append(Object(x, y, z, confidence, class_id, tracker_id, timestamp))
        return detected_objects, detected_objects_to_fitler01, detected_objects_to_fitler02, detected_objects_to_fitler03, detected_objects_to_fitler04
    
    


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
                #cv2.circle(image, (cX, cY), 5, (0, 255, 0), 5)

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

        self.init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        self.init.camera_resolution = sl.RESOLUTION.HD1080
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
    
    track_history = defaultdict(lambda: [])
    track_history_pixels = defaultdict(lambda: [])
        # Prepare new image size to retrieve half-resolution images
    image_size = zed.zed.get_camera_information().camera_configuration.resolution
    image_size.width = image_size.width #/ 2
    image_size.height = image_size.height#  / 2

    image_zed = sl.Mat(image_size.width, image_size.height)  
    zed_point_cloud = sl.Mat(image_size.width, image_size.height)
    zed_normal_map = sl.Mat()
    depth_image = sl.Mat(image_size.width, image_size.height)

    broker = 'broker.hivemq.com'  # Public broker for demonstration
    port = 1883
    topic = "robotic_arm/object_detection"

    pub = Publisher(broker, port, topic)
    KF = Kalman_filter()
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
                objects, objects_01, objects_02, objects_03, objects_04 = detector.detect(frame, zed_point_cloud, transform_M, track_history, track_history_pixels)
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
            if success:
                pos_x = []
                pos_y = []
                pos_z = []
                success_2 = False
                speed_dict = defaultdict(lambda: {'speed_x': [], 'speed_y': []})
                for key, points in track_history.items():
                    if(len(points) < 3):                            #The amount of lectures used to calculate the speed,
                        
                        break                                       #and after the average speed
                    if(len(points) < 15):
                        wl = len(points)
                    else:
                        wl = 15
                    success_2 = True
                    i = 0
                    last_points = points[-15:]
                    for point in last_points:
                        
                        pos_x.append(np.round(point[0], 1))
                        pos_y.append(np.round(point[1], 1))
                        pos_z.append(np.round(point[2], 1))
                        i += 1
                        if i > 15 or i > (len(points) - 1): break
                    
                    #No filter
                    pos_x = round_vector(pos_x, decimals=1)
                    pos_y = round_vector(pos_y, decimals=1)
                    pos_z = round_vector(pos_z, decimals=1)


                    #Kalman Filter 
                    #nx, ny, nz = KF.correct(pos_x[-1], pos_y[-1], pos_z[-1])
                    #print('nx', nx, 'ny, ', ny, ', nz ', nz)
                    
                    cx, cy, cz, px, py, pz = KF.predict(pos_x[-1], pos_y[-1], pos_z[-1])
                        #px, py, pz = KF.correct(px, py, pz)

                    print('in ', i, ' I have: ', px, py, pz)
                    
                    velocities_x = np.diff(pos_x) * fps         #To pass to meter/second

                    velocities_y = np.diff(pos_y) * fps

                    velocities_z = np.diff(pos_z) * fps

                    #Filtro com poly = 0
                    smtp_x01 = savgol_filter(pos_x, window_length=wl, polyorder=0)
                    smtp_x01 = round_vector(smtp_x01, decimals=1)
                    smtp_y01 = savgol_filter(pos_y, window_length=wl, polyorder=0)
                    smtp_y01 = round_vector(smtp_y01, decimals=1)
                    smtp_z01 = savgol_filter(pos_z, window_length=wl, polyorder=0)
                    smtp_z01 = round_vector(smtp_z01, decimals=1)

                    
                    velocities_x_smooth_01 = np.diff(smtp_x01) * fps
                    velocities_y_smooth_01 = np.diff(smtp_y01) * fps
                    velocities_z_smooth_01 = np.diff(smtp_z01) * fps

                    #Filtro com poly = 1 
                    smtp_x02 = savgol_filter(pos_x, window_length=wl, polyorder=1)
                    smtp_x02 = round_vector(smtp_x02, decimals=1)
                    smtp_y02 = savgol_filter(pos_y, window_length=wl, polyorder=1)
                    smtp_y02 = round_vector(smtp_y02, decimals=1)
                    smtp_z02 = savgol_filter(pos_z, window_length=wl, polyorder=1)
                    smtp_z02 = round_vector(smtp_z02, decimals=1)

                    velocities_x_smooth_02 = np.diff(smtp_x02) * fps
                    velocities_y_smooth_02 = np.diff(smtp_y02) * fps
                    velocities_z_smooth_02 = np.diff(smtp_z02) * fps

                    #Filtro com poly = 2
                    smtp_x03 = savgol_filter(pos_x, window_length=wl, polyorder=2)
                    smtp_x03 = round_vector(smtp_x03, decimals=1)
                    smtp_y03 = savgol_filter(pos_y, window_length=wl, polyorder=2)
                    smtp_y03 = round_vector(smtp_y03, decimals=1)
                    smtp_z03 = savgol_filter(pos_z, window_length=wl, polyorder=2)
                    smtp_z03 = round_vector(smtp_z03, decimals=1)

                    velocities_x_smooth_03 = np.diff(smtp_x03) * fps
                    velocities_y_smooth_03 = np.diff(smtp_y03) * fps
                    velocities_z_smooth_03 = np.diff(smtp_z03) * fps
                    #smtp_y01 = savgol_filter(pos_y, window_length=5, polyorder=2)
                    # Calculate velocities using the smoothed positions

                    
                    update_point(objects_01, key, key + 0.1, smtp_x01[-1], smtp_y01[-1], smtp_z01[-1])
                    update_point(objects_02, key, key + 0.2, smtp_x02[-1], smtp_y02[-1], smtp_z02[-1])
                    update_point(objects_03, key, key + 0.3, smtp_x03[-1], smtp_y03[-1], smtp_z03[-1])
                    update_point(objects_04, key, key + 0.4, np.round(cx, 1), np.round(cy,1), np.round(cz,1))
                    
                    update_speed(objects, key, np.round(velocities_x[-1],1), round((velocities_y[-1]),1), round((velocities_z[-1]),1))

                    update_speed(objects_01, key + 0.1, np.round(velocities_x_smooth_01[-1],1), round((velocities_y_smooth_01[-1]),1), round((velocities_z_smooth_01[-1]),1))
                    update_speed(objects_02, key + 0.2, np.round(velocities_x_smooth_02[-1],1), round((velocities_y_smooth_02[-1]),1), round((velocities_z_smooth_02[-1]),1))
                    update_speed(objects_03, key + 0.3, np.round(velocities_x_smooth_03[-1],1), round((velocities_y_smooth_03[-1]),1), round((velocities_z_smooth_03[-1]),1))
                    update_speed(objects_04, key + 0.4, px, py, pz)
                                
                #if success_2:    
                 #   cv2.circle(frame, (int(smtp_x01[-1]), int(smtp_y01[-1])), 5, (255, 0, 0), -1)
                object_data = object_data2pub(objects, pub)
                object_data = object_data2pub(objects_01, pub)
                object_data = object_data2pub(objects_02, pub)
                object_data = object_data2pub(objects_03, pub)
                object_data = object_data2pub(objects_04, pub)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Detection', frame)
            
                
        else:
            break



    cv2.destroyAllWindows()
