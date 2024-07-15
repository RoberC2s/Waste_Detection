import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/'
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

class Kalman_filter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def predict(self, x, y):
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return int(predicted[0]), int(predicted[1])

kf = Kalman_filter()
cap = cv2.VideoCapture(0)

positions = []  # List to store the positions of the detected circle
pox = []
poy = []

while(True):
    time.sleep(0.1)
    ret, captured_frame = cap.read()
    if not ret:
        break
    output_frame = captured_frame.copy()

    captured_frame_bgr = cv2.cvtColor(captured_frame, cv2.COLOR_BGRA2BGR)
    captured_frame_bgr = cv2.medianBlur(captured_frame_bgr, 5)
    captured_frame_lab = cv2.cvtColor(captured_frame_bgr, cv2.COLOR_BGR2Lab)
    captured_frame_lab_red = cv2.inRange(captured_frame_lab, np.array([20, 160, 120]), np.array([190, 255, 255]))
    captured_frame_lab_red = cv2.GaussianBlur(captured_frame_lab_red, (5, 5), 2, 2)
    circles = cv2.HoughCircles(captured_frame_lab_red, cv2.HOUGH_GRADIENT, 1, captured_frame_lab_red.shape[0] / 8, param1=100, param2=18, minRadius=5, maxRadius=100)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        cv2.circle(output_frame, center=(circles[0, 0], circles[0, 1]), radius=circles[0, 2], color=(0, 255, 0), thickness=2)
        pred_x, pred_y = kf.predict(circles[0, 0], circles[0, 1])
        cv2.circle(output_frame, center=(pred_x, pred_y), radius=circles[0, 2], color=(255, 0, 0), thickness=4)

        # Store the position
        positions.append((circles[0, 0],circles[0, 1]))
        pox.append(circles[0, 0])
        poy.append(circles[0, 1])

    cv2.imshow('frame', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pred_pox = []
pred_poy = []
# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Plotting the positions
if positions:
    # Extract x and y coordinates
    pred_x, pred_y = kf.predict(pox[-1], poy[-1])
    print('yo', pred_x, pred_y)
    pred_pox.append(pred_x)
    pred_poy.append(pred_y)
    for i in range(20):
        pred_x, pred_y = kf.predict(pred_pox[-1], pred_poy[-1])
        pred_pox.append(pred_x)
        pred_poy.append(pred_y)
    x_coords = pox
    print(x_coords)
    y_coords = poy
    print(y_coords)

    # Create the plot
    plt.plot(pred_pox, pred_poy, label='Predicted positions')
    plt.plot(x_coords, y_coords, label='Measured positions', linestyle='dashed')

    # Add labels and title
    plt.xlabel('Y values')
    plt.ylabel('X values')
    plt.title('Plot of Kalman Filter Predicted Positions and Measured Positions')
    plt.legend()
    plt.show()
