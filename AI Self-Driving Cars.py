import tensorflow as tf
import cv2
import numpy as np

# Load pre-trained model
model = tf.saved_model.load('ssd_mobilenet_v2/saved_model')

# Function to run inference
def detect_objects(image):
    input_tensor = tf.convert_to_tensor([image])
    detections = model(input_tensor)

    return detections

# Load an image
image = cv2.imread('test_image.jpg')
detections = detect_objects(image)

# Process detections and visualize
for i in range(int(detections.pop('num_detections'))):
    bbox = detections['detection_boxes'][0][i].numpy()
    class_id = int(detections['detection_classes'][0][i].numpy())
    score = detections['detection_scores'][0][i].numpy()

    if score > 0.5:
        # Draw bounding box and label
        cv2.rectangle(image, (int(bbox[1]*image.shape[1]), int(bbox[0]*image.shape[0])),
                      (int(bbox[3]*image.shape[1]), int(bbox[2]*image.shape[0])), (0, 255, 0), 2)
        cv2.putText(image, f'ID: {class_id}, Score: {score:.2f}', 
                    (int(bbox[1]*image.shape[1]), int(bbox[0]*image.shape[0]-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import heapq

def a_star(start, goal, graph):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(start, goal)

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor, cost in graph[current].items():
            tentative_g_score = g_score[current] + cost
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

# Example graph and usage
graph = {
    (0, 0): {(1, 0): 1, (0, 1): 1},
    (1, 0): {(1, 1): 1},
    (0, 1): {(1, 1): 1},
    (1, 1): {}
}
start = (0, 0)
goal = (1, 1)

path = a_star(start, goal, graph)
print("Path:", path)


class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.previous_error = 0

    def compute(self, setpoint, measured_value):
        error = setpoint - measured_value
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

# Example usage
pid = PID(1.0, 0.1, 0.01)
setpoint = 100
measured_value = 90
control_signal = pid.compute(setpoint, measured_value)
print("Control Signal:", control_signal)


import carla

# Connect to the Carla server
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

# Load the world
world = client.get_world()

# Get the blueprint library and spawn a vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Apply a control command to the vehicle
vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))

# Run the simulation for a few seconds
import time
time.sleep(5)

# Destroy the vehicle actor
vehicle.destroy()
