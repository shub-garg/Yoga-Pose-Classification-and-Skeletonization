import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Input and output folder paths
input_folder = 'path/to/input/folder'
output_folder = 'path/to/output/folder'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # Read the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find the pose
        result = pose.process(image_rgb)

        # Create a blank numpy array with zeros
        keypoints_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        # Draw the keypoints on the blank image
        if result.pose_landmarks:
            for landmark in result.pose_landmarks.landmark:
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                cv2.circle(keypoints_image, (x, y), 5, (255, 255, 255), -1)

        # Save the image with keypoints
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, keypoints_image)

print("Processing complete.")
