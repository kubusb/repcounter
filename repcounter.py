import cv2
import mediapipe as mp
import numpy as np
import time
from enum import Enum

class ExerciseType(Enum):
    JUMPING_JACK = 1
    HAMMER_CURL = 2

class RepCounter:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Exercise state variables
        self.current_exercise = ExerciseType.JUMPING_JACK
        self.rep_count = {
            ExerciseType.JUMPING_JACK: 0,
            ExerciseType.HAMMER_CURL: 0
        }
        
        # State tracking for jumping jacks
        self.jumping_jack_state = "down"  # "down" or "up"
        self.jj_confidence_threshold = 0.7
        
        # State tracking for hammer curls
        self.hammer_curl_state = "down"  # "down" or "up"
        self.hc_confidence_threshold = 0.7
        
        # Cooldown to prevent multiple counts
        self.last_rep_time = time.time()
        self.rep_cooldown = 0.5  # seconds
    
    def detect_jumping_jack(self, landmarks):
        """Detect if a jumping jack is being performed based on pose landmarks."""
        # Get relevant landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Check confidence levels
        if (left_shoulder.visibility < self.jj_confidence_threshold or
            right_shoulder.visibility < self.jj_confidence_threshold or
            left_wrist.visibility < self.jj_confidence_threshold or
            right_wrist.visibility < self.jj_confidence_threshold or
            left_ankle.visibility < self.jj_confidence_threshold or
            right_ankle.visibility < self.jj_confidence_threshold):
            return
        
        # Calculate horizontal distances
        wrist_distance = abs(left_wrist.x - right_wrist.x)
        ankle_distance = abs(left_ankle.x - right_ankle.x)
        
        # Check jumping jack state transitions
        current_time = time.time()
        # Up position: hands are up and feet are apart
        if (wrist_distance > 0.5 and ankle_distance > 0.3 and 
            self.jumping_jack_state == "down" and 
            current_time - self.last_rep_time > self.rep_cooldown):
            self.jumping_jack_state = "up"
        
        # Down position: hands are down and feet are together
        elif (wrist_distance < 0.3 and ankle_distance < 0.2 and 
              self.jumping_jack_state == "up" and 
              current_time - self.last_rep_time > self.rep_cooldown):
            self.jumping_jack_state = "down"
            self.rep_count[ExerciseType.JUMPING_JACK] += 1
            self.last_rep_time = current_time
    
    def detect_hammer_curl(self, landmarks):
        """Detect if a hammer curl is being performed based on pose landmarks."""
        # Get relevant landmarks (focusing on right arm for simplicity)
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # Check confidence levels
        if (right_shoulder.visibility < self.hc_confidence_threshold or
            right_elbow.visibility < self.hc_confidence_threshold or
            right_wrist.visibility < self.hc_confidence_threshold):
            return
        
        # Calculate vertical position of wrist relative to elbow
        wrist_above_elbow = right_wrist.y < right_elbow.y
        
        # Check hammer curl state transitions
        current_time = time.time()
        # Up position: wrist is above elbow
        if (wrist_above_elbow and 
            self.hammer_curl_state == "down" and 
            current_time - self.last_rep_time > self.rep_cooldown):
            self.hammer_curl_state = "up"
            
        # Down position: wrist is below elbow
        elif (not wrist_above_elbow and 
              self.hammer_curl_state == "up" and 
              current_time - self.last_rep_time > self.rep_cooldown):
            self.hammer_curl_state = "down"
            self.rep_count[ExerciseType.HAMMER_CURL] += 1
            self.last_rep_time = current_time
    
    def process_frame(self, frame):
        """Process a video frame and return the frame with annotations."""
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and get pose landmarks
        results = self.pose.process(image_rgb)
        
        # Convert back to BGR for OpenCV
        annotated_frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Draw the pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame, 
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Detect exercises based on current mode
            if self.current_exercise == ExerciseType.JUMPING_JACK:
                self.detect_jumping_jack(results.pose_landmarks.landmark)
            elif self.current_exercise == ExerciseType.HAMMER_CURL:
                self.detect_hammer_curl(results.pose_landmarks.landmark)
        
        # Add text overlays for rep counts and current exercise
        cv2.putText(
            annotated_frame,
            f"Current Exercise: {self.current_exercise.name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            annotated_frame,
            f"Jumping Jacks: {self.rep_count[ExerciseType.JUMPING_JACK]}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            annotated_frame,
            f"Hammer Curls: {self.rep_count[ExerciseType.HAMMER_CURL]}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return annotated_frame
    
    def toggle_exercise(self):
        """Toggle between exercise types."""
        if self.current_exercise == ExerciseType.JUMPING_JACK:
            self.current_exercise = ExerciseType.HAMMER_CURL
        else:
            self.current_exercise = ExerciseType.JUMPING_JACK
    
    def reset_counts(self):
        """Reset all rep counts."""
        self.rep_count = {
            ExerciseType.JUMPING_JACK: 0,
            ExerciseType.HAMMER_CURL: 0
        }


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize rep counter
    counter = RepCounter()
    
    print("Fitness Rep Counter started!")
    print("Press 'e' to toggle between exercises")
    print("Press 'r' to reset counts")
    print("Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Process frame
        annotated_frame = counter.process_frame(frame)
        
        # Display the resulting frame
        cv2.imshow('Fitness Rep Counter', annotated_frame)
        
        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            counter.toggle_exercise()
        elif key == ord('r'):
            counter.reset_counts()
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()