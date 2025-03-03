import cv2
import numpy as np
import time
from enum import Enum

class ExerciseType(Enum):
    JUMPING_JACK = 1
    HAMMER_CURL = 2

class RepCounter:
    def __init__(self):
        # Exercise state variables
        self.current_exercise = ExerciseType.JUMPING_JACK
        self.rep_count = {
            ExerciseType.JUMPING_JACK: 0,
            ExerciseType.HAMMER_CURL: 0
        }
        
        # Initialize background subtractor for motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, 
            varThreshold=50, 
            detectShadows=False
        )
        
        # State tracking for exercises
        self.jumping_jack_state = "down"  # "down" or "up"
        self.hammer_curl_state = "down"   # "down" or "up"
        
        # Motion tracking variables
        self.prev_motion_value = 0
        self.motion_threshold = 15000  # Adjust based on testing
        self.upper_region_threshold = 10000  # For hammer curl detection
        
        # Cooldown to prevent multiple counts
        self.last_rep_time = time.time()
        self.rep_cooldown = 1.0  # seconds
        
        # Frame dimensions
        self.frame_width = 0
        self.frame_height = 0
        self.is_initialized = False
    
    def initialize_dimensions(self, frame):
        """Initialize frame dimensions for region calculations."""
        self.frame_height, self.frame_width = frame.shape[:2]
        self.is_initialized = True
    
    def detect_jumping_jack(self, frame):
        """Detect jumping jacks based on overall motion in the frame."""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Calculate total motion
        motion_value = np.sum(fg_mask) / 255  # Normalize
        
        # Detect state transitions based on motion peaks and valleys
        current_time = time.time()
        
        # Detect a significant motion peak (arms/legs spreading out)
        if (motion_value > self.motion_threshold and 
            self.prev_motion_value < self.motion_threshold and
            self.jumping_jack_state == "down" and 
            current_time - self.last_rep_time > self.rep_cooldown):
            self.jumping_jack_state = "up"
        
        # Detect motion returning to normal (arms/legs coming together)
        elif (motion_value < self.motion_threshold and 
              self.prev_motion_value > self.motion_threshold and
              self.jumping_jack_state == "up" and 
              current_time - self.last_rep_time > self.rep_cooldown):
            self.jumping_jack_state = "down"
            self.rep_count[ExerciseType.JUMPING_JACK] += 1
            self.last_rep_time = current_time
        
        # Update previous motion value
        self.prev_motion_value = motion_value
        
        # Return the foreground mask for visualization
        return fg_mask
    
    def detect_hammer_curl(self, frame):
        """Detect hammer curls based on motion in the upper part of the frame."""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Define upper region of interest (where arms would be)
        upper_region = fg_mask[0:int(self.frame_height * 0.4), :]
        
        # Calculate motion in the upper region
        upper_motion = np.sum(upper_region) / 255
        
        # Detect state transitions based on motion in upper region
        current_time = time.time()
        
        # Detect upward motion
        if (upper_motion > self.upper_region_threshold and 
            self.prev_motion_value < self.upper_region_threshold and
            self.hammer_curl_state == "down" and 
            current_time - self.last_rep_time > self.rep_cooldown):
            self.hammer_curl_state = "up"
        
        # Detect downward motion
        elif (upper_motion < self.upper_region_threshold and 
              self.prev_motion_value > self.upper_region_threshold and
              self.hammer_curl_state == "up" and 
              current_time - self.last_rep_time > self.rep_cooldown):
            self.hammer_curl_state = "down"
            self.rep_count[ExerciseType.HAMMER_CURL] += 1
            self.last_rep_time = current_time
        
        # Update previous motion value
        self.prev_motion_value = upper_motion
        
        # Return the foreground mask for visualization
        return fg_mask
    
    def process_frame(self, frame):
        """Process a video frame and return the frame with annotations."""
        # Initialize frame dimensions if not already done
        if not self.is_initialized:
            self.initialize_dimensions(frame)
        
        # Make a copy of the frame for annotations
        annotated_frame = frame.copy()
        
        # Process frame based on current exercise
        if self.current_exercise == ExerciseType.JUMPING_JACK:
            motion_mask = self.detect_jumping_jack(frame)
        else:
            motion_mask = self.detect_hammer_curl(frame)
        
        # Create smaller display of the motion mask
        mask_display = cv2.resize(motion_mask, (160, 120))
        mask_display = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
        
        # Overlay the motion mask in the corner of the frame
        annotated_frame[10:130, 10:170] = mask_display
        
        # Add text overlays for rep counts and current exercise
        cv2.putText(
            annotated_frame,
            f"Current Exercise: {self.current_exercise.name}",
            (180, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            annotated_frame,
            f"Jumping Jacks: {self.rep_count[ExerciseType.JUMPING_JACK]}",
            (180, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            annotated_frame,
            f"Hammer Curls: {self.rep_count[ExerciseType.HAMMER_CURL]}",
            (180, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Add calibration instructions
        cv2.putText(
            annotated_frame,
            "Stand in front of camera and keep background clear",
            (10, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        return annotated_frame
    
    def toggle_exercise(self):
        """Toggle between exercise types."""
        if self.current_exercise == ExerciseType.JUMPING_JACK:
            self.current_exercise = ExerciseType.HAMMER_CURL
        else:
            self.current_exercise = ExerciseType.JUMPING_JACK
        
        # Reset background model when switching exercises
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, 
            varThreshold=50, 
            detectShadows=False
        )
    
    def reset_counts(self):
        """Reset all rep counts."""
        self.rep_count = {
            ExerciseType.JUMPING_JACK: 0,
            ExerciseType.HAMMER_CURL: 0
        }
        
        # Reset background model
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, 
            varThreshold=50, 
            detectShadows=False
        )


def calibrate_thresholds(cap):
    """Interactive calibration process to set motion thresholds."""
    print("CALIBRATION MODE")
    print("Stand in position for 3 seconds...")
    
    # Create background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=100, 
        varThreshold=50, 
        detectShadows=False
    )
    
    # Wait for background model to initialize
    for i in range(30):
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame during calibration")
            return 15000, 10000  # Default values
        
        bg_subtractor.apply(frame)
        cv2.putText(
            frame,
            f"Calibrating background: {i+1}/30",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.imshow('Calibration', frame)
        cv2.waitKey(1)
    
    # Measure baseline motion
    baseline_values = []
    print("Now perform ONE jumping jack...")
    
    for i in range(60):  # 2 seconds at 30fps
        ret, frame = cap.read()
        if not ret:
            break
        
        fg_mask = bg_subtractor.apply(frame)
        motion_value = np.sum(fg_mask) / 255
        baseline_values.append(motion_value)
        
        cv2.putText(
            frame,
            f"Perform ONE jumping jack: {i+1}/60",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Show the mask
        mask_display = cv2.resize(fg_mask, (160, 120))
        mask_display = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
        frame[10:130, 10:170] = mask_display
        
        cv2.imshow('Calibration', frame)
        cv2.waitKey(1)
    
    # Calculate thresholds based on measured motion
    max_motion = max(baseline_values)
    motion_threshold = max_motion * 0.7  # 70% of max motion
    upper_region_threshold = motion_threshold * 0.6  # 60% of motion threshold
    
    print(f"Calibrated motion threshold: {motion_threshold}")
    print(f"Calibrated upper region threshold: {upper_region_threshold}")
    
    return motion_threshold, upper_region_threshold


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set resolution (optional, adjust based on your webcam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Calibrate motion thresholds
    motion_threshold, upper_region_threshold = calibrate_thresholds(cap)
    
    # Initialize rep counter
    counter = RepCounter()
    counter.motion_threshold = motion_threshold
    counter.upper_region_threshold = upper_region_threshold
    
    print("Fitness Rep Counter started!")
    print("Press 'e' to toggle between exercises")
    print("Press 'r' to reset counts")
    print("Press 'c' to recalibrate")
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
        elif key == ord('c'):
            motion_threshold, upper_region_threshold = calibrate_thresholds(cap)
            counter.motion_threshold = motion_threshold
            counter.upper_region_threshold = upper_region_threshold
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()