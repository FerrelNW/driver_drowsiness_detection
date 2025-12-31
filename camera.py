# camera.py
import cv2
import numpy as np
from collections import deque
import sys

print("Initializing VideoCamera...")

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("✗ Error: Cannot open webcam")
            sys.exit(1)
        
        try:
            import mediapipe as mp
            print(f"✓ MediaPipe {mp.__version__} imported")
            
            # Access face_mesh through mp.solutions
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✓ FaceMesh initialized successfully")
            
        except Exception as e:
            print(f"✗ Error initializing MediaPipe: {e}")
            print("\nDebug info:")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Buffer untuk sequence data
        self.sequence_length = 30
        self.feature_buffer = deque(maxlen=self.sequence_length)
        
        # State Variables
        self.is_drowsy = False
        self.status_text = "Active"
        self.color = (0, 255, 0)

    def __del__(self):
        if hasattr(self, 'video'):
            self.video.release()
        print("✓ Camera released")

    def calculate_ear(self, landmarks, indices):
        """Calculate Eye Aspect Ratio"""
        # Unpack indices
        p1, p2, p3, p4, p5, p6 = indices
        
        # Calculate distances
        vertical1 = np.linalg.norm(landmarks[p2] - landmarks[p6])
        vertical2 = np.linalg.norm(landmarks[p3] - landmarks[p5])
        horizontal = np.linalg.norm(landmarks[p1] - landmarks[p4])
        
        # Calculate EAR
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            print("✗ Failed to read frame from camera")
            return None

        # Flip and convert frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with FaceMesh
        results = self.face_mesh.process(rgb_frame)
        
        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convert landmarks to pixel coordinates
                landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])

                # Eye indices for MediaPipe Face Mesh
                left_eye_indices = [362, 385, 387, 263, 373, 380]  # Left eye
                right_eye_indices = [33, 160, 158, 133, 153, 144]  # Right eye

                # Calculate EAR for both eyes
                left_ear = self.calculate_ear(landmarks, left_eye_indices)
                right_ear = self.calculate_ear(landmarks, right_eye_indices)
                avg_ear = (left_ear + right_ear) / 2.0

                # Simple drowsiness detection
                if avg_ear < 0.22:
                    self.is_drowsy = True
                    self.status_text = "MENGANTUK / DROWSY"
                    self.color = (0, 0, 255)  # Red
                else:
                    self.is_drowsy = False
                    self.status_text = "SADAR / ALERT"
                    self.color = (0, 255, 0)  # Green
                
                # Add text to frame
                cv2.putText(frame, f"Status: {self.status_text}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2)
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # No face detected
            self.is_drowsy = False
            cv2.putText(frame, "TIDAK ADA WAJAH TERDETEKSI", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Encode frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            return jpeg.tobytes()
        return None