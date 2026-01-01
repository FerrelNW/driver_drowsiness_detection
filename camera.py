# camera.py
import cv2
import numpy as np
from collections import deque
import sys
import os

print("🚀 Initializing Deep Learning Drowsiness Detection System...")

class VideoCamera:
    def __init__(self, model_path='models/final_best_model.keras'):
        """
        Initialize camera with deep learning model
        No MAR calculation needed - Pure visual learning!
        """
        
        # Initialize webcam
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("❌ Error: Cannot open webcam")
            sys.exit(1)
        
        print("✅ Webcam initialized")
        
        # Load deep learning model
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            
            print(f"📦 TensorFlow version: {tf.__version__}")
            
            if not os.path.exists(model_path):
                print(f"❌ Error: Model not found at {model_path}")
                print("Please place your trained model in the 'models' folder")
                sys.exit(1)
            
            self.model = load_model(model_path)
            self.preprocess_input = preprocess_input
            print(f"✅ Model loaded from {model_path}")
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Output shape: {self.model.output_shape}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Initialize face detection (for ROI extraction only, not for MAR!)
        try:
            import dlib
            
            # Download shape predictor if needed
            if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
                print("📥 Downloading face landmark model...")
                os.system("wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                os.system("bzip2 -d shape_predictor_68_face_landmarks.dat.bz2 -f")
            
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            print("✅ Dlib face detector initialized")
            
        except Exception as e:
            print(f"❌ Error initializing Dlib: {e}")
            sys.exit(1)
        
        # Model configuration
        self.IMG_SIZE = 128
        self.SEQUENCE_LENGTH = 12  # Same as training
        self.TARGET_FPS = 6
        
        # Frame buffer for sequence
        self.frame_buffer = deque(maxlen=self.SEQUENCE_LENGTH)
        self.frame_counter = 0
        
        # State variables
        self.is_drowsy = False
        self.drowsy_confidence = 0.0
        self.status_text = "Initializing..."
        self.color = (255, 255, 0)  # Yellow for init
        self.face_detected = False
        
        # Statistics
        self.total_predictions = 0
        self.drowsy_detections = 0
        
        # Smoothing predictions (avoid flickering)
        self.prediction_buffer = deque(maxlen=5)  # Last 5 predictions
        
        print("✅ Camera system ready!")
        print("="*50)

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'video'):
            self.video.release()
        print("✅ Camera released")

    def extract_mouth_region(self, frame, landmarks):
        """
        Extract mouth ROI from frame
        Same as preprocessing - maintain consistency!
        """
        h, w = frame.shape[:2]
        
        # Mouth landmarks: 48-67
        mouth_points = landmarks[48:68]
        
        x_min, y_min = mouth_points.min(axis=0)
        x_max, y_max = mouth_points.max(axis=0)
        
        mouth_width = x_max - x_min
        mouth_height = y_max - y_min
        
        # Add margins (same as training)
        margin_x = int(mouth_width * 0.4)
        margin_y = int(mouth_height * 0.5)
        
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(w, x_max + margin_x)
        y_max = min(h, y_max + margin_y)
        
        # Extract ROI
        mouth_roi = frame[y_min:y_max, x_min:x_max]
        
        if mouth_roi.size == 0:
            return None
        
        # Resize with aspect ratio preservation
        roi_h, roi_w = mouth_roi.shape[:2]
        if roi_w > roi_h:
            new_w = self.IMG_SIZE
            new_h = int(self.IMG_SIZE * roi_h / roi_w)
        else:
            new_h = self.IMG_SIZE
            new_w = int(self.IMG_SIZE * roi_w / roi_h)
        
        mouth_roi = cv2.resize(mouth_roi, (new_w, new_h))
        
        # Pad to square
        pad_h = (self.IMG_SIZE - new_h) // 2
        pad_w = (self.IMG_SIZE - new_w) // 2
        
        mouth_roi = cv2.copyMakeBorder(
            mouth_roi,
            pad_h, self.IMG_SIZE - new_h - pad_h,
            pad_w, self.IMG_SIZE - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        
        return mouth_roi

    def get_landmarks(self, frame):
        """Get facial landmarks using Dlib"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get landmarks
        landmarks = self.predictor(gray, face)
        coords = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
        
        return coords

    def predict_drowsiness(self):
        """
        Predict drowsiness from frame buffer
        Pure visual - no MAR calculation!
        """
        if len(self.frame_buffer) < self.SEQUENCE_LENGTH:
            return 0.5, False  # Not enough frames yet
        
        # Prepare sequence
        sequence = np.array(list(self.frame_buffer))
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
        
        # Predict
        prediction = self.model.predict(sequence, verbose=0)[0][0]
        
        # Add to buffer for smoothing
        self.prediction_buffer.append(prediction)
        
        # Smooth prediction (average of last 5)
        smoothed_prediction = np.mean(self.prediction_buffer)
        
        # Threshold: >0.5 = Yawn/Drowsy
        is_drowsy = smoothed_prediction > 0.5
        
        return smoothed_prediction, is_drowsy

    def get_frame(self):
        """
        Capture frame, process with DL model, and return annotated frame
        """
        success, frame = self.video.read()
        if not success:
            print("❌ Failed to read frame")
            return None
        
        # Flip and prepare
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Only process every N frames (to match TARGET_FPS)
        self.frame_counter += 1
        should_process = (self.frame_counter % (30 // self.TARGET_FPS)) == 0
        
        if should_process:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face and extract landmarks
            landmarks = self.get_landmarks(rgb_frame)
            
            if landmarks is not None:
                self.face_detected = True
                
                # Extract mouth region
                mouth_roi = self.extract_mouth_region(rgb_frame, landmarks)
                
                if mouth_roi is not None:
                    # Preprocess for MobileNet
                    mouth_preprocessed = self.preprocess_input(mouth_roi.astype(np.float32))
                    
                    # Add to buffer
                    self.frame_buffer.append(mouth_preprocessed)
                    
                    # Predict if buffer is full
                    if len(self.frame_buffer) == self.SEQUENCE_LENGTH:
                        confidence, is_drowsy = self.predict_drowsiness()
                        
                        self.is_drowsy = is_drowsy
                        self.drowsy_confidence = confidence
                        self.total_predictions += 1
                        
                        if is_drowsy:
                            self.drowsy_detections += 1
                            self.status_text = "MENGANTUK / DROWSY"
                            self.color = (0, 0, 255)  # Red
                        else:
                            self.status_text = "WASPADA / ALERT"
                            self.color = (0, 255, 0)  # Green
                    else:
                        self.status_text = f"Buffering... ({len(self.frame_buffer)}/{self.SEQUENCE_LENGTH})"
                        self.color = (255, 255, 0)  # Yellow
                    
                    # Draw mouth ROI box on original frame
                    mouth_points = landmarks[48:68]
                    x_min, y_min = mouth_points.min(axis=0)
                    x_max, y_max = mouth_points.max(axis=0)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), self.color, 2)
            else:
                self.face_detected = False
                self.status_text = "WAJAH TIDAK TERDETEKSI"
                self.color = (0, 255, 255)  # Cyan
        
        # Overlay information
        cv2.putText(frame, f"Status: {self.status_text}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2)
        
        if self.face_detected:
            cv2.putText(frame, f"Confidence: {self.drowsy_confidence:.2%}", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Buffer: {len(self.frame_buffer)}/{self.SEQUENCE_LENGTH}", (20, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Statistics
        if self.total_predictions > 0:
            accuracy = (1 - self.drowsy_detections / self.total_predictions) * 100
            cv2.putText(frame, f"Predictions: {self.total_predictions} | Alert Rate: {accuracy:.1f}%", 
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Encode to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            return jpeg.tobytes()
        return None
    
    def get_status(self):
        """
        Get current status for API endpoint
        """
        return {
            'drowsy': self.is_drowsy,
            'confidence': float(self.drowsy_confidence),
            'status': self.status_text,
            'face_detected': self.face_detected,
            'buffer_size': len(self.frame_buffer),
            'total_predictions': self.total_predictions,
            'drowsy_count': self.drowsy_detections
        }