# camera.py
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import numpy as np
from collections import deque
import sys
import tensorflow as tf
import tf_keras
from tf_keras.applications import MobileNetV2
from tf_keras.layers import TimeDistributed, GRU, Dense, Dropout, GlobalAveragePooling2D, Input
from tf_keras.models import Model

print("🚀 Initializing Deep Learning Drowsiness Detection System...")

class VideoCamera:
    def __init__(self, model_path='models/final_best_model.keras'):
        # 1. Setup Webcam
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("❌ Error: Cannot open webcam")
            sys.exit(1)
        print("✅ Webcam initialized")

        # 2. Setup Model Architecture (Manual Rebuild - Sesuai Training Kaggle)
        try:
            print(f"📦 TensorFlow version: {tf.__version__}")
            print("🔨 Rebuilding model architecture manually to match Training Code...")
            
            # --- KONFIGURASI SESUAI KODE TRAINING KAMU ---
            self.IMG_SIZE = 128
            self.SEQUENCE_LENGTH = 12
            self.TARGET_FPS = 6
            # ---------------------------------------------
            
            # A. Input Definition
            # Shape: (Sequence Length, Height, Width, Channels)
            input_seq = Input(shape=(self.SEQUENCE_LENGTH, self.IMG_SIZE, self.IMG_SIZE, 3))
            
            # B. Base Model (MobileNetV2)
            # Sesuai training: include_top=False, weights="imagenet"
            base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3))
            base_model.trainable = False 
            
            # C. Sequence Processing (TimeDistributed Wrapper)
            x = TimeDistributed(base_model)(input_seq)
            x = TimeDistributed(GlobalAveragePooling2D())(x)
            
            # D. RNN Block (GRU)
            # Sesuai training: x = GRU(64, dropout=0.3)(x)
            # Note: return_sequences=False adalah default di Keras
            x = GRU(64, dropout=0.3, return_sequences=False)(x)
            
            # Sesuai training: x = Dense(64, activation="relu")(x)
            x = Dense(64, activation='relu')(x)
            
            # Sesuai training: x = Dropout(0.4)(x)
            x = Dropout(0.4)(x)
            
            # Output: output = Dense(1, activation="sigmoid")(x)
            outputs = Dense(1, activation='sigmoid')(x)
            
            # E. Gabungkan menjadi Model Utuh
            self.model = Model(inputs=input_seq, outputs=outputs)
            
            # Setup Preprocessing MobileNetV2
            from tf_keras.applications.mobilenet_v2 import preprocess_input
            self.preprocess_input = preprocess_input
            
            print("✅ Model architecture built successfully!")
            
            # 3. Load Weights
            if os.path.exists(model_path):
                print(f"📥 Attempting to load weights from {model_path}...")
                try:
                    # Load weights only (Bypass struktur config yang error)
                    self.model.load_weights(model_path)
                    print("✅ Weights loaded successfully! Model is SMART now.")
                except Exception as w_err:
                    print(f"⚠️ WARNING: Gagal load weights ({w_err})")
                    print("   Pastikan file .keras tidak korup.")
                    print("   System running in DUMMY MODE (Web tetap jalan).")
            else:
                print("⚠️ Model file not found. Running in Dummy Mode.")
                
        except Exception as e:
            print(f"❌ Critical Error setting up model: {e}")
            pass
        
        # 3. Setup Face Detector (Dlib)
        try:
            import dlib
            landmark_path = "shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(landmark_path):
                print(f"⚠️ Warning: {landmark_path} not found.")
            
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(landmark_path)
            print("✅ Dlib face detector initialized")
            
        except Exception as e:
            print(f"❌ Error initializing Dlib: {e}")
            self.detector = None
        
        # 4. State Variables & Buffers
        self.frame_buffer = deque(maxlen=self.SEQUENCE_LENGTH)
        self.frame_counter = 0
        self.prediction_buffer = deque(maxlen=5)
        
        self.is_drowsy = False
        self.drowsy_confidence = 0.0
        self.status_text = "Active"
        self.color = (0, 255, 0)
        self.face_detected = False
        self.total_predictions = 0
        self.drowsy_detections = 0
        
        print("✅ Camera system ready!")

    def __del__(self):
        if hasattr(self, 'video'):
            self.video.release()

    def extract_mouth_region(self, frame, landmarks):
        # ... (Kode ekstraksi sama seperti sebelumnya, tidak ada perubahan) ...
        h, w = frame.shape[:2]
        mouth_points = landmarks[48:68]
        
        x_min, y_min = mouth_points.min(axis=0)
        x_max, y_max = mouth_points.max(axis=0)
        
        mouth_width = x_max - x_min
        mouth_height = y_max - y_min
        
        margin_x = int(mouth_width * 0.4)
        margin_y = int(mouth_height * 0.5)
        
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(w, x_max + margin_x)
        y_max = min(h, y_max + margin_y)
        
        mouth_roi = frame[y_min:y_max, x_min:x_max]
        if mouth_roi.size == 0: return None
        
        roi_h, roi_w = mouth_roi.shape[:2]
        if roi_w > roi_h:
            new_w = self.IMG_SIZE
            new_h = int(self.IMG_SIZE * roi_h / roi_w)
        else:
            new_h = self.IMG_SIZE
            new_w = int(self.IMG_SIZE * roi_w / roi_h)
        
        mouth_roi = cv2.resize(mouth_roi, (new_w, new_h))
        
        pad_h = (self.IMG_SIZE - new_h) // 2
        pad_w = (self.IMG_SIZE - new_w) // 2
        
        mouth_roi = cv2.copyMakeBorder(
            mouth_roi, pad_h, self.IMG_SIZE - new_h - pad_h,
            pad_w, self.IMG_SIZE - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return mouth_roi

    def get_landmarks(self, frame):
        if self.detector is None: return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        if len(faces) == 0: return None
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        landmarks = self.predictor(gray, face)
        return np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])

    def predict_drowsiness(self):
        if len(self.frame_buffer) < self.SEQUENCE_LENGTH:
            return 0.0, False
        
        sequence = np.array(list(self.frame_buffer))
        sequence = np.expand_dims(sequence, axis=0)
        
        # PREDICT (Verbose 0 agar terminal bersih)
        prediction = self.model.predict(sequence, verbose=0)[0][0]
        
        self.prediction_buffer.append(prediction)
        smoothed_prediction = np.mean(self.prediction_buffer)
        
        # Threshold: > 0.5 = Yawn (Mengantuk)
        is_drowsy = smoothed_prediction > 0.5
        return smoothed_prediction, is_drowsy

    def get_frame(self):
        success, frame = self.video.read()
        if not success: return None
        
        frame = cv2.flip(frame, 1)
        self.frame_counter += 1
        
        # Skip frame logic (30 FPS -> 6 FPS)
        if (self.frame_counter % (30 // self.TARGET_FPS)) == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = self.get_landmarks(rgb_frame)
            
            if landmarks is not None:
                self.face_detected = True
                mouth_roi = self.extract_mouth_region(rgb_frame, landmarks)
                
                if mouth_roi is not None:
                    input_data = self.preprocess_input(mouth_roi.astype(np.float32))
                    self.frame_buffer.append(input_data)
                    
                    if len(self.frame_buffer) == self.SEQUENCE_LENGTH:
                        confidence, is_drowsy = self.predict_drowsiness()
                        self.is_drowsy = is_drowsy
                        self.drowsy_confidence = confidence
                        self.total_predictions += 1
                        
                        if is_drowsy:
                            self.drowsy_detections += 1
                            self.status_text = "MENGANTUK / DROWSY"
                            self.color = (0, 0, 255)
                        else:
                            self.status_text = "WASPADA / ALERT"
                            self.color = (0, 255, 0)
                    else:
                        self.status_text = f"Buffering... ({len(self.frame_buffer)}/{self.SEQUENCE_LENGTH})"
                        self.color = (255, 255, 0)
                    
                    # Kotak Mulut
                    pts = landmarks[48:68]
                    x_min, y_min = pts.min(axis=0)
                    x_max, y_max = pts.max(axis=0)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), self.color, 2)
            else:
                self.face_detected = False
                self.status_text = "WAJAH TIDAK TERDETEKSI"
                self.color = (0, 255, 255)
        
        cv2.putText(frame, f"Status: {self.status_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color, 2)
        if self.face_detected:
            cv2.putText(frame, f"Conf: {self.drowsy_confidence:.1%}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes() if ret else None

    def get_status(self):
        return {
            'drowsy': self.is_drowsy,
            'confidence': float(self.drowsy_confidence),
            'status': self.status_text,
            'face_detected': self.face_detected,
            'buffer_size': len(self.frame_buffer),
            'total_predictions': self.total_predictions,
            'drowsy_count': self.drowsy_detections
        }