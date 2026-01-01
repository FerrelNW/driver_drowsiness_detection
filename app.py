# app.py
from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera
import os

app = Flask(__name__)

# Check if model exists
MODEL_PATH = 'models/final_best_model.keras'
if not os.path.exists(MODEL_PATH):
    print("="*70)
    print("⚠️  WARNING: Model file not found!")
    print(f"   Expected location: {MODEL_PATH}")
    print("   Please place your trained model in the 'models' folder")
    print("="*70)

# Initialize camera with deep learning model
print("="*70)
print("🚀 STARTING FLASK SERVER")
print("="*70)

global_camera = VideoCamera(model_path=MODEL_PATH)

print("✅ Flask server ready!")
print("   Access at: http://localhost:5000")
print("="*70)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

def gen(camera):
    """Video streaming generator"""
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(gen(global_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status_feed')
def status_feed():
    """
    Status API endpoint
    Returns current drowsiness detection status
    """
    status = global_camera.get_status()
    return jsonify(status)

@app.route('/stats')
def stats():
    """
    Statistics endpoint (optional)
    """
    stats = {
        'total_predictions': global_camera.total_predictions,
        'drowsy_detections': global_camera.drowsy_detections,
        'buffer_status': f"{len(global_camera.frame_buffer)}/{global_camera.SEQUENCE_LENGTH}",
        'model_loaded': global_camera.model is not None
    }
    return jsonify(stats)

if __name__ == '__main__':
    # Run Flask server
    print("\n🌐 Starting web server...")
    print("   URL: http://0.0.0.0:5000")
    print("   Press CTRL+C to stop\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set False for production
        threaded=True
    )