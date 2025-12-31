from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera

app = Flask(__name__)

# Inisialisasi kamera sebagai objek global
# Ini penting agar variable 'is_drowsy' tersinkronisasi antara video feed dan status feed
global_camera = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    # Menggunakan global_camera agar instance-nya sama
    return Response(gen(global_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status_feed')
def status_feed():
    # Endpoint ini dipanggil oleh JavaScript setiap 0.5 detik
    # Mengembalikan status apakah sopir mengantuk atau tidak
    return jsonify({'drowsy': global_camera.is_drowsy})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)