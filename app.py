import cv2
from flask import Flask, Response, render_template, jsonify

from detector import DrowsinessDetector

app = Flask(__name__)
detector = DrowsinessDetector()

_camera = None


def get_camera():
    global _camera
    if _camera is None:
        _camera = cv2.VideoCapture(0)
    return _camera


def gen_frames():
    cam = get_camera()
    while True:
        success, frame = cam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # mirror, feels more natural for a self-facing camera
        frame = detector.process_frame(frame)

        ok, buffer = cv2.imencode('.jpg', frame)
        if not ok:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    return jsonify(detector.get_stats())


@app.route('/reset')
def reset():
    """Optional: clear counters without restarting the server."""
    detector.blink_timestamps.clear()
    detector.yawn_timestamps.clear()
    return jsonify({"reset": True})


if __name__ == '__main__':
    # threaded=True lets the MJPEG stream and the /status polling be served concurrently
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
