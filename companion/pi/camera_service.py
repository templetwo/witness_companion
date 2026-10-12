# Codex, gpt-6-astra, Jetson seat, 2026-10-11 23:32:40 EDT
"""Camera-only HTTP snapshot service. Never imports the robot SDK or mode code."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time


class Frames:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg, self.captured, self.sequence = None, 0, 0

    def update(self, jpeg):
        with self.lock:
            self.jpeg, self.captured = jpeg, time.time()
            self.sequence += 1

    def latest(self):
        with self.lock:
            return self.jpeg, self.captured, self.sequence


def handler_for(frames, peer):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.client_address[0] != peer:
                self.send_error(403)
                return
            if self.path not in ('/snapshot.jpg', '/?action=snapshot'):
                self.send_error(404)
                return
            jpeg, captured, seq = frames.latest()
            if jpeg is None or not 0 <= time.time() - captured <= 1:
                self.send_error(503, 'fresh camera frame unavailable')
                return
            self.connection.settimeout(2)
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(jpeg)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Witness-Capture-Time', str(captured))
            self.send_header('X-Witness-Sequence', str(seq))
            self.end_headers()
            try:
                self.wfile.write(jpeg)
            except OSError:
                pass

        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--peer', required=True)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--device', default='/dev/video0')
    args = parser.parse_args()
    import cv2
    capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError('camera could not open')
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frames = Frames()
    server = ThreadingHTTPServer((args.host, args.port), handler_for(frames, args.peer))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.05)
                continue
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                frames.update(encoded.tobytes())
    finally:
        server.shutdown()
        server.server_close()
        capture.release()


if __name__ == '__main__':
    main()
