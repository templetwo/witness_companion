# Codex, gpt-6-astra, Jetson seat, 2026-10-12 10:05 EDT
import http.client
from http.server import ThreadingHTTPServer
import tempfile
import threading
import time
import unittest

from companion.pi.camera_service import Frames, handler_for
from witness_mcp.live import CameraSnapshot


class CameraHandoverTests(unittest.TestCase):
    def setUp(self):
        self.frames = Frames()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.frames, '127.0.0.1'))
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
        self.thread.start()
        self.addCleanup(self.finish)

    def finish(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1)

    def request(self, path='/snapshot.jpg'):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=1)
        try:
            conn.request('GET', path)
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def test_missing_and_stale_frames_are_unavailable(self):
        self.assertEqual(self.request()[0], 503)
        self.frames.update(b'\xff\xd8test\xff\xd9')
        self.frames.captured = time.time()-2
        self.assertEqual(self.request()[0], 503)

    def test_fresh_snapshot_has_sequence_and_no_cache(self):
        jpeg = b'\xff\xd8test\xff\xd9'
        self.frames.update(jpeg)
        status, headers, data = self.request()
        self.assertEqual((status, data), (200, jpeg))
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(headers['X-Witness-Sequence'], '1')
        self.frames.update(jpeg)
        self.assertEqual(self.request()[1]['X-Witness-Sequence'], '2')
        self.assertEqual(self.request('/')[0], 404)

    def test_unapproved_peer_cannot_read(self):
        self.server.RequestHandlerClass = handler_for(self.frames, '192.0.2.1')
        self.assertEqual(self.request()[0], 403)

    def test_camera_adapter_saves_valid_frame_and_rejects_oversize(self):
        with tempfile.TemporaryDirectory() as directory:
            camera = CameraSnapshot('http://127.0.0.1:%d/snapshot.jpg' % self.server.server_port, directory)
            self.frames.update(b'\xff\xd8test\xff\xd9')
            self.assertTrue(camera.snapshot(1)['ok'])
            self.frames.update(b'\xff\xd8' + b'x' * (1024 * 1024) + b'\xff\xd9')
            self.assertFalse(camera.snapshot(1)['ok'])
