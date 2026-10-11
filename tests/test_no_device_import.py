"""Importing companion (and every submodule) must touch no device library.

Runs in a fresh, isolated subprocess so other tests' imports cannot pollute
sys.modules.
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN = ["cv2", "sounddevice", "serial", "pyaudio", "faster_whisper", "whisper",
             "piper", "requests", "numpy", "luma", "smbus", "spidev", "RPi", "gpiozero",
             "yaml", "witness_brain"]

PROBE = r"""
import sys
sys.path.insert(0, %r)
import pkgutil, importlib
import companion
names = [m.name for m in pkgutil.walk_packages(companion.__path__, "companion.")]
for n in names:
    importlib.import_module(n)
bad = sorted(n for n in sys.modules for f in %r if n == f or n.startswith(f + "."))
print("MODULES:%%d" %% len(names))
print("LOADED:" + ",".join(bad))
""" % (ROOT, FORBIDDEN)


class NoDeviceOnImport(unittest.TestCase):
    def test_import_touches_no_device_library(self):
        out = subprocess.run([sys.executable, "-I", "-c", PROBE], cwd=ROOT,
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = out.stdout.splitlines()
        self.assertIn("LOADED:", lines, out.stdout)
        n = [int(x.split(":")[1]) for x in lines if x.startswith("MODULES:")][0]
        self.assertGreaterEqual(n, 12)  # every submodule really was imported


if __name__ == "__main__":
    unittest.main()
