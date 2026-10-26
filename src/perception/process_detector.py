"""RGB-only worker bridge, keeping perception dependencies out of the policy env."""
import base64
from collections import deque
import io
import json
from pathlib import Path
import queue
import subprocess
import threading

import numpy as np

from src.perception.visual_evidence import VisualEvidence


class ProcessVisualDetector:
    """Sequential episode inference in a dedicated Python environment.

    No shell, pickle, simulator handle or policy state crosses this boundary.
    Failure/timeout raises; there is no heuristic or CPU fallback.
    """
    def __init__(self, python_executable, model_dir, device="cuda", timeout_seconds=120):
        self.timeout = timeout_seconds
        self._responses = queue.Queue()
        self._stderr = deque(maxlen=40)
        self._lock = threading.Lock()
        self._closed = False
        root = Path(__file__).resolve().parents[2]
        command = [str(Path(python_executable).resolve()), str(root / "scripts/memory_detector_worker.py"),
                   "--model-dir", str(Path(model_dir).resolve()), "--device", device]
        self._process = subprocess.Popen(command, cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        try:
            response = self._receive()
            if response.get("status") != "ready":
                raise RuntimeError(f"Detector worker did not initialize: {response}")
            self.provenance = response["provenance"]
        except BaseException:
            self.close()
            raise

    def _read_stdout(self):
        for line in self._process.stdout:
            self._responses.put(line)
        self._responses.put(None)

    def _read_stderr(self):
        for line in self._process.stderr:
            self._stderr.append(line.rstrip())

    def _receive(self):
        try:
            line = self._responses.get(timeout=self.timeout)
        except queue.Empty as error:
            raise TimeoutError("RGB detector worker timed out") from error
        if line is None:
            raise RuntimeError("RGB detector worker exited: " + "\n".join(self._stderr))
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError("RGB detector worker failed: " + response["error"])
        return response

    def detect(self, step, images):
        with self._lock:
            if self._closed:
                raise RuntimeError("RGB detector worker is closed")
            buffer = io.BytesIO()
            np.savez(buffer, **images)
            request = dict(step=step, rgb_npz=base64.b64encode(buffer.getvalue()).decode("ascii"))
            try:
                self._process.stdin.write(json.dumps(request) + "\n")
                self._process.stdin.flush()
                evidence = VisualEvidence.from_dict(self._receive()["evidence"])
                evidence.validate(step, images)
                if evidence.model_id != self.provenance["model_id"] or evidence.revision != self.provenance["revision"]:
                    raise ValueError("Worker prediction provenance changed")
                return evidence
            except BaseException:
                self.close()
                raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._process.stdin:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass  # preserve the original worker error during cleanup
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
