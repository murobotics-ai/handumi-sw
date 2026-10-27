"""Browser dashboard and browser-microphone voice control for ``handumi record``.

The recorder serves one page and one WebSocket on the workstation. The page
shows recorder state, gripper widths, controller tracking and camera
thumbnails, and it streams the browser's microphone back as 16 kHz PCM. The
workstation runs a closed-grammar Vosk recognizer on that audio and hands the
recorder its usual ``start``/``stop``/``restart`` commands. The operator says
"start", "stop" or "discard", and the workstation needs no microphone or
PortAudio.

Browsers only grant microphone access to secure origins, and
``http://localhost`` counts as one. The server therefore binds to localhost and
the operator reaches it through an SSH tunnel::

    ssh -L 8890:localhost:8890 user@workstation
    # then open http://localhost:8890 on the laptop
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import Any

import numpy as np

from handumi.utils.voice import (
    SAMPLE_RATE,
    VoiceCommandListener,
    VoiceUnavailableError,
    resolve_model_path,
)

log = logging.getLogger("handumi.web_dashboard")

DEFAULT_PORT = 8890
_PAGE_PATH = Path(__file__).with_name("index.html")
_STATUS_HZ = 10.0

# Spoken phrase -> recorder action. The short words are the ones the page
# advertises. The recorder's own longer phrases stay accepted.
BROWSER_COMMANDS: dict[str, str] = {
    "start": "start",
    "stop": "stop",
    "discard": "restart",
    "start recording": "start",
    "stop recording": "stop",
    "restart": "restart",
}
_UNKNOWN = "[unk]"
# Single words show up inside ordinary sentences. The recognizer folds several
# out-of-grammar words into one "[unk]", so counting them is unreliable.
# Instead the command must end the utterance ("okay stop" counts, "stop for
# lunch" does not), after at most this many unknown tokens.
_MAX_LEADING_UNKNOWN = 2
_THUMBNAIL_HZ = 2.0
_THUMBNAIL_WIDTH = 320


class BrowserVoiceListener(VoiceCommandListener):
    """Voice control fed with PCM pushed from the dashboard page.

    The recorder polls it exactly like :class:`VoiceCommandListener`. Audio
    arrives through :meth:`feed_audio` instead of a local input stream, and the
    page's buttons call :meth:`push_command` directly.
    """

    def __init__(self, *, confidence: float = 0.7, debounce_s: float = 1.5) -> None:
        super().__init__(device=None, confidence=confidence, debounce_s=debounce_s)
        self._audio: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self.device_name = "dashboard browser microphone"
        self.last_heard: dict[str, Any] | None = None
        self.audio_seen_s = 0.0

    def start(self) -> None:
        try:
            import vosk
        except ImportError as exc:
            raise VoiceUnavailableError("Browser voice control needs the `vosk` package.") from exc

        vosk.SetLogLevel(-1)
        model_path = resolve_model_path()
        try:
            self._model = vosk.Model(str(model_path))
        except Exception as exc:
            raise VoiceUnavailableError(
                f"Could not load the speech model at {model_path}: {exc}"
            ) from exc
        self._thread = threading.Thread(
            target=self._run, name="handumi_browser_voice", daemon=True
        )
        self._thread.start()
        log.info(
            "Browser voice control ready (say: start, stop, discard).",
        )

    def feed_audio(self, pcm: bytes) -> None:
        """Queue one block of mono int16 PCM at :data:`SAMPLE_RATE`."""
        self.audio_seen_s = time.monotonic()
        try:
            self._audio.put_nowait(pcm)
        except queue.Full:
            # Recognition fell behind. Dropping old audio keeps commands timely.
            try:
                self._audio.get_nowait()
            except queue.Empty:
                pass
            self._audio.put_nowait(pcm)

    def push_command(self, action: str) -> None:
        """Queue a command from a dashboard button."""
        if action not in BROWSER_COMMANDS.values():
            log.warning("Ignoring unknown dashboard command %r.", action)
            return
        log.info("Dashboard button: %s", action)
        self.last_heard = {"text": f"[button] {action}", "action": action, "t": time.time()}
        with self._lock:
            self._commands.append(action)

    def _run(self) -> None:
        import vosk

        recognizer = vosk.KaldiRecognizer(self._model, SAMPLE_RATE, self._grammar())
        recognizer.SetWords(True)
        while not self._closed.is_set():
            try:
                pcm = self._audio.get(timeout=0.2)
            except queue.Empty:
                continue
            if time.monotonic() < self._muted_until:
                recognizer.Reset()
                continue
            if recognizer.AcceptWaveform(pcm):
                self._handle_result(recognizer.Result())

    def _grammar(self) -> str:
        return json.dumps([*BROWSER_COMMANDS, _UNKNOWN])

    def _handle_result(self, raw: str) -> None:
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            return
        tokens = str(result.get("text", "")).split()
        text = " ".join(word for word in tokens if word != _UNKNOWN)
        if not text:
            return
        action = BROWSER_COMMANDS.get(text)
        leading = 0
        while leading < len(tokens) and tokens[leading] == _UNKNOWN:
            leading += 1
        command_ends_utterance = " ".join(tokens[leading:]) == text
        words = [w for w in (result.get("result") or []) if w.get("word") != _UNKNOWN]
        score = min((float(w.get("conf", 1.0)) for w in words), default=1.0)
        reason = None
        if action is None:
            reason = "not a command"
        elif not command_ends_utterance or leading > _MAX_LEADING_UNKNOWN:
            reason = "part of a longer sentence"
        elif score < self._confidence:
            reason = f"confidence {score:.2f}"
        else:
            now = time.monotonic()
            last = self._last_command
            if last is not None and last[0] == action and now - last[1] < self._debounce_s:
                reason = "repeat"
            else:
                self._last_command = (action, now)
                log.info("Voice command: %r -> %s", text, action)
                with self._lock:
                    self._commands.append(action)
        self.last_heard = {
            "text": " ".join(tokens),
            "action": None if reason else action,
            "reason": reason,
            "t": time.time(),
        }


@dataclass
class _Thumbnail:
    jpeg_b64: str = ""
    encoded_at_s: float = 0.0
    sequence: int = -1


@dataclass
class WebDashboard:
    """Serve the dashboard page and stream recorder status to every client.

    ``status`` is the recorder's terminal dashboard object. The server reads
    its state fields and reads widths, tracking and frames straight from the
    background samplers, so the page keeps updating while the recorder's main
    loop is blocked waiting for a trigger or for tracking.
    """

    status: Any
    tracker: Any
    grippers: Any
    voice: BrowserVoiceListener | None
    cameras: Sequence[Any | None]
    camera_names: Sequence[str]
    port: int = DEFAULT_PORT
    host: str = "127.0.0.1"
    _server: Any = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _thumbs: dict[str, _Thumbnail] = field(default_factory=dict, init=False, repr=False)
    _thumb_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def start(self) -> None:
        from websockets.sync.server import serve

        self._server = serve(
            self._handle,
            self.host,
            self.port,
            process_request=self._process_request,
            max_size=1 << 20,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="handumi_web_dashboard", daemon=True
        )
        self._thread.start()
        log.info(
            "Web dashboard on http://%s:%d (from a laptop: ssh -L %d:localhost:%d this-host).",
            self.host,
            self.bound_port,
            self.bound_port,
            self.bound_port,
        )

    @property
    def bound_port(self) -> int:
        """Port the server listens on, which differs from ``port`` when it is 0."""
        if self._server is None:
            return self.port
        return int(self._server.socket.getsockname()[1])

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # -- HTTP and WebSocket ------------------------------------------------

    def _process_request(self, connection, request):
        from websockets.datastructures import Headers
        from websockets.http11 import Response

        if request.path.startswith("/ws"):
            return None
        if request.path not in ("/", "/index.html"):
            return connection.respond(HTTPStatus.NOT_FOUND, "Not found\n")
        body = _PAGE_PATH.read_bytes()
        headers = Headers(
            [
                ("Content-Type", "text/html; charset=utf-8"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ]
        )
        return Response(HTTPStatus.OK.value, "OK", headers, body)

    def _handle(self, connection) -> None:
        closed = threading.Event()
        sender = threading.Thread(
            target=self._send_loop, args=(connection, closed), daemon=True
        )
        sender.start()
        try:
            for message in connection:
                if isinstance(message, bytes):
                    if self.voice is not None:
                        self.voice.feed_audio(message)
                    continue
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    continue
                command = payload.get("command")
                if command and self.voice is not None:
                    self.voice.push_command(str(command))
        except Exception:  # client went away
            pass
        finally:
            closed.set()
            sender.join(timeout=1.0)

    def _send_loop(self, connection, closed: threading.Event) -> None:
        period = 1.0 / _STATUS_HZ
        while not closed.is_set():
            try:
                connection.send(json.dumps(self.snapshot()))
            except Exception:
                closed.set()
                return
            closed.wait(period)

    # -- status ------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        status = self.status
        return {
            "time": time.time(),
            "state": getattr(status, "state", "STARTING"),
            "detail": getattr(status, "detail", ""),
            "task": getattr(status, "task", ""),
            "dataset": getattr(status, "dataset", ""),
            "episode": getattr(status, "episode", 0),
            "total": getattr(status, "total", ""),
            "fps": getattr(status, "fps", 30),
            "current_frames": getattr(status, "current_frames", 0),
            "saved_episodes": getattr(status, "saved_episodes", 0),
            "saved_frames": getattr(status, "saved_frames", 0),
            "runtime_s": getattr(status, "program_runtime_s", 0.0),
            "grippers": self._gripper_status(),
            "tracking": self._tracking_status(),
            "cameras": self._camera_status(),
            "voice": self._voice_status(),
        }

    def _gripper_status(self) -> dict[str, Any]:
        latest = getattr(self.grippers, "latest", None)
        sample = latest() if callable(latest) else None
        if sample is None:
            return {"available": False}
        widths = sample.widths
        age_s = max(0.0, (time.monotonic_ns() - int(sample.sample_time_ns)) / 1e9)
        return {
            "available": True,
            "left_mm": float(widths.left_mm),
            "right_mm": float(widths.right_mm),
            "left_ticks": int(widths.left_ticks),
            "right_ticks": int(widths.right_ticks),
            "age_s": age_s,
        }

    def _tracking_status(self) -> dict[str, Any]:
        try:
            sample = self.tracker.latest()
        except Exception:
            return {"connected": False}
        received_ns = int(getattr(sample, "pc_monotonic_ns", 0) or 0)
        age_s = (time.monotonic_ns() - received_ns) / 1e9 if received_ns else None
        return {
            "connected": bool(getattr(sample, "connected", False)),
            "streaming": bool(getattr(sample, "streaming", False)),
            "left": bool(sample.left_tracked),
            "right": bool(sample.right_tracked),
            "hmd": bool(getattr(sample, "hmd_tracked", False)),
            "age_s": age_s,
        }

    def _camera_status(self) -> list[dict[str, Any]]:
        out = []
        now_ns = time.monotonic_ns()
        for name, camera in zip(self.camera_names, self.cameras):
            entry: dict[str, Any] = {"name": name, "enabled": camera is not None}
            if camera is not None:
                try:
                    sample = camera.sample_at()
                except Exception as exc:
                    entry.update(ok=False, error=str(exc))
                else:
                    entry.update(
                        ok=True,
                        age_s=max(0.0, (now_ns - int(sample.capture_time_ns)) / 1e9),
                        jpeg=self._thumbnail(name, sample),
                    )
            out.append(entry)
        return out

    def _thumbnail(self, name: str, sample) -> str:
        with self._thumb_lock:
            thumb = self._thumbs.setdefault(name, _Thumbnail())
            now = time.monotonic()
            if sample.sequence == thumb.sequence or now - thumb.encoded_at_s < 1.0 / _THUMBNAIL_HZ:
                return thumb.jpeg_b64
            thumb.jpeg_b64 = _encode_jpeg(sample.image)
            thumb.encoded_at_s = now
            thumb.sequence = sample.sequence
            return thumb.jpeg_b64

    def _voice_status(self) -> dict[str, Any]:
        voice = self.voice
        if voice is None:
            return {"enabled": False}
        audio_age = time.monotonic() - voice.audio_seen_s if voice.audio_seen_s else None
        return {
            "enabled": True,
            "phrases": ["start", "stop", "discard"],
            "last_heard": voice.last_heard,
            "audio_age_s": audio_age,
            "muted": time.monotonic() < voice._muted_until,
        }


def _encode_jpeg(image: np.ndarray) -> str:
    import cv2

    frame = np.asarray(image)
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    height, width = frame.shape[:2]
    if width > _THUMBNAIL_WIDTH:
        frame = cv2.resize(
            frame, (_THUMBNAIL_WIDTH, int(height * _THUMBNAIL_WIDTH / width)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
