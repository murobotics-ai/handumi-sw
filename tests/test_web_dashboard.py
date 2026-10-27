import json
import threading
import time
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from handumi.cameras.base import CameraSample
from handumi.scripts.record import _RecordingDashboard, _wait_for_tracking
from handumi.web_dashboard import BrowserVoiceListener, WebDashboard


def _vosk_result(text: str, conf: float = 0.95) -> str:
    words = [{"word": word, "conf": conf} for word in text.split()]
    return json.dumps({"text": text, "result": words})


@pytest.mark.parametrize(
    ("text", "action"),
    [
        ("start", "start"),
        ("stop", "stop"),
        ("discard", "restart"),
        ("start recording", "start"),
        ("restart", "restart"),
        ("[unk] stop", "stop"),
        ("[unk] [unk] stop", "stop"),
    ],
)
def test_browser_voice_accepts_a_command_that_ends_the_utterance(text, action) -> None:
    listener = BrowserVoiceListener()

    listener._handle_result(_vosk_result(text))

    assert listener.poll() == action
    assert listener.last_heard["action"] == action


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("stop [unk]", "part of a longer sentence"),
        ("[unk] start [unk]", "part of a longer sentence"),
        ("[unk] [unk] [unk] stop", "part of a longer sentence"),
        ("[unk]", None),
    ],
)
def test_browser_voice_ignores_commands_inside_sentences(text, reason) -> None:
    listener = BrowserVoiceListener()

    listener._handle_result(_vosk_result(text))

    assert listener.poll() is None
    if reason is None:
        assert listener.last_heard is None
    else:
        assert listener.last_heard["reason"] == reason


def test_browser_voice_applies_confidence_and_debounce() -> None:
    listener = BrowserVoiceListener(confidence=0.7)

    listener._handle_result(_vosk_result("stop", conf=0.4))
    assert listener.poll() is None
    assert listener.last_heard["reason"] == "confidence 0.40"

    listener._handle_result(_vosk_result("stop"))
    listener._handle_result(_vosk_result("stop"))
    assert listener.poll() == "stop"
    assert listener.poll() is None
    assert listener.last_heard["reason"] == "repeat"


def test_dashboard_buttons_queue_known_commands_only() -> None:
    listener = BrowserVoiceListener()

    listener.push_command("restart")
    listener.push_command("launch")

    assert listener.poll() == "restart"
    assert listener.poll() is None


class _FakeGrippers:
    def read_normalized_widths(self):
        return SimpleNamespace(left_mm=42.0, right_mm=7.5, left_ticks=1800, right_ticks=1300)

    def latest(self):
        widths = self.read_normalized_widths()
        return SimpleNamespace(widths=widths, sample_time_ns=time.monotonic_ns())


class _FakeTracker:
    def __init__(self, left: bool = True, right: bool = False) -> None:
        self.left = left
        self.right = right

    def latest(self):
        return SimpleNamespace(
            connected=True,
            streaming=True,
            left_tracked=self.left,
            right_tracked=self.right,
            hmd_tracked=True,
            pc_monotonic_ns=time.monotonic_ns(),
        )


class _FakeCamera:
    def sample_at(self, target_time_ns=None):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        return CameraSample(image=image, capture_time_ns=time.monotonic_ns(), sequence=1)


class _BrokenCamera:
    def sample_at(self, target_time_ns=None):
        raise TimeoutError("Camera 'x' produced no timestamped frame.")


def _dashboard(**overrides) -> WebDashboard:
    status = SimpleNamespace(
        state="READY", detail="Waiting to start", task="t", dataset="d", episode=1,
        total="3", fps=30, current_frames=0, saved_episodes=0, saved_frames=0,
        program_runtime_s=1.0,
    )
    kwargs = dict(
        status=status,
        tracker=_FakeTracker(),
        grippers=_FakeGrippers(),
        voice=BrowserVoiceListener(),
        cameras=[_FakeCamera(), _BrokenCamera(), None],
        camera_names=["left_wrist", "right_wrist", "workspace"],
        port=0,
    )
    kwargs.update(overrides)
    return WebDashboard(**kwargs)


def test_snapshot_reports_samplers_and_camera_health() -> None:
    snapshot = _dashboard().snapshot()

    assert snapshot["state"] == "READY"
    assert snapshot["grippers"]["left_mm"] == pytest.approx(42.0)
    assert snapshot["tracking"]["left"] is True
    assert snapshot["tracking"]["right"] is False
    cameras = {camera["name"]: camera for camera in snapshot["cameras"]}
    assert cameras["left_wrist"]["ok"] is True
    assert cameras["left_wrist"]["jpeg"]
    assert cameras["right_wrist"]["ok"] is False
    assert cameras["workspace"]["enabled"] is False


def test_snapshot_without_grippers_or_voice() -> None:
    snapshot = _dashboard(grippers=None, voice=None).snapshot()

    assert snapshot["grippers"] == {"available": False}
    assert snapshot["voice"] == {"enabled": False}


def test_server_serves_page_and_forwards_button_commands() -> None:
    from websockets.sync.client import connect

    dashboard = _dashboard()
    dashboard.start()
    try:
        base = f"http://127.0.0.1:{dashboard.bound_port}"
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "HandUMI Recorder" in page

        with connect(f"ws://127.0.0.1:{dashboard.bound_port}/ws") as ws:
            first = json.loads(ws.recv(timeout=2.0))
            assert first["state"] == "READY"
            ws.send(json.dumps({"command": "start"}))
            deadline = time.monotonic() + 2.0
            command = None
            while command is None and time.monotonic() < deadline:
                command = dashboard.voice.poll()
                time.sleep(0.01)
        assert command == "start"
    finally:
        dashboard.stop()


def test_tracking_wait_shows_which_controller_is_missing() -> None:
    samples = iter([(True, False), (True, False), (True, True)])

    class Tracker:
        def latest(self):
            left, right = next(samples)
            return SimpleNamespace(left_tracked=left, right_tracked=right)

    status = _RecordingDashboard(
        task="t", dataset="d", fps=30, existing_episodes=0, existing_frames=0,
        controls=(),
    )
    details = []
    update_grippers = status.update_grippers

    def record_update(left_mm, right_mm):
        details.append(status.detail)
        update_grippers(left_mm, right_mm)

    status.update_grippers = record_update

    assert _wait_for_tracking(
        Tracker(),
        threading.Event(),
        poll_s=0.0,
        dashboard=status,
        grippers=_FakeGrippers(),
    )
    assert details[0] == "Waiting for Quest controller tracking (left=1 right=0)"
    assert status.left_gripper_mm == pytest.approx(42.0)
