#!/usr/bin/env python3
"""Unified HandUMI recorder for PICO and Meta Quest tracking backends.

Episode control: hands-free by voice unless disabled -- say "start recording",
"stop recording", or "restart". Feetech double-clap control is enabled by
default (double-squeeze the right gripper once to start the session; then right saves the
current episode and starts the next one immediately, left discards/restarts
the active attempt, and squeezing both grippers together discards the active
episode and ends the session), PICO buttons with --manual-control, and optional
--episode-time-s timing (unlimited unless set). Voice and clap control can
run together, so the squeeze still works when the room is too noisy to be
heard. --episodes defaults to recording until stopped; pass a positive count
to cap the session.

Spoken status announcements ("Recording episode 3", "Episode 3 saved, 812
frames", ...) are on by default — pass --no-sounds to disable them. The
required --output-dir selects the local output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import select
import signal
import sys
import tempfile
import termios
import threading
import time
import tty
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from handumi.audio import (
    AudioCaptureError,
    PicoAudioRecorder,
    audio_features,
    audio_metadata,
    validate_audio_files,
)
from handumi.calibration.control_tcp import (
    ControllerTcpCalibration,
    calibration_path_for_device,
    controller_tcp_calibration_metadata,
)
from handumi.calibration.spatial import (
    pose7_from_dict,
    session_calibration_metadata,
)
from handumi.cameras import (
    build_camera_specs,
    camera_output_size,
    connect_cameras,
    disconnect_cameras,
    read_camera_samples,
    resolve_camera_ids,
)
from handumi.config import (
    DEFAULT_RIG_CONFIG,
    camera_role,
    dataset_active_sides,
    load_optional_rig_section,
    resolve_active_sides,
)
from handumi.dataset.raw import (
    HANDUMI_CAPTURE_SCHEMA,
    HANDUMI_STATE_SEMANTICS,
    HANDUMI_TRACKING_SCHEMA,
    camera_health_features,
    capture_timing_features,
    feetech_features,
    pose_to_state_vector,
    raw_state_feature,
    raw_tracking_features,
)
from handumi.feetech import (
    FeetechGripperPair,
    FeetechGripperSampler,
    GripperWidths,
    assert_calibrated,
    load_config,
    user_calibration_path,
    zero_gripper_widths,
)
from handumi.feetech.bus import FeetechUnavailableError
from handumi.robots.utils import IDENTITY_POSE7
from handumi.synchronization import (
    SustainedHealthGate,
    capture_timing_frame,
    synchronized_gripper_frame,
    tracking_sample_at,
)
from handumi.tracking.base import ControllerPairSample, TrackingProvider
from handumi.tracking.gestures import BilateralClapArbiter, DoubleClapDetector
from handumi.tracking.meta_quest import MetaQuestConfig, MetaQuestTrackingProvider
from handumi.tracking.pico import (
    PICO_MODE_CHOICES,
    PicoTrackingFrozenError,
    START_BUTTON_CHOICES,
    PicoTrackingProvider,
    normalize_pico_mode,
    read_start_button_value,
    wait_for_manual_start,
    wait_for_start_button,
)
from handumi.tracking.transforms import Pose
from handumi.utils.speech import log_say
from handumi.utils.trajectory import TrajectoryTrail
from handumi.utils.voice import (
    VoiceCommandListener,
    VoiceUnavailableError,
    speech_duration_s,
)
from handumi.visualization import BACKGROUND_COLOR, LEFT_COLOR, RIGHT_COLOR
from handumi.web_dashboard import DEFAULT_PORT as DEFAULT_WEB_DASHBOARD_PORT
from handumi.web_dashboard import BrowserVoiceListener, WebDashboard

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("handumi.record")

try:
    from rich.align import Align
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:  # Keep recording usable in minimal source checkouts.
    Console = Group = Live = Panel = Table = Text = Align = None  # type: ignore[assignment,misc]

ROBOT_CONFIG_DIR = Path("configs/robots")
_RERUN_TRAIL_SECONDS = 10.0
_RERUN_CHART_WINDOW_S = 20.0
_SOFTWARE_VIDEO_CODEC = "h264"
_HARDWARE_VIDEO_CODECS = (
    "h264_videotoolbox",
    "hevc_videotoolbox",
    "h264_nvenc",
    "hevc_nvenc",
    "h264_vaapi",
    "h264_qsv",
)
_VALID_VIDEO_CODECS = {
    _SOFTWARE_VIDEO_CODEC,
    "hevc",
    "libsvtav1",
    *_HARDWARE_VIDEO_CODECS,
}
_RECORDING_DEFAULTS: dict[str, object] = {
    "device": "meta",
    "cameras": ["left_wrist", "right_wrist"],
    "cam_width": 640,
    "cam_height": 480,
    "cam_fps": 30,
    "fps": 30,
    "tracking_loss_timeout_s": 1.0,
    "sync_lag_s": 0.04,
    "max_sync_skew_s": 0.06,
    "camera_stale_timeout_s": 0.25,
    "gripper_stale_timeout_s": 0.10,
    "sensor_loss_timeout_s": 1.0,
    "feetech_sample_hz": 100.0,
    "skip_feetech": False,
    "no_video": False,
    "record_audio": False,
    "robot": "piper",
}


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class _RecordingDashboard:
    """Small live terminal UI shared by raw and teleop recording."""

    def __init__(
        self,
        *,
        task: str,
        dataset: str,
        fps: int,
        existing_episodes: int,
        existing_frames: int,
        controls: tuple[tuple[str, str], ...],
        started_at_s: float | None = None,
    ) -> None:
        self.task = task
        self.dataset = dataset
        self.fps = fps
        self.existing_episodes = existing_episodes
        self.existing_frames = existing_frames
        self.controls = controls
        self.saved_episodes = 0
        self.saved_frames = 0
        self.episode = existing_episodes + 1
        self.total = "∞"
        self.state = "STARTING"
        self.detail = "Preparing the recording session"
        self.current_frames = 0
        self.left_gripper_mm: float | None = None
        self.right_gripper_mm: float | None = None
        self._last_render_s = 0.0
        # ``started_at_s`` is captured as early as possible by the caller (at
        # process start, before hardware setup) so "program runtime" reflects
        # the whole run, not just the time spent inside the dashboard.
        self._started_at_s = (
            started_at_s if started_at_s is not None else time.perf_counter()
        )
        self._console = Console() if Console is not None else None
        self._live = None

    @property
    def program_runtime_s(self) -> float:
        """Wall-clock time elapsed since the program started running.

        This is distinct from the "effective data" durations below, which
        are derived from recorded frame counts and therefore only grow while
        data is actually being captured.
        """
        return max(0.0, time.perf_counter() - self._started_at_s)

    @property
    def enabled(self) -> bool:
        return self._console is not None and self._console.is_terminal

    def start(self) -> None:
        if not self.enabled or Live is None:
            log.info(
                "Recording UI: task=%s dataset=%s fps=%d existing_episodes=%d",
                self.task,
                self.dataset,
                self.fps,
                self.existing_episodes,
            )
            return
        self._live = Live(
            self,
            console=self._console,
            refresh_per_second=4,
            transient=False,
            screen=True,
        )
        self._live.start()

    def set_episode(
        self,
        episode: int,
        total: str,
        *,
        state: str = "READY",
        detail: str = "Waiting to start",
    ) -> None:
        self.episode = episode
        self.total = "∞" if total == "inf" else total
        self.state = state
        self.detail = detail
        self.current_frames = 0
        self._refresh(force=True)

    def recording(self, frames: int = 0) -> None:
        self.state = "RECORDING"
        self.detail = "Data capture in progress"
        self.current_frames = frames
        self._refresh()

    def update_frames(self, frames: int) -> None:
        self.current_frames = frames
        self._refresh()

    def update_grippers(self, left_mm: float, right_mm: float) -> None:
        self.left_gripper_mm = left_mm
        self.right_gripper_mm = right_mm
        self._refresh()

    def saved(self, frames: int) -> None:
        self.saved_episodes += 1
        self.saved_frames += frames
        self.current_frames = 0
        self.state = "SAVED"
        self.detail = f"Episode {self.episode} saved ({frames} frames)"
        self._refresh(force=True)

    def discarded(self, frames: int, reason: str) -> None:
        self.current_frames = 0
        self.state = "DISCARDED"
        self.detail = f"{frames} frames discarded: {reason}"
        self._refresh(force=True)

    def stop(self) -> None:
        self.current_frames = 0
        self.state = "COMPLETED"
        self.detail = f"{self.saved_episodes} episode(s) saved this session"
        self._refresh(force=True)
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _refresh(self, *, force: bool = False) -> None:
        if self._live is None:
            return
        now = time.perf_counter()
        if not force and now - self._last_render_s < 0.25:
            return
        self._last_render_s = now
        self._live.update(self, refresh=True)

    def __rich__(self):
        return self._render()

    def _render(self):
        assert (
            Table is not None
            and Panel is not None
            and Group is not None
            and Text is not None
            and Align is not None
        )
        active_frames = self.current_frames if self.state == "RECORDING" else 0
        session_frames = self.saved_frames + active_frames
        dataset_frames = self.existing_frames + session_frames

        stats = Table.grid(padding=(0, 2))
        stats.add_column(style="bold cyan", no_wrap=True)
        stats.add_column(style="white")
        stats.add_row("Status", f"[bold {'red' if self.state == 'DISCARDED' else 'green'}]{self.state}[/]")
        stats.add_row("Episode", f"[bold yellow]{self.episode} / {self.total}[/]")
        stats.add_row("Task", self.task)
        stats.add_row("Dataset", self.dataset)
        stats.add_row("Program runtime (wall clock)", _format_duration(self.program_runtime_s))
        stats.add_row("Effective data - current episode", _format_duration(active_frames / self.fps))
        stats.add_row("Effective data - this session", _format_duration(session_frames / self.fps))
        stats.add_row("Effective data - total dataset", _format_duration(dataset_frames / self.fps))
        stats.add_row("Frames (current / session)", f"{active_frames} / {session_frames}")
        if self.left_gripper_mm is not None and self.right_gripper_mm is not None:
            stats.add_row(
                "Grippers (left / right)",
                f"{self.left_gripper_mm:.1f} / {self.right_gripper_mm:.1f} mm "
                "[dim](close <12, reopen >20)[/]",
            )

        guide = Table.grid(padding=(0, 2))
        guide.add_column(style="bold magenta", no_wrap=True)
        guide.add_column(style="white")
        for control, action in self.controls:
            guide.add_row(control, action)

        title = Text("● HandUMI Data Recording", style="bold magenta")
        return Group(
            Panel(Align.center(title), border_style="magenta"),
            Panel(stats, title="Live status", border_style="cyan"),
            Panel(guide, title="Operator guide", border_style="green"),
            Panel(self.detail, border_style="yellow"),
        )


class StreamingEncodingError(RuntimeError):
    """The streaming encoder cannot produce a frame-aligned episode."""


class CameraCaptureError(RuntimeError):
    """A required recording camera stopped delivering usable frames."""


@dataclass(frozen=True)
class _VideoEncoderSelection:
    vcodec: str
    hardware: bool
    threads: int | None


class _StreamingEncoderProtocol(Protocol):
    def start_episode(self, video_keys: list[str], temp_dir: Path) -> None: ...

    def feed_frame(self, video_key: str, image: np.ndarray) -> None: ...

    def finish_episode(self) -> dict[str, tuple[Path, dict | None]]: ...

    def cancel_episode(self) -> None: ...

    def close(self) -> None: ...


class _StrictStreamingEncoder:
    """Make LeRobot streaming failures transactional at the episode boundary.

    LeRobot normally finishes streaming after it starts writing episode data.
    HandUMI prepares the videos first and stages the results so a failed encoder
    can discard the episode without appending partial rows to Parquet.
    """

    def __init__(self, encoder: _StreamingEncoderProtocol) -> None:
        self._encoder = encoder
        self._video_keys: tuple[str, ...] = ()
        self._frame_counts: dict[str, int] = {}
        self._prepared_results: dict[str, tuple[Path, dict | None]] | None = None

    def start_episode(self, video_keys: list[str], temp_dir: Path) -> None:
        self._prepared_results = None
        self._video_keys = tuple(video_keys)
        self._frame_counts = dict.fromkeys(video_keys, 0)
        self._encoder.start_episode(video_keys=video_keys, temp_dir=temp_dir)

    def feed_frame(self, video_key: str, image: np.ndarray) -> None:
        before = self._drop_count(video_key)
        try:
            self._encoder.feed_frame(video_key, image)
        except RuntimeError as exc:
            raise StreamingEncodingError(str(exc)) from exc
        after = self._drop_count(video_key)
        if after > before:
            raise StreamingEncodingError(
                f"encoder queue dropped a frame for {video_key}"
            )
        self._frame_counts[video_key] = self._frame_counts.get(video_key, 0) + 1

    def prepare_episode(self, expected_frames: int) -> None:
        if self._prepared_results is not None:
            return
        dropped = {
            key: self._drop_count(key)
            for key in self._video_keys
            if self._drop_count(key) > 0
        }
        if dropped:
            raise StreamingEncodingError(f"encoder dropped video frames: {dropped}")
        mismatched = {
            key: count
            for key, count in self._frame_counts.items()
            if count != expected_frames
        }
        if mismatched:
            raise StreamingEncodingError(
                f"streamed frame counts do not match {expected_frames}: {mismatched}"
            )

        try:
            results = self._encoder.finish_episode()
        except Exception as exc:
            raise StreamingEncodingError(f"encoder failed while finishing: {exc}") from exc

        try:
            for key in self._video_keys:
                if key not in results:
                    raise StreamingEncodingError(f"encoder returned no video for {key}")
                path, stats = results[key]
                path = Path(path)
                if not path.is_file() or path.stat().st_size <= 0:
                    raise StreamingEncodingError(f"encoder produced an empty video for {key}")
                if expected_frames >= 2 and stats is None:
                    raise StreamingEncodingError(f"encoder returned no statistics for {key}")
            self._prepared_results = results
        except Exception:
            self._remove_results(results)
            raise

    def finish_episode(self) -> dict[str, tuple[Path, dict | None]]:
        if self._prepared_results is None:
            raise StreamingEncodingError(
                "streaming video was not prepared before saving the episode"
            )
        results = self._prepared_results
        self._prepared_results = None
        return results

    def cancel_episode(self) -> None:
        if self._prepared_results is not None:
            self._remove_results(self._prepared_results)
            self._prepared_results = None
        self._encoder.cancel_episode()
        self._video_keys = ()
        self._frame_counts.clear()

    def close(self) -> None:
        if self._prepared_results is not None:
            self._remove_results(self._prepared_results)
            self._prepared_results = None
        self._encoder.close()

    def _drop_count(self, video_key: str) -> int:
        dropped = getattr(self._encoder, "_dropped_frames", {})
        return int(dropped.get(video_key, 0)) if isinstance(dropped, dict) else 0

    @staticmethod
    def _remove_results(results: dict[str, tuple[Path, dict | None]]) -> None:
        for path, _ in results.values():
            video_path = Path(path)
            try:
                video_path.unlink(missing_ok=True)
                video_path.parent.rmdir()
            except OSError:
                pass


def _recommended_encoder_threads(camera_count: int) -> int:
    """Reserve one logical CPU and avoid every camera claiming every core."""
    usable_cpus = max(1, (os.cpu_count() or 1) - 1)
    return max(1, min(4, usable_cpus // max(1, camera_count)))


def _available_hardware_vcodecs() -> list[str]:
    from av.codec import Codec

    available: list[str] = []
    for vcodec in _HARDWARE_VIDEO_CODECS:
        try:
            Codec(vcodec, "w")
        except Exception:
            continue
        available.append(vcodec)
    return available


def _probe_codec_options(vcodec: str, encoder_threads: int | None) -> dict[str, str]:
    options: dict[str, str] = {}
    if vcodec in ("h264", "hevc"):
        options.update({"g": "2", "crf": "30"})
    elif vcodec == "libsvtav1":
        options.update({"g": "2", "crf": "30", "preset": "12"})
    elif vcodec in ("h264_videotoolbox", "hevc_videotoolbox"):
        options.update({"g": "2", "q:v": "40"})
    elif vcodec in ("h264_nvenc", "hevc_nvenc"):
        options.update({"rc": "constqp", "qp": "30"})
    elif vcodec == "h264_vaapi":
        options["qp"] = "30"
    elif vcodec == "h264_qsv":
        options["global_quality"] = "30"

    if encoder_threads is not None:
        if vcodec == "libsvtav1":
            options["svtav1-params"] = f"lp={encoder_threads}"
        elif vcodec not in _HARDWARE_VIDEO_CODECS:
            options["threads"] = str(encoder_threads)
    return options


def _probe_video_encoder(
    vcodec: str,
    *,
    width: int,
    height: int,
    fps: int,
    encoder_threads: int | None,
) -> tuple[bool, str | None]:
    """Encode real frames through the same PyAV/FFmpeg backend as LeRobot."""
    import av
    from av.video.stream import VideoStream

    container = None
    try:
        with tempfile.TemporaryDirectory(prefix="handumi-encoder-probe-") as tmp:
            path = Path(tmp) / "probe.mp4"
            container = av.open(str(path), "w")
            stream = cast(
                VideoStream,
                container.add_stream(
                    vcodec,
                    fps,
                    options=_probe_codec_options(vcodec, encoder_threads),
                ),
            )
            stream.pix_fmt = "yuv420p"
            stream.width = width
            stream.height = height
            stream.time_base = Fraction(1, fps)
            for value in (0, 64, 128):
                frame = np.full((height, width, 3), value, dtype=np.uint8)
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                video_frame.pts = value // 64
                video_frame.time_base = Fraction(1, fps)
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            container = None
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError("probe produced an empty MP4")
        return True, None
    except Exception as exc:
        return False, str(exc)
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass


def _select_video_encoder(
    *,
    policy: str,
    requested_vcodec: str | None,
    width: int,
    height: int,
    fps: int,
    camera_count: int,
    requested_threads: int | None,
) -> _VideoEncoderSelection:
    """Resolve auto/cpu/gpu to a codec that passes a real encode probe."""
    hardware_codecs = _available_hardware_vcodecs()
    explicit_codec = requested_vcodec not in (None, "auto")
    if explicit_codec:
        if requested_vcodec not in _VALID_VIDEO_CODECS:
            choices = ", ".join(sorted(_VALID_VIDEO_CODECS))
            raise SystemExit(
                f"Unsupported --vcodec {requested_vcodec!r}; choose one of: {choices}"
            )
        candidates = [str(requested_vcodec)]
    elif policy == "cpu":
        candidates = [_SOFTWARE_VIDEO_CODEC]
    elif policy == "gpu":
        candidates = hardware_codecs
    else:
        candidates = [*hardware_codecs, _SOFTWARE_VIDEO_CODEC]

    if not candidates:
        raise SystemExit(
            "--encoder gpu requested, but PyAV/FFmpeg exposes no supported hardware encoder."
        )

    failures: list[str] = []
    for candidate in candidates:
        hardware = candidate in _HARDWARE_VIDEO_CODECS
        threads = (
            requested_threads
            if requested_threads is not None
            else None if hardware else _recommended_encoder_threads(camera_count)
        )
        ok, error = _probe_video_encoder(
            candidate,
            width=width,
            height=height,
            fps=fps,
            encoder_threads=threads,
        )
        if ok:
            return _VideoEncoderSelection(
                vcodec=candidate,
                hardware=hardware,
                threads=threads,
            )
        failures.append(f"{candidate}: {error or 'unknown error'}")
        if hardware:
            log.warning("Hardware encoder %s failed its startup probe: %s", candidate, error)

    requested = requested_vcodec if explicit_codec else policy
    details = "; ".join(failures)
    raise SystemExit(f"No usable video encoder for {requested!r}: {details}")


def _install_strict_streaming_encoder(dataset: object) -> None:
    writer = getattr(dataset, "writer", None)
    encoder = getattr(writer, "_streaming_encoder", None)
    if writer is None or encoder is None:
        raise RuntimeError("LeRobot did not create the requested streaming encoder.")
    writer._streaming_encoder = _StrictStreamingEncoder(encoder)


def _prepare_streaming_episode(dataset: object, expected_frames: int) -> None:
    writer = getattr(dataset, "writer", None)
    encoder = getattr(writer, "_streaming_encoder", None)
    if not isinstance(encoder, _StrictStreamingEncoder):
        # RuntimeError on purpose: this guards an internal setup invariant
        # (install_strict_streaming_encoder ran), not a caller-supplied type.
        raise RuntimeError(  # noqa: TRY004
            "HandUMI strict streaming encoder is not installed."
        )
    encoder.prepare_episode(expected_frames)


class _EscapeStopListener:
    """Turn terminal Escape into the recorder's graceful stop event."""

    def __init__(self, stop_event: threading.Event, fd: int | None = None) -> None:
        self.stop_event = stop_event
        self.fd = fd
        self._original: list | None = None
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        fd = self.fd
        if fd is None:
            if not sys.stdin.isatty():
                log.warning("Escape stop disabled: stdin is not a terminal.")
                return False
            fd = sys.stdin.fileno()
            self.fd = fd
        if not os.isatty(fd):
            return False
        self._original = termios.tcgetattr(fd)
        tty.setcbreak(fd, termios.TCSANOW)
        self._thread = threading.Thread(
            target=self._run,
            name="handumi_escape_stop",
            daemon=True,
        )
        self._thread.start()
        return True

    def _run(self) -> None:
        assert self.fd is not None
        while not self._closed.is_set() and not self.stop_event.is_set():
            readable, _, _ = select.select([self.fd], [], [], 0.1)
            if readable and os.read(self.fd, 1) == b"\x1b":
                log.info("Escape pressed - discarding active episode and stopping ...")
                self.stop_event.set()
                return

    def stop(self) -> None:
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.fd is not None and self._original is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self._original)
            self._original = None


def build_features(
    cam_names: list[str],
    cam_width: int,
    cam_height: int,
    use_videos: bool,
    camera_specs: list[dict[str, object]] | None = None,
    record_audio: bool = False,
) -> dict:
    img_dtype = "video" if use_videos else "image"
    features: dict = {}
    specs_by_name = {
        str(spec["name"]): spec for spec in (camera_specs or [])
    }
    for cam in cam_names:
        width, height = camera_output_size(
            specs_by_name.get(cam, {}),
            default_width=cam_width,
            default_height=cam_height,
        )
        features[f"observation.images.{cam}"] = {
            "dtype": img_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
    features["observation.state"] = _tuple_shape(raw_state_feature())
    features["action"] = _tuple_shape(raw_state_feature())
    features.update(feetech_features())
    features.update(raw_tracking_features())
    features.update(capture_timing_features())
    features.update(camera_health_features(cam_names))
    if record_audio:
        features.update(audio_features())
    return features


def _tuple_shape(feature: dict) -> dict:
    feature = dict(feature)
    feature["shape"] = tuple(feature["shape"])
    return feature


def build_observation(
    sample: ControllerPairSample, widths: GripperWidths,
    active_sides: tuple[str, ...] = ("left", "right"),
) -> dict:
    # Keep the raw 16D schema portable. Absent sensors have neutral poses and
    # zero widths, with false tracking flags; metadata declares their absence.
    for side in ("left", "right"):
        if side in active_sides:
            continue
        sample = replace(sample, **{
            f"{side}_controller_pose": IDENTITY_POSE7.astype(np.float32).copy(),
            f"{side}_tcp_pose": IDENTITY_POSE7.astype(np.float32).copy(),
            f"{side}_device_controller_pose": IDENTITY_POSE7.astype(np.float32).copy(),
            f"{side}_tracked": False,
            f"{side}_device_tracked": False,
            f"{side}_pose_valid": False,
        })
        widths = replace(widths, **{
            side: 0.0, f"{side}_mm": 0.0, f"{side}_normalized": 0.0, f"{side}_ticks": 0,
        })
    left_controller = _pose_from_pose7(sample.left_controller_pose)
    right_controller = _pose_from_pose7(sample.right_controller_pose)
    state = pose_to_state_vector(
        left_controller,
        right_controller,
        widths.left,
        widths.right,
    )
    return {
        "observation.state": state,
        "action": state.copy(),
        "observation.feetech.left_ticks": np.array([widths.left_ticks], dtype=np.int64),
        "observation.feetech.right_ticks": np.array([widths.right_ticks], dtype=np.int64),
        "observation.feetech.left_width_mm": np.array([widths.left_mm], dtype=np.float32),
        "observation.feetech.right_width_mm": np.array([widths.right_mm], dtype=np.float32),
        "observation.feetech.left_normalized": np.array([widths.left_normalized], dtype=np.float32),
        "observation.feetech.right_normalized": np.array([widths.right_normalized], dtype=np.float32),
        **sample.tracking_frame(),
    }


def _pose_from_pose7(pose7: np.ndarray) -> Pose:
    pose = np.asarray(pose7, dtype=np.float32).reshape(7)
    return Pose(pose[:3], pose[3:7])


def _tracking_healthy(
    sample: ControllerPairSample, active_sides: tuple[str, ...] = ("left", "right")
) -> bool:
    return all(getattr(sample, f"{side}_tracked") for side in active_sides)


def _wait_for_tracking(
    tracker: TrackingProvider,
    stop_event: threading.Event,
    *,
    poll_s: float = 0.05,
    active_sides: tuple[str, ...] = ("left", "right"),
    dashboard: _RecordingDashboard | None = None,
    grippers: FeetechGripperSampler | FeetechGripperPair | None = None,
) -> bool:
    """Wait until both controller poses are fresh and valid."""
    last_report = float("-inf")
    while not stop_event.is_set():
        sample = tracker.latest()
        if dashboard is not None:
            widths = _latest_gripper_widths(grippers)
            dashboard.state = "WAITING"
            dashboard.detail = (
                "Waiting for Quest controller tracking "
                f"(left={int(sample.left_tracked)} right={int(sample.right_tracked)})"
            )
            dashboard.update_grippers(widths.left_mm, widths.right_mm)
        if _tracking_healthy(sample, active_sides):
            log.info("Required controllers tracked (%s); recording gate open.", "/".join(active_sides))
            return True

        now = time.monotonic()
        if now - last_report >= 2.0:
            log.warning(
                "Waiting for controller tracking (left=%d right=%d) ...",
                int(sample.left_tracked),
                int(sample.right_tracked),
            )
            last_report = now
        time.sleep(poll_s)
    return False


class _RecordingRerun:
    """Live Rerun stream owned by the recorder (never opens another device)."""

    def __init__(self, cam_names: list[str], fps: int) -> None:
        import rerun as rr
        import rerun.blueprint as rrb
        import rerun.datatypes as rdt

        self.rr = rr
        self.trails = {
            "left": TrajectoryTrail(max(2, int(_RERUN_TRAIL_SECONDS * fps))),
            "right": TrajectoryTrail(max(2, int(_RERUN_TRAIL_SECONDS * fps))),
        }
        self.raw_trails = {
            "left": TrajectoryTrail(max(2, int(_RERUN_TRAIL_SECONDS * fps))),
            "right": TrajectoryTrail(max(2, int(_RERUN_TRAIL_SECONDS * fps))),
        }
        self._healthy = True
        rr.init("handumi_record", spawn=True)
        rr.log("tracking", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        for path, name, color in (
            ("observation.feetech.left_width_mm", "left_width_mm", LEFT_COLOR),
            ("observation.feetech.right_width_mm", "right_width_mm", RIGHT_COLOR),
        ):
            rr.log(path, rr.SeriesLines(colors=[[*color, 255]], widths=[2.5], names=[name]), static=True)

        corners = [[sx * 0.75, sy * 0.75, sz * 0.4]
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        rr.log("tracking/bounds", rr.Points3D(corners, colors=[[128, 100, 100, 90]] * 8, radii=0.004), static=True)
        recent = rrb.VisibleTimeRanges(rrb.VisibleTimeRange(
            timeline="log_time",
            range=rdt.TimeRange(
                start=rdt.TimeRangeBoundary.cursor_relative(seconds=-_RERUN_CHART_WINDOW_S),
                end=rdt.TimeRangeBoundary.cursor_relative(seconds=0.0),
            ),
        ))
        chart = rrb.TimeSeriesView(
            origin="/",
            contents=["/observation.feetech.left_width_mm", "/observation.feetech.right_width_mm"],
            name="gripper_width_mm",
            axis_y=rrb.ScalarAxis(range=(0.0, 90.0)),
            time_ranges=recent,
            plot_legend=rrb.Corner2D.LeftTop,
        )
        right = chart if not cam_names else rrb.Vertical(
            rrb.Horizontal(*[
                rrb.Spatial2DView(origin=f"/observation.images.{name}", name=name)
                for name in cam_names
            ]),
            chart,
            row_shares=[3, 2],
        )
        status = rrb.TextDocumentView(
            origin="/recording",
            contents=["/recording/status"],
            name="recording_status",
        )
        rr.send_blueprint(rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="/tracking", name="controller_trajectory",
                                       background=rrb.Background(color=[*BACKGROUND_COLOR, 255])),
                    right,
                    column_shares=[2, 3],
                ),
                status,
                row_shares=[10, 1],
            ),
            rrb.BlueprintPanel(state="collapsed"),
            rrb.SelectionPanel(state="collapsed"),
            rrb.TimePanel(state="collapsed"),
        ), make_active=True, make_default=True)
        self.set_status("READY", "Waiting to start the first episode")

    def set_status(self, state: str, detail: str) -> None:
        """Show the current recorder state as a persistent operator flag."""
        self.rr.log(
            "recording/status",
            self.rr.TextDocument(
                f"# {state}\n\n{detail}", media_type="text/markdown"
            ),
        )

    def log(self, cam_frames: dict, sample: ControllerPairSample, widths: GripperWidths) -> None:
        """Log a frame without allowing a viewer problem to stop recording."""
        if not self._healthy:
            return
        rr = self.rr
        try:
            # ``cam_frames`` also contains camera-health scalar fields. Only
            # image entries belong in Rerun's image archetype.
            for key, frame in cam_frames.items():
                if key.startswith("observation.images."):
                    rr.log(key, rr.Image(frame).compress(jpeg_quality=75))
            rr.log("observation.feetech.left_width_mm", rr.Scalars(float(widths.left_mm)))
            rr.log("observation.feetech.right_width_mm", rr.Scalars(float(widths.right_mm)))
            for side, tcp, raw, color, tracked in (
                ("left", sample.left_tcp_pose, sample.left_controller_pose, LEFT_COLOR, sample.left_tracked),
                ("right", sample.right_tcp_pose, sample.right_controller_pose, RIGHT_COLOR, sample.right_tracked),
            ):
                if not tracked:
                    continue
                trail = self.trails[side]
                trail.append(tcp[:3])
                rr.log(f"tracking/{side}/tcp", rr.Points3D([tcp[:3]], colors=[color], radii=0.012))
                if len(points := trail.points()) >= 2:
                    rr.log(f"tracking/{side}/trail", rr.LineStrips3D([points], colors=[color], radii=0.003))
                raw_trail = self.raw_trails[side]
                raw_trail.append(raw[:3])
                rr.log(f"tracking/{side}/raw", rr.Points3D([raw[:3]], colors=[[*color, 90]], radii=0.007))
                if len(raw_points := raw_trail.points()) >= 2:
                    rr.log(f"tracking/{side}/raw_trail", rr.LineStrips3D(
                        [raw_points], colors=[[*color, 90]], radii=0.0015
                    ))
        except Exception:
            self._healthy = False
            log.exception("Rerun failed; disabling live view while recording continues.")


def record_episode(
    *,
    dataset,
    cameras: list,
    cam_names: list[str],
    tracker: TrackingProvider,
    grippers: FeetechGripperSampler | FeetechGripperPair | None,
    episode_time_s: float | None,
    fps: int,
    task: str,
    cam_width: int,
    cam_height: int,
    stop_event: threading.Event,
    manual_control: bool,
    start_button: str,
    repeat_button: str,
    finish_button: str,
    start_threshold: float,
    clap_detector: DoubleClapDetector | None = None,
    clap_arbiter: BilateralClapArbiter | None = None,
    voice: VoiceCommandListener | None = None,
    tracking_loss_timeout_s: float = 1.0,
    sync_lag_s: float = 0.04,
    max_sync_skew_s: float = 0.06,
    camera_stale_timeout_s: float = 0.25,
    gripper_stale_timeout_s: float = 0.10,
    sensor_loss_timeout_s: float = 1.0,
    rerun: _RecordingRerun | None = None,
    dashboard: _RecordingDashboard | None = None,
    audio_recorder: PicoAudioRecorder | None = None,
    active_sides: tuple[str, ...] = ("left", "right"),
) -> tuple[int, str, bool]:
    control_interval = 1.0 / fps
    n_frames = 0
    start_t = time.perf_counter()
    status = "recorded"
    advance_after_save = False
    clap_control = clap_detector is not None
    xrt = getattr(tracker, "xrt", None)
    prev_start = (
        read_start_button_value(xrt, start_button) >= start_threshold
        if manual_control and xrt is not None
        else False
    )
    prev_repeat = (
        read_start_button_value(xrt, repeat_button) >= start_threshold
        if manual_control and xrt is not None
        else False
    )
    prev_finish = (
        read_start_button_value(xrt, finish_button) >= start_threshold
        if manual_control and xrt is not None
        else False
    )

    # Clap/voice start episodes hands-free. Once recording, another stop
    # command saves the episode. --episode-time-s is an optional max duration.
    episode_deadline_s = (
        episode_time_s
        if not manual_control and episode_time_s is not None and episode_time_s > 0
        else None
    )
    tracking_loss_timeout_ns = int(tracking_loss_timeout_s * 1e9)
    tracking_lost_since_ns: int | None = None
    episode_start_ns: int | None = None
    sync_lag_ns = int(sync_lag_s * 1e9)
    max_sync_skew_ns = int(max_sync_skew_s * 1e9)
    health_gate = SustainedHealthGate(sensor_loss_timeout_s)

    while True:
        loop_start = time.perf_counter()
        tracking_now_ns = time.monotonic_ns()
        if episode_start_ns is None:
            episode_start_ns = tracking_now_ns
        elapsed = loop_start - start_t
        if stop_event.is_set():
            status = "interrupted"
            dataset.clear_episode_buffer()
            break
        target_time_ns = max(episode_start_ns, tracking_now_ns - sync_lag_ns)
        # Check freshness before accepting a timer/button save as well as rows.
        # Fatal provider errors use main's discard-and-finalize path.
        try:
            sample = tracking_sample_at(tracker, target_time_ns)
        except PicoTrackingFrozenError:
            dataset.clear_episode_buffer()
            raise
        if episode_deadline_s is not None and elapsed >= episode_deadline_s:
            if (
                tracking_lost_since_ns is not None
                and tracking_now_ns - tracking_lost_since_ns >= tracking_loss_timeout_ns
            ):
                status = "tracking_lost"
                log.error(
                    "Controller tracking unavailable for %.2fs; discarding episode.",
                    (tracking_now_ns - tracking_lost_since_ns) / 1e9,
                )
            break

        if manual_control and xrt is not None:
            start_pressed = read_start_button_value(xrt, start_button) >= start_threshold
            repeat_pressed = read_start_button_value(xrt, repeat_button) >= start_threshold
            finish_pressed = read_start_button_value(xrt, finish_button) >= start_threshold
            start_rise = start_pressed and not prev_start
            repeat_rise = repeat_pressed and not prev_repeat
            finish_rise = finish_pressed and not prev_finish
            prev_start, prev_repeat, prev_finish = start_pressed, repeat_pressed, finish_pressed
            if repeat_rise:
                status = "repeat"
                dataset.clear_episode_buffer()
                break
            if finish_rise:
                status = "finish"
                break
            if start_rise:
                status = "recorded"
                break

        target_time_ns = max(episode_start_ns, tracking_now_ns - sync_lag_ns)
        cam_frames, camera_health = read_camera_samples(
            cameras,
            cam_names,
            target_time_ns=target_time_ns,
            record_time_ns=tracking_now_ns,
            width=cam_width,
            height=cam_height,
            stale_timeout_s=camera_stale_timeout_s,
            max_sync_skew_s=max_sync_skew_s,
        )
        gripper_frame = synchronized_gripper_frame(
            grippers,
            target_time_ns=target_time_ns,
            record_time_ns=tracking_now_ns,
            stale_timeout_s=gripper_stale_timeout_s,
            max_sync_skew_s=max_sync_skew_s,
        )
        audio_frame: dict[str, Any] = {}
        audio_healthy = True
        if audio_recorder is not None:
            audio_frame, audio_healthy = audio_recorder.synchronized_frame(
                target_time_ns,
                tracking_now_ns,
                stale_timeout_s=camera_stale_timeout_s,
                max_sync_skew_s=max_sync_skew_s,
            )
        widths = gripper_frame.widths
        # Gesture control must use the freshest physical reading, just like
        # teleop_sim.  The synchronized, deliberately delayed sample remains
        # the correct one to write into this dataset row.
        control_widths = _latest_gripper_widths(grippers)
        if dashboard is not None:
            dashboard.update_grippers(
                control_widths.left_mm, control_widths.right_mm
            )
        sample_time_ns = int(sample.aligned_time_ns or sample.pc_monotonic_ns)
        tracking_sync_ok = bool(
            sample_time_ns > 0
            and abs(sample_time_ns - target_time_ns) <= max_sync_skew_ns
        )
        if _tracking_healthy(sample, active_sides) and tracking_sync_ok:
            if tracking_lost_since_ns is not None:
                log.info("Controller tracking recovered before the episode timeout.")
            tracking_lost_since_ns = None
        elif tracking_lost_since_ns is None:
            # For a stale cached frame, count loss from its receive timestamp
            # instead of adding the freshness timeout to the one-second grace.
            sample_time_ns = int(sample.pc_monotonic_ns)
            tracking_lost_since_ns = (
                min(sample_time_ns, tracking_now_ns)
                if sample_time_ns > 0
                else tracking_now_ns
            )
            log.warning(
                "Controller tracking lost (left=%d right=%d); "
                "discarding if it lasts %.2fs.",
                int(sample.left_tracked),
                int(sample.right_tracked),
                tracking_loss_timeout_s,
            )
        elif tracking_now_ns - tracking_lost_since_ns >= tracking_loss_timeout_ns:
            status = "tracking_lost"
            log.error(
                "Controller tracking unavailable for %.2fs; discarding episode.",
                (tracking_now_ns - tracking_lost_since_ns) / 1e9,
            )
            break

        sensor_health = {
            **camera_health,
            "feetech": gripper_frame.healthy_for_gate,
            "audio": audio_healthy,
        }
        recovered, timed_out_sensors = health_gate.update(
            sensor_health, tracking_now_ns
        )
        for sensor in recovered:
            log.info("Sensor health recovered before timeout: %s.", sensor)
        if timed_out_sensors:
            failed_cameras = sorted(
                sensor.removeprefix("camera.")
                for sensor in timed_out_sensors
                if sensor.startswith("camera.")
            )
            if failed_cameras:
                names = ", ".join(failed_cameras)
                raise CameraCaptureError(
                    f"camera capture failed for {names}: no fresh synchronized "
                    f"frames for {sensor_loss_timeout_s:.2f}s; the camera may "
                    "have disconnected or changed device ID"
                )
            status = "sensor_unhealthy"
            log.error(
                "Sensor health unavailable for %.2fs (%s); discarding episode.",
                sensor_loss_timeout_s,
                ", ".join(sorted(timed_out_sensors)),
            )
            break

        if voice is not None:
            command = voice.poll()
            if command == "stop":
                voice.drain()
                status = "recorded"
                break
            if command == "restart":
                voice.drain()
                status = "repeat"
                dataset.clear_episode_buffer()
                break
        if clap_control:
            assert clap_arbiter is not None
            clap_gesture = clap_arbiter.update(
                clap_detector,
                control_widths.left_mm,
                control_widths.right_mm,
                loop_start,
            )
            if clap_gesture is not None:
                log.info(
                    "Double-squeeze detected: %s (left=%.1f mm, right=%.1f mm).",
                    clap_gesture.upper(),
                    control_widths.left_mm,
                    control_widths.right_mm,
                )
            if clap_gesture == "both":
                status = "session_finished"
                dataset.clear_episode_buffer()
                break
            if clap_gesture == "right":
                status = "recorded"
                advance_after_save = True
                break
            if clap_gesture == "left":
                status = "repeat"
                dataset.clear_episode_buffer()
                break
        if rerun is not None:
            rerun.log(cam_frames, sample, widths)
        try:
            dataset.add_frame(
                {
                    **cam_frames,
                    **build_observation(sample, widths, active_sides),
                    **gripper_frame.frame,
                    **audio_frame,
                    **capture_timing_frame(target_time_ns, tracking_now_ns),
                    "task": task,
                }
            )
        except StreamingEncodingError as exc:
            status = "encoder_unhealthy"
            log.error("Streaming encoder failed; discarding episode: %s", exc)
            break
        n_frames += 1
        if dashboard is not None:
            dashboard.update_frames(n_frames)

        dt = time.perf_counter() - loop_start
        sleep = control_interval - dt
        if sleep > 0:
            time.sleep(sleep)
        else:
            log.warning("Loop slower than %d Hz (%.1f Hz actual).", fps, 1.0 / max(dt, 1e-6))

    return n_frames, status, advance_after_save


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    show_advanced = "--help-advanced" in raw_argv
    raw_argv = [value for value in raw_argv if value != "--help-advanced"]

    def advanced(text: str) -> str:
        return text if show_advanced else argparse.SUPPRESS

    p = argparse.ArgumentParser(
        description="Record HandUMI data with PICO or Meta Quest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Device, cameras, FPS, and robot resolve from CLI, a resumed dataset, "
            "configs/rig.yaml, then safe defaults."
        ),
    )
    p.add_argument("--help-advanced", action="store_true", help="Show every expert option.")
    p.add_argument("--device", choices=("pico", "meta"), default=None)
    p.add_argument("--side", choices=("left", "right", "both"), default=None,
                   help="Sensors to record; defaults to the robot's declared arms.")
    p.add_argument(
        "--cameras",
        type=_camera_list_arg,
        default=None,
        help="Comma-separated camera names declared under cameras in the rig config.",
    )
    p.add_argument(
        "--rig-config",
        type=Path,
        default=DEFAULT_RIG_CONFIG,
        help=advanced("Machine-local cameras, Feetech, and Meta Quest configuration."),
    )
    p.add_argument("--cam-width", type=int, default=None, help=advanced("Camera width."))
    p.add_argument("--cam-height", type=int, default=None, help=advanced("Camera height."))
    p.add_argument("--cam-fps", type=int, default=None, help=advanced("Camera capture FPS."))
    p.add_argument("--feetech-port", type=str, default=None, help=advanced("Legacy shared Feetech port."))
    p.add_argument(
        "--skip-feetech",
        action="store_true",
        default=None,
        help=advanced("Record without gripper sensors and zero-fill their values."),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Local dataset directory, for example outputs/my-dataset.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Append episodes to the finalized dataset in --output-dir. "
            "--episodes is the number of additional episodes to record."
        ),
    )
    p.add_argument("--task", type=str, default="HandUMI recording")
    p.add_argument(
        "--episodes",
        dest="num_episodes",
        type=int,
        default=0,
        help="Number of episodes to record; omit or pass 0 to record until stopped.",
    )
    p.add_argument(
        "--episode-time-s",
        type=float,
        default=None,
        help=advanced("Maximum episode duration in seconds; omit for unlimited."),
    )
    p.add_argument("--fps", type=int, default=None, help=advanced("Dataset row rate."))
    p.add_argument(
        "--tracking-loss-timeout-s",
        type=float,
        default=None,
        help=advanced("Discard after sustained controller tracking loss."),
    )
    p.add_argument(
        "--sync-lag-s",
        type=float,
        default=None,
        help=advanced("Capture rows behind real time for sensor alignment."),
    )
    p.add_argument(
        "--max-sync-skew-s",
        type=float,
        default=None,
        help=advanced("Maximum source-to-row synchronization skew."),
    )
    p.add_argument("--camera-stale-timeout-s", type=float, default=None, help=advanced("Camera freshness timeout."))
    p.add_argument("--gripper-stale-timeout-s", type=float, default=None, help=advanced("Gripper freshness timeout."))
    p.add_argument(
        "--sensor-loss-timeout-s",
        type=float,
        default=None,
        help=advanced("Discard after a sensor remains unhealthy."),
    )
    p.add_argument("--feetech-sample-hz", type=float, default=None, help=advanced("Feetech sampler frequency."))
    p.add_argument("--no-video", action="store_true", default=None, help=advanced("Store individual images instead of MP4."))
    p.add_argument(
        "--record-audio",
        action="store_true",
        default=None,
        help="Record the PICO app microphone stream as one WAV per episode.",
    )
    p.add_argument(
        "--rerun",
        action="store_true",
        help="Open a live Rerun view with recorded cameras, controller/TCP trails, and gripper widths.",
    )
    p.add_argument(
        "--encoder",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help=advanced(
            "Video encoder policy. 'auto' probes hardware first and falls back "
            "to H.264 on CPU; 'gpu' requires a working hardware encoder."
        ),
    )
    p.add_argument(
        "--vcodec",
        type=str,
        default=None,
        help=advanced(
            "Advanced codec override (for example h264, h264_nvenc, or "
            "libsvtav1). Overrides automatic codec selection."
        ),
    )
    p.add_argument(
        "--encoder-threads",
        type=int,
        default=None,
        help=advanced("Per-camera CPU encoder thread override."),
    )
    p.add_argument(
        "--encoder-queue-size",
        type=int,
        default=None,
        help=advanced("Streaming queue capacity per camera."),
    )
    p.add_argument("--push-to-hub", action="store_true", help=advanced("Upload after finalization."))
    p.add_argument(
        "--dataset-license",
        default="other",
        help=advanced("Hugging Face dataset-card license identifier."),
    )
    p.add_argument(
        "--controller-tcp-calibration",
        type=Path,
        default=None,
        help=advanced(
            "Explicit Controller-to-HandUMI-TCP override. Defaults to the "
            "robot/device setup and snapshots its tool identity in dataset metadata. "
            "Raw controller poses remain unchanged."
        ),
    )
    p.add_argument(
        "--session-calibration",
        type=Path,
        default=None,
        help=advanced(
            "Tracking-device-to-table session calibration from "
            "`handumi calibrate spatial`. "
            "Locks all episodes to the same table frame."
        ),
    )
    p.add_argument(
        "--robot",
        default=None,
        help=advanced(
            "Intended robot embodiment. Snapshots configs/robots/<robot>.yaml in "
            "metadata; raw recordings remain robot-agnostic."
        ),
    )

    p.add_argument("--quest-ip", type=str, default=None, help=advanced("Quest IP override."))
    p.add_argument("--tcp-port", type=int, default=None, help=advanced("Quest TCP port override."))
    p.add_argument("--sync-port", type=int, default=None, help=advanced("Quest sync port override."))

    p.add_argument(
        "--pico-mode",
        type=normalize_pico_mode,
        choices=PICO_MODE_CHOICES,
        default="controllers",
        help=advanced("PICO tracking mode."),
    )
    pico_transport = p.add_mutually_exclusive_group()
    pico_transport.add_argument("--pico-adb", action="store_true", help=advanced("Use PICO over ADB."))
    pico_transport.add_argument("--pico-wifi", action="store_true", help=advanced("Use PICO over Wi-Fi."))
    p.add_argument("--skip-adb-check", action="store_true", help=advanced("Skip PICO ADB validation."))
    p.add_argument("--start-button", choices=START_BUTTON_CHOICES, default="enter", help=advanced("Episode start button."))
    p.add_argument("--start-threshold", type=float, default=0.75, help=advanced("Controller button threshold."))
    p.add_argument("--manual-control", action="store_true", help=advanced("Use PICO controller buttons."))
    p.add_argument("--repeat-button", choices=START_BUTTON_CHOICES, default="B", help=advanced("Repeat button."))
    p.add_argument("--finish-button", choices=START_BUTTON_CHOICES, default="Y", help=advanced("Finish button."))
    p.add_argument(
        "--voice-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Hands-free by voice: say "start recording", "stop recording", '
        'or "restart". On by default; --no-voice-control falls back to clap, '
        "manual buttons, or --episode-time-s if set.",
    )
    p.add_argument("--voice-device", default=None, help=advanced("Microphone name or index (default: system default)."))
    p.add_argument("--voice-confidence", type=float, default=0.7, help=advanced("Minimum recognition confidence (0-1)."))
    p.add_argument(
        "--web-dashboard",
        type=int,
        nargs="?",
        const=DEFAULT_WEB_DASHBOARD_PORT,
        default=None,
        metavar="PORT",
        help="Serve a browser dashboard on localhost:PORT (default 8890) with live "
        "widths, tracking and camera previews. Voice commands come from the "
        "browser microphone instead of a local one.",
    )
    p.add_argument(
        "--clap-control",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow hands-free gripper squeezes (on by default with Feetech): "
        "double-squeeze right once "
        "to start the session, then right saves the episode and starts the "
        "next one immediately; double-squeeze left discards/restarts the "
        "active attempt; double-squeeze both grippers discards the active "
        "episode and ends the session. Use --no-clap-control to disable it.",
    )
    p.add_argument(
        "--no-sounds",
        action="store_true",
        help="Disable spoken episode-status announcements (start/save/discard/stop).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the recording plan without opening hardware.",
    )
    if show_advanced:
        p.print_help()
        raise SystemExit(0)
    return p.parse_args(raw_argv)


def _resolve_recording_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply CLI > resumed dataset > rig.yaml > built-in defaults."""
    args.output_dir = Path(args.output_dir)
    args.repo_id = f"local/{args.output_dir.name}"
    args.cam_ids = None

    explicit = {
        name: getattr(args, name, None) is not None
        for name in (
            "device",
            "cameras",
            "cam_width",
            "cam_height",
            "cam_fps",
            "fps",
            "skip_feetech",
            "no_video",
            "record_audio",
            "robot",
            "side",
            "session_calibration",
            "controller_tcp_calibration",
        )
    }
    args._explicit_recording = explicit
    args._resume_info = None
    args._resume_handumi = None

    resume_values: dict[str, object] = {}
    if bool(getattr(args, "resume", False)):
        info_path = Path(args.output_dir) / "meta" / "info.json"
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Cannot resume: invalid {info_path}: {exc}") from exc
        handumi = info.get("handumi")
        if not isinstance(handumi, dict):
            raise SystemExit(f"Cannot resume: {info_path} has no HandUMI metadata.")
        args._resume_info = info
        args._resume_handumi = handumi
        resume_values = _recording_values_from_dataset(info, handumi)
        resume_cam_ids = resume_values.get("cam_ids")
        if isinstance(resume_cam_ids, list):
            args.cam_ids = list(resume_cam_ids)

    rig = load_optional_rig_section(args.rig_config, "recording")
    rig_cameras = load_optional_rig_section(args.rig_config, "cameras")
    aliases = {
        "width": "cam_width",
        "height": "cam_height",
        "camera_fps": "cam_fps",
        "sample_fps": "fps",
    }
    rig_values = {aliases.get(key, key): value for key, value in rig.items()}
    if "cameras" not in rig_values and rig_cameras:
        wrist_names = [
            name
            for name, entry in rig_cameras.items()
            if camera_role(name, entry) in ("left", "right")
        ]
        rig_values["cameras"] = wrist_names or list(rig_cameras)

    for name, fallback in _RECORDING_DEFAULTS.items():
        if getattr(args, name, None) is not None:
            continue
        value = resume_values.get(name, rig_values.get(name, fallback))
        if name == "cameras":
            value = _normalize_camera_list(value)
        setattr(args, name, value)

    for name in (
        "tracking_loss_timeout_s",
        "sync_lag_s",
        "max_sync_skew_s",
        "camera_stale_timeout_s",
        "gripper_stale_timeout_s",
        "sensor_loss_timeout_s",
        "feetech_sample_hz",
    ):
        if getattr(args, name, None) is None:
            setattr(
                args,
                name,
                resume_values.get(name, rig_values.get(name, _RECORDING_DEFAULTS[name])),
            )

    robot_config = _robot_metadata(args.robot)["configuration"]
    if not isinstance(robot_config, dict):
        raise SystemExit(f"Invalid robot configuration for {args.robot!r}.")
    declared_arms = robot_config.get("arms", robot_config.get("ee_links", {}))
    available = tuple(side for side in ("left", "right") if side in declared_arms)
    requested_side = getattr(args, "side", None)
    if requested_side is None:
        requested_side = resume_values.get("side", rig_values.get("side"))
    if requested_side is not None and not isinstance(requested_side, str):
        raise SystemExit("recording.side must be left, right, or both.")
    try:
        args.active_sides = resolve_active_sides(requested_side, available=available)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.side = "both" if len(args.active_sides) == 2 else args.active_sides[0]
    unknown_cameras = [name for name in args.cameras if rig_cameras and name not in rig_cameras]
    if unknown_cameras and not args.resume:
        raise SystemExit(
            f"Unknown camera(s) in {args.rig_config}: {', '.join(unknown_cameras)}. "
            f"Choose from: {', '.join(rig_cameras)}."
        )
    if not explicit["cameras"] and not args.resume:
        args.cameras = [
            name
            for name in args.cameras
            if camera_role(name, rig_cameras.get(name)) not in ("left", "right")
            or camera_role(name, rig_cameras.get(name)) in args.active_sides
        ]
    if not args.cameras:
        raise SystemExit("No cameras selected for the active side. Set --cameras to an available wrist or workspace view.")

    if args.session_calibration is None:
        configured_session = rig_values.get("session_calibration")
        if configured_session:
            args.session_calibration = Path(str(configured_session))
        elif args.resume:
            sessions = args._resume_handumi.get("recording_sessions") or []
            saved = (
                sessions[-1].get("spatial_session_calibration")
                if sessions else args._resume_handumi.get("spatial_session_calibration")
            )
            if isinstance(saved, dict) and saved.get("path"):
                args.session_calibration = Path(str(saved["path"]))

    if args.clap_control is None:
        # The shells are already connected for width capture, so hands-free
        # episode control should work without a hidden opt-in flag. Manual
        # controller mode remains authoritative when explicitly selected.
        args.clap_control = not args.skip_feetech and not args.manual_control

    return args


def _recording_values_from_dataset(
    info: dict[str, object], handumi: dict[str, object]
) -> dict[str, object]:
    values: dict[str, object] = {
        "side": "both" if len(dataset_active_sides(handumi)) == 2 else dataset_active_sides(handumi)[0],
        "device": handumi.get("recording_device"),
        "fps": info.get("fps"),
        "cam_fps": handumi.get("camera_fps", info.get("fps")),
        "tracking_loss_timeout_s": handumi.get("tracking_loss_timeout_s"),
        "sync_lag_s": handumi.get("sync_lag_s"),
        "max_sync_skew_s": handumi.get("max_sync_skew_s"),
        "camera_stale_timeout_s": handumi.get("camera_stale_timeout_s"),
        "gripper_stale_timeout_s": handumi.get("gripper_stale_timeout_s"),
        "sensor_loss_timeout_s": handumi.get("sensor_loss_timeout_s"),
        "feetech_sample_hz": handumi.get("feetech_sample_hz"),
    }
    camera_metadata = handumi.get("cameras")
    if isinstance(camera_metadata, list) and camera_metadata:
        valid = [item for item in camera_metadata if isinstance(item, dict)]
        values["cameras"] = [item.get("name") for item in valid]
        values["cam_ids"] = [item.get("index_or_path") for item in valid]

    features = info.get("features")
    if isinstance(features, dict):
        image_features = [
            feature
            for key, feature in features.items()
            if str(key).startswith("observation.images.") and isinstance(feature, dict)
        ]
        if image_features:
            shape = image_features[0].get("shape")
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                values["cam_height"], values["cam_width"] = int(shape[0]), int(shape[1])
            values["no_video"] = image_features[0].get("dtype") != "video"

    sources = handumi.get("sources")
    if isinstance(sources, dict):
        feetech = sources.get("feetech")
        if isinstance(feetech, dict) and "enabled" in feetech:
            values["skip_feetech"] = not bool(feetech["enabled"])
    audio = handumi.get("audio")
    if isinstance(audio, dict) and "enabled" in audio:
        values["record_audio"] = bool(audio["enabled"])
    robot = handumi.get("target_robot")
    if isinstance(robot, dict):
        values["robot"] = robot.get("name")
    return {key: value for key, value in values.items() if value is not None}


def _validate_args(args: argparse.Namespace) -> None:
    if (
        getattr(args, "vcodec", None) not in (None, "auto")
        and getattr(args, "encoder", "auto") != "auto"
    ):
        raise SystemExit("Use either --encoder cpu/gpu or an explicit --vcodec, not both.")
    if getattr(args, "encoder_threads", None) is not None and args.encoder_threads <= 0:
        raise SystemExit("--encoder-threads must be greater than zero.")
    if getattr(args, "encoder_queue_size", None) is not None and args.encoder_queue_size <= 0:
        raise SystemExit("--encoder-queue-size must be greater than zero.")
    if args.manual_control and args.device != "pico":
        raise SystemExit("--manual-control currently requires --device pico.")
    if args.record_audio and args.device != "pico":
        raise SystemExit("--record-audio currently requires --device pico.")
    if args.manual_control and args.start_button == "enter":
        args.start_button = "A"
        log.info("--manual-control set: using PICO A as start/stop button.")
    if args.clap_control and args.skip_feetech:
        raise SystemExit("--clap-control needs real Feetech widths; drop --skip-feetech.")
    if args.clap_control and args.manual_control:
        raise SystemExit("--clap-control and --manual-control are mutually exclusive.")
    if getattr(args, "voice_control", False) and not (
        0.0 < args.voice_confidence <= 1.0
    ):
        raise SystemExit("--voice-confidence must be within (0, 1].")
    if getattr(args, "voice_control", False) and args.manual_control:
        # Voice is on by default, so an explicit --manual-control means the
        # collector chose the controller; yield instead of erroring.
        args.voice_control = False
        log.info("--manual-control set: voice control disabled.")
    if args.episode_time_s is not None and args.episode_time_s <= 0:
        raise SystemExit("--episode-time-s must be greater than zero when set.")
    if args.tracking_loss_timeout_s <= 0:
        raise SystemExit("--tracking-loss-timeout-s must be greater than zero.")
    for name in (
        "sync_lag_s",
        "max_sync_skew_s",
        "camera_stale_timeout_s",
        "gripper_stale_timeout_s",
        "sensor_loss_timeout_s",
        "feetech_sample_hz",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be greater than zero.")


def _resolve_spatial_session_metadata(
    args: argparse.Namespace,
) -> dict[str, object] | None:
    resume_handumi = getattr(args, "_resume_handumi", None)
    embedded = (
        resume_handumi.get("spatial_session_calibration")
        if isinstance(resume_handumi, dict)
        else None
    )
    if args.resume and embedded is not None and args.session_calibration is None:
        raise SystemExit(
            "Resuming a table-calibrated dataset requires a current session file. "
            "Pass --session-calibration PATH (the same calibration may be reused "
            "while the VR origin is unchanged)."
        )
    try:
        return session_calibration_metadata(args.session_calibration)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid session calibration: {exc}") from exc


def _print_recording_plan(
    args: argparse.Namespace,
    *,
    camera_specs: list[dict[str, object]],
    encoder: _VideoEncoderSelection | None,
    calibration_source: str,
    spatial_session_metadata: dict[str, object] | None,
) -> None:
    if encoder is None:
        encoder_label = "disabled (individual images)"
    else:
        kind = "hardware" if encoder.hardware else "CPU"
        thread_label = "codec-managed" if encoder.threads is None else str(encoder.threads)
        encoder_label = f"{encoder.vcodec} ({kind}, threads/camera={thread_label})"
    camera_label = ", ".join(
        (
            f"{spec['name']}={spec['id']} ({spec.get('type', 'opencv')}, "
            f"{camera_output_size(spec, default_width=args.cam_width, default_height=args.cam_height)[0]}x"
            f"{camera_output_size(spec, default_width=args.cam_width, default_height=args.cam_height)[1]}"
            f"/{spec.get('fps', args.cam_fps)} fps)"
        )
        for spec in camera_specs
    )
    mode = "resume" if args.resume else "new dataset"
    workspace = "table-calibrated" if spatial_session_metadata else "HMD-recentered"
    episodes = "until stopped" if args.num_episodes <= 0 else str(args.num_episodes)
    print("\nRecording plan")
    print(f"  Dataset:  {args.output_dir} ({mode})")
    print(f"  Device:   {args.device}; robot profile: {args.robot}")
    print(f"  Sides:    {', '.join(args.active_sides)}")
    print(f"  Cameras:  {camera_label}")
    print(f"  Rows:     {args.fps} fps; {episodes} episode(s)")
    print(f"  Feetech:  {'disabled' if args.skip_feetech else 'enabled'}")
    print(f"  Encoder:  {encoder_label}")
    print(f"  Audio:    {'PICO PCM WAV' if args.record_audio else 'disabled'}")
    print(f"  Workspace: {workspace}; Controller->TCP: {calibration_source}")
    if spatial_session_metadata:
        print(
            f"  Session: {spatial_session_metadata.get('path')} "
            f"created={spatial_session_metadata.get('created_at')} "
            f"sha256={spatial_session_metadata.get('sha256')}"
        )
    if args.dry_run:
        print("  Result:   plan resolved; hardware was not opened")


def main() -> None:
    # Captured before any hardware/dataset setup so the dashboard's "program
    # runtime" reflects the whole run, not just the recording loop.
    program_start_s = time.perf_counter()
    args = _resolve_recording_args(parse_args())
    _validate_args(args)
    play_sounds = not args.no_sounds

    camera_names = _selected_camera_names(args)
    resume_handumi = getattr(args, "_resume_handumi", None)
    explicit = getattr(args, "_explicit_recording", {})
    embedded_robot = (
        resume_handumi.get("target_robot") if isinstance(resume_handumi, dict) else None
    )
    if args.resume and not explicit.get("robot") and isinstance(embedded_robot, dict):
        robot_metadata = embedded_robot
    else:
        robot_metadata = _robot_metadata(args.robot)

    embedded_tcp = (
        resume_handumi.get("controller_tcp_calibration")
        if isinstance(resume_handumi, dict)
        else None
    )
    if (
        args.resume
        and not explicit.get("robot")
        and not explicit.get("controller_tcp_calibration")
        and isinstance(embedded_tcp, dict)
    ):
        calibration_metadata = embedded_tcp
        calibration_source = "embedded dataset snapshot"
    else:
        calibration_metadata, calibration_source = _recording_tcp_calibration_metadata(
            robot_metadata=robot_metadata,
            device=args.device,
            explicit_path=args.controller_tcp_calibration,
        )
    log.info("Controller->TCP setup: %s", calibration_source)
    spatial_session_metadata = _resolve_spatial_session_metadata(args)

    cam_ids = resolve_camera_ids(
        args.cam_ids,
        args.rig_config,
        camera_names=camera_names,
    )
    _validate_unique_camera_ids(camera_names, cam_ids)
    camera_specs, _ = build_camera_specs(
        cam_ids,
        camera_names=camera_names,
        laptop_camera=False,
        laptop_cam_id=0,
        laptop_cam_name="laptop",
        rig_config=args.rig_config,
        default_fps=args.cam_fps,
        default_width=args.cam_width,
        default_height=args.cam_height,
    )
    cam_names = [spec["name"] for spec in camera_specs]
    use_videos = not args.no_video
    features = build_features(
        cam_names,
        args.cam_width,
        args.cam_height,
        use_videos,
        camera_specs=camera_specs,
        record_audio=bool(args.record_audio),
    )
    encoder_selection: _VideoEncoderSelection | None = None
    dataset_vcodec = _SOFTWARE_VIDEO_CODEC
    encoder_queue_size = args.encoder_queue_size or max(1, args.fps)
    if use_videos:
        output_sizes = [
            camera_output_size(
                spec,
                default_width=args.cam_width,
                default_height=args.cam_height,
            )
            for spec in camera_specs
        ]
        encoder_selection = _select_video_encoder(
            policy=args.encoder,
            requested_vcodec=args.vcodec,
            width=max(width for width, _ in output_sizes),
            height=max(height for _, height in output_sizes),
            fps=args.fps,
            camera_count=len(cam_names),
            requested_threads=args.encoder_threads,
        )
        dataset_vcodec = encoder_selection.vcodec
        encoder_kind = "hardware" if encoder_selection.hardware else "CPU"
        thread_detail = (
            "codec-managed threads"
            if encoder_selection.threads is None
            else f"{encoder_selection.threads} thread(s) per camera"
        )
        log.info(
            "Encoder: %s (%s, streaming, %s).",
            dataset_vcodec,
            encoder_kind,
            thread_detail,
        )
    else:
        log.info("Encoder: disabled (--no-video stores individual images).")
    if args.resume:
        _validate_resume_target(
            args.output_dir,
            fps=args.fps,
            features=features,
            vcodec=dataset_vcodec,
            handumi=_resume_handumi_metadata(
                args=args,
                camera_specs=camera_specs,
                calibration_metadata=calibration_metadata,
                spatial_session_metadata=spatial_session_metadata,
                robot_metadata=robot_metadata,
            ),
        )

    _print_recording_plan(
        args,
        camera_specs=camera_specs,
        encoder=encoder_selection,
        calibration_source=calibration_source,
        spatial_session_metadata=spatial_session_metadata,
    )
    if args.dry_run:
        return

    log.info("--- Tracking setup ---")
    calibration = ControllerTcpCalibration(
        left=IDENTITY_POSE7.astype(np.float32).copy(),
        right=IDENTITY_POSE7.astype(np.float32).copy(),
        source=None,
    )
    tracker = build_tracker(args, calibration)
    if spatial_session_metadata is not None:
        session_device = str(spatial_session_metadata.get("tracking_device") or "")
        if session_device and session_device != args.device:
            raise SystemExit(
                f"Session calibration is for {session_device}, "
                f"but --device {args.device} was selected."
            )
        set_workspace = getattr(tracker, "set_workspace_from_device_pose", None)
        if set_workspace is None:
            raise SystemExit("Selected tracking backend cannot apply a table calibration.")
        table_from_device = pose7_from_dict(spatial_session_metadata["table_from_device"])
        set_workspace(table_from_device, locked=True)
    tracker.start()

    log.info("--- Camera setup ---")
    cameras = connect_cameras(
        camera_specs,
        fps=args.cam_fps,
        width=args.cam_width,
        height=args.cam_height,
        zero_non_laptop=False,
    )
    rerun = _RecordingRerun(cam_names, args.fps) if args.rerun else None

    log.info("--- Feetech setup ---")
    gripper_pair = connect_feetech(args)
    grippers = None
    if gripper_pair is not None:
        grippers = FeetechGripperSampler(
            gripper_pair,
            sample_hz=args.feetech_sample_hz,
        )
        grippers.start()

    log.info("--- Dataset setup ---")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_kwargs = {
        "repo_id": args.repo_id,
        "root": args.output_dir,
        "image_writer_processes": 0,
        "image_writer_threads": 0 if use_videos else max(1, 4 * len(cam_names)),
        "vcodec": dataset_vcodec,
        "streaming_encoding": use_videos,
        "encoder_queue_maxsize": encoder_queue_size,
        "encoder_threads": (
            encoder_selection.threads if encoder_selection is not None else None
        ),
    }
    if args.resume:
        dataset = LeRobotDataset.resume(**dataset_kwargs)
        log.info(
            "Dataset resumed at: %s (%d existing episode(s); recording %s additional).",
            dataset.root,
            dataset.num_episodes,
            "until stopped" if args.num_episodes <= 0 else args.num_episodes,
        )
    else:
        dataset = LeRobotDataset.create(
            **dataset_kwargs,
            fps=args.fps,
            robot_type="handumi_raw",
            features=features,
            use_videos=use_videos,
        )
        log.info("Dataset created at: %s", dataset.root)
    if use_videos:
        _install_strict_streaming_encoder(dataset)

    stop_event = threading.Event()

    def _on_signal(signum, frame):
        log.info("Signal received - discarding active episode and stopping ...")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    voice = _start_voice_control(args, clap_control=args.clap_control)
    escape_listener = _EscapeStopListener(stop_event)
    if args.clap_control or voice is not None:
        escape_listener.start()

    def announce(text: str, *, blocking: bool = False) -> None:
        """Speak recorder status, deafening voice control while it talks.

        The announcements ("Stop recording", "Restart recording") are the
        commands, so an unmuted microphone would hear the recorder command
        itself.
        """
        if voice is not None and play_sounds:
            voice.mute(speech_duration_s(text))
        log_say(text, play_sounds=play_sounds, blocking=blocking)

    recorded = 0
    existing_episodes = int(dataset.num_episodes)
    existing_frames = int(getattr(dataset, "num_frames", 0))
    controls: list[tuple[str, str]] = []
    if voice is not None:
        controls.extend(
            (
                ('Say "start recording"', "Start an episode"),
                ('Say "stop recording"', "Save the current episode"),
                ('Say "restart"', "Discard and re-record the episode"),
            )
        )
    if args.clap_control:
        controls.extend(
            (
                ("Double-squeeze RIGHT", "Save and start the next episode"),
                ("Double-squeeze LEFT", "Discard and re-record the episode"),
                ("Double-squeeze BOTH", "Discard current episode and finish"),
            )
        )
    if args.manual_control:
        controls.extend(
            (
                (args.start_button, "Save / advance"),
                (args.repeat_button, "Discard / re-record"),
                (args.finish_button, "Save and finish session"),
            )
        )
    elif voice is None and not args.clap_control:
        controls.append((str(args.start_button), "Start recording"))
    controls.append(("Esc / Ctrl+C", "Discard current episode and stop"))
    dashboard = _RecordingDashboard(
        task=args.task,
        dataset=str(dataset.root),
        fps=args.fps,
        existing_episodes=existing_episodes,
        existing_frames=existing_frames,
        controls=tuple(controls),
        started_at_s=program_start_s,
    )
    web_dashboard = None
    if args.web_dashboard:
        # The browser shows the live view, so the terminal keeps plain log
        # lines instead of a full-screen panel that hides warnings.
        dashboard._console = None
        web_dashboard = WebDashboard(
            status=dashboard,
            tracker=tracker,
            grippers=grippers,
            voice=voice if isinstance(voice, BrowserVoiceListener) else None,
            cameras=cameras,
            camera_names=cam_names,
            port=args.web_dashboard,
        )
        web_dashboard.start()
    dashboard.start()
    clap_detector = DoubleClapDetector() if args.clap_control else None
    clap_arbiter = BilateralClapArbiter() if args.clap_control else None
    # None: wait for an explicit trigger. "repeat": restart the same episode
    # immediately. "advance": a clap-driven save already asked for the next
    # episode to start right away, without waiting for another trigger.
    pending_start: str | None = None
    audio_recorder = (
        PicoAudioRecorder(lambda: getattr(tracker, "xrt", None), Path(dataset.root))
        if args.record_audio
        else None
    )
    fatal_recording_error: Exception | None = None
    try:
        if audio_recorder is not None:
            audio_recorder.start()
        while (args.num_episodes <= 0 or recorded < args.num_episodes) and not stop_event.is_set():
            ep_num = dataset.num_episodes + 1
            ep_total = (
                "inf"
                if args.num_episodes <= 0
                else str(existing_episodes + args.num_episodes)
            )
            log.info("--- Episode %d/%s ---", ep_num, ep_total)
            dashboard.set_episode(ep_num, ep_total)
            transition_announcement: str | None = None
            if rerun is not None:
                rerun.set_status("WAITING", f"Episode {ep_num}/{ep_total}: waiting to start")
            if voice is not None or clap_detector is not None:
                if pending_start == "repeat":
                    pending_start = None
                    transition_announcement = f"Restarting episode {ep_num}"
                    log.info("  Restarting episode %d immediately ...", ep_num)
                    if rerun is not None:
                        rerun.set_status("RESTARTED", f"Episode {ep_num}/{ep_total}: restarting now")
                else:
                    if pending_start == "advance":
                        pending_start = None
                        transition_announcement = f"Next episode, episode {ep_num}"
                        log.info("  Continuing directly to episode %d ...", ep_num)
                        if rerun is not None:
                            rerun.set_status(
                                "RECORDING",
                                f"Episode {ep_num}/{ep_total}: continuing directly",
                            )
                    else:
                        log.info(
                            "  %s to start episode %d ...",
                            _start_prompt(voice, clap_detector),
                            ep_num,
                        )
                        if not _wait_for_start_trigger(
                            grippers,
                            clap_detector,
                            voice,
                            stop_event,
                            dashboard=dashboard,
                        ):
                            break
                    # A calibrated table workspace is locked and ignores this
                    # legacy HMD recenter; uncalibrated sessions retain it.
                    reset_workspace = getattr(tracker, "reset_workspace", None)
                    if reset_workspace is not None:
                        reset_workspace()
                if clap_detector is not None:
                    clap_detector.reset()
                if clap_arbiter is not None:
                    clap_arbiter.reset()
            elif args.manual_control:
                action = wait_for_manual_start(
                    getattr(tracker, "xrt"),
                    start_button=args.start_button,
                    finish_button=args.finish_button,
                    threshold=args.start_threshold,
                    stop_event=stop_event,
                )
                if action == "finish":
                    break
            elif args.start_button == "enter":
                input(f"  Press ENTER to start recording episode {ep_num} ...")
            elif args.device == "pico":
                if not wait_for_start_button(
                    getattr(tracker, "xrt"),
                    button=args.start_button,
                    threshold=args.start_threshold,
                    stop_event=stop_event,
                ):
                    break
            else:
                raise SystemExit("--start-button other than enter currently requires --device pico.")

            if not _wait_for_tracking(
                tracker, stop_event, active_sides=args.active_sides,
                dashboard=dashboard, grippers=grippers,
            ):
                break
            # Finish the cue before capture starts so fast transitions cannot
            # overlap multiple asynchronous TTS processes.
            announce(
                transition_announcement or f"Recording episode {ep_num}",
                blocking=True,
            )
            dashboard.recording()
            if audio_recorder is not None:
                audio_recorder.begin_episode(time.monotonic_ns())
            if rerun is not None:
                rerun.set_status("RECORDING", f"Episode {ep_num}/{ep_total} is being recorded")
            try:
                n_frames, status, advance_after_save = record_episode(
                    dataset=dataset,
                    cameras=cameras,
                    cam_names=cam_names,
                    tracker=tracker,
                    grippers=grippers,
                    episode_time_s=args.episode_time_s,
                    fps=args.fps,
                    task=args.task,
                    cam_width=args.cam_width,
                    cam_height=args.cam_height,
                    stop_event=stop_event,
                    manual_control=args.manual_control,
                    start_button=args.start_button,
                    repeat_button=args.repeat_button,
                    finish_button=args.finish_button,
                    start_threshold=args.start_threshold,
                    clap_detector=clap_detector,
                    clap_arbiter=clap_arbiter,
                    voice=voice,
                    tracking_loss_timeout_s=args.tracking_loss_timeout_s,
                    sync_lag_s=args.sync_lag_s,
                    max_sync_skew_s=args.max_sync_skew_s,
                    camera_stale_timeout_s=args.camera_stale_timeout_s,
                    gripper_stale_timeout_s=args.gripper_stale_timeout_s,
                    sensor_loss_timeout_s=args.sensor_loss_timeout_s,
                    rerun=rerun,
                    dashboard=dashboard,
                    audio_recorder=audio_recorder,
                    active_sides=args.active_sides,
                )
            except Exception as exc:
                # A hardware disconnect (cable pull, USB drop, ...) or any
                # other unexpected failure must not lose already-saved
                # episodes: discard only the one that was in flight and stop.
                fatal_recording_error = exc
                if isinstance(exc, (CameraCaptureError, PicoTrackingFrozenError)):
                    log.error(
                        "Capture failure during episode %d; discarding it and "
                        "aborting recording: %s",
                        ep_num,
                        exc,
                    )
                else:
                    log.exception(
                        "Unexpected failure during episode %d; discarding it "
                        "and stopping.",
                        ep_num,
                    )
                if audio_recorder is not None:
                    audio_recorder.cancel_episode()
                dataset.clear_episode_buffer()
                announce("Episode discarded")
                if rerun is not None:
                    rerun.set_status(
                        "DISCARDED",
                        f"Episode {ep_num}/{ep_total}: unexpected error ({exc}); stopping",
                    )
                stop_event.set()
                break
            if status == "repeat":
                if audio_recorder is not None:
                    audio_recorder.cancel_episode()
                log.warning("Episode restart requested (%d frames discarded).", n_frames)
                if rerun is not None:
                    rerun.set_status(
                        "RESTARTED",
                        f"Episode {ep_num}/{ep_total}: {n_frames} frames discarded; restarting",
                    )
                dataset.clear_episode_buffer()
                dashboard.discarded(n_frames, "restart requested")
                pending_start = "repeat"
                continue
            if n_frames == 0 or status in {
                "tracking_lost",
                "sensor_unhealthy",
                "encoder_unhealthy",
                "interrupted",
                "session_finished",
            }:
                if audio_recorder is not None:
                    audio_recorder.cancel_episode()
                log.warning("Episode discarded (%s, %d frames).", status, n_frames)
                announce("Episode discarded")
                if rerun is not None:
                    rerun.set_status(
                        "DISCARDED",
                        f"Episode {ep_num}/{ep_total}: {status} after {n_frames} frames",
                    )
                dataset.clear_episode_buffer()
                dashboard.discarded(n_frames, status)
                if status in {"finish", "interrupted", "session_finished"}:
                    break
                continue
            if use_videos:
                try:
                    _prepare_streaming_episode(dataset, n_frames)
                except StreamingEncodingError as exc:
                    log.error("Episode discarded before commit: %s", exc)
                    announce("Episode discarded")
                    dataset.clear_episode_buffer()
                    dashboard.discarded(n_frames, "video encoding failed")
                    if status == "finish":
                        break
                    continue
            if audio_recorder is not None:
                try:
                    audio_recorder.prepare_episode(expected_frames=n_frames)
                except AudioCaptureError as exc:
                    log.error("Episode discarded before commit: %s", exc)
                    announce("Episode discarded")
                    dataset.clear_episode_buffer()
                    audio_recorder.cancel_episode()
                    dashboard.discarded(n_frames, "audio capture failed")
                    if status == "finish":
                        break
                    continue
            episode_index = int(dataset.num_episodes)
            audio_path = None
            if audio_recorder is not None:
                audio_path = audio_recorder.commit_episode(
                    episode_index,
                    chunks_size=int(dataset.meta.chunks_size),
                )
            try:
                dataset.save_episode()
            except BaseException:
                if audio_path is not None:
                    audio_path.unlink(missing_ok=True)
                raise
            recorded += 1
            dashboard.saved(n_frames)
            log.info("Episode %d saved (%d frames).", ep_num, n_frames)
            announce(f"Episode {ep_num} saved, {n_frames} frames")
            if rerun is not None:
                rerun.set_status(
                    "SAVED", f"Episode {ep_num}/{ep_total}: {n_frames} frames saved"
                )
            if advance_after_save:
                pending_start = "advance"
            if status == "finish":
                break
    finally:
        dashboard.stop()
        if web_dashboard is not None:
            web_dashboard.stop()
        escape_listener.stop()
        if voice is not None:
            voice.stop()
        if rerun is not None:
            rerun.set_status("STOPPED", f"Recording stopped: {recorded} episode(s) saved")
        announce("Stop recording", blocking=True)
        log.info("--- Finalising ---")
        finalization_error: BaseException | None = None
        try:
            dataset.finalize()
            root = Path(dataset.root)
            handumi_metadata = _resume_handumi_metadata(
                args=args,
                camera_specs=camera_specs,
                calibration_metadata=calibration_metadata,
                spatial_session_metadata=spatial_session_metadata,
                robot_metadata=robot_metadata,
            )
            handumi_metadata["sources"] = _capture_sources_metadata(
                camera_specs,
                cameras,
                grippers,
                audio_enabled=bool(args.record_audio),
            )
            _record_session_history(
                handumi_metadata,
                previous=resume_handumi,
                first_episode=existing_episodes,
                end_episode=existing_episodes + recorded,
            )
            updated_info = _update_info_json(
                root,
                handumi_metadata,
            )
            if updated_info is not None:
                dataset.meta.info = updated_info
            card_kwargs = _write_dataset_readme(
                root,
                repo_id=args.repo_id,
                task=args.task,
                license_id=args.dataset_license,
            )
            _validate_finalized_lerobot_dataset(root)
            log.info("LeRobot v3 integrity validation passed.")
            if args.push_to_hub:
                dataset.push_to_hub(
                    license=args.dataset_license,
                    tags=["HandUMI"],
                    **cast(dict[str, Any], card_kwargs),
                )
        except BaseException as exc:
            finalization_error = exc
            log.exception("Dataset finalization failed; do not upload this dataset.")
        finally:
            if audio_recorder is not None:
                audio_recorder.close()
            disconnect_cameras(cameras)
            if grippers is not None:
                grippers.stop()
            if gripper_pair is not None:
                gripper_pair.close()
            tracker.stop()
        if finalization_error is not None:
            raise finalization_error
        if fatal_recording_error is not None:
            raise SystemExit(
                f"Recording aborted; active episode was discarded: "
                f"{fatal_recording_error}"
            ) from fatal_recording_error
        log.info("Done. Recorded %d episode(s). Dataset at: %s", recorded, dataset.root)
        log_say("Exiting", play_sounds=play_sounds)


def build_tracker(
    args: argparse.Namespace, calibration, *, reset_workspace_on_x: bool = True
) -> TrackingProvider:
    if args.device == "pico":
        transport = "wifi" if args.pico_wifi else "adb"
        return PicoTrackingProvider(
            calibration=calibration,
            mode=args.pico_mode,
            transport=transport,
            skip_adb_check=args.skip_adb_check,
        )

    base = MetaQuestConfig.from_yaml(args.rig_config)
    config = MetaQuestConfig(
        quest_ip=args.quest_ip if args.quest_ip is not None else base.quest_ip,
        tcp_port=args.tcp_port if args.tcp_port is not None else base.tcp_port,
        sync_port=args.sync_port if args.sync_port is not None else base.sync_port,
        connect_retry_s=base.connect_retry_s,
        frame_stale_timeout_s=base.frame_stale_timeout_s,
    )
    return MetaQuestTrackingProvider(
        config=config, calibration=calibration, reset_workspace_on_x=reset_workspace_on_x
    )


def connect_feetech(args: argparse.Namespace) -> FeetechGripperPair | None:
    if args.skip_feetech:
        log.info("Feetech disabled: gripper widths will be zero-filled.")
        return None
    feetech_config = load_config(args.rig_config)
    if args.feetech_port is not None:
        feetech_config = type(feetech_config)(
            port=args.feetech_port,
            baudrate=feetech_config.baudrate,
            protocol_version=feetech_config.protocol_version,
            left=feetech_config.left,
            right=feetech_config.right,
        )
    sides = getattr(args, "active_sides", None) or resolve_active_sides(getattr(args, "side", None))
    assert_calibrated(feetech_config, source=user_calibration_path(), active_sides=sides)
    grippers = FeetechGripperPair(feetech_config, active_sides=sides)
    try:
        grippers.open()
    except FeetechUnavailableError as exc:
        raise SystemExit(str(exc)) from exc
    return grippers


def _start_voice_control(
    args: argparse.Namespace, *, clap_control: bool
) -> VoiceCommandListener | None:
    """Bring up voice control, or explain why the run cannot continue.

    Voice is the default control, so a missing microphone is fatal unless the
    collector already has another hands-free control (--clap-control) or opted
    out of voice entirely.
    """
    if getattr(args, "web_dashboard", None):
        # The dashboard's buttons feed this listener too, so it starts even
        # with --no-voice-control.
        browser_listener = BrowserVoiceListener(confidence=args.voice_confidence)
        try:
            browser_listener.start()
        except VoiceUnavailableError as exc:
            raise SystemExit(f"Browser voice control unavailable.\n{exc}") from exc
        return browser_listener
    if not args.voice_control:
        return None
    listener = VoiceCommandListener(
        device=args.voice_device,
        confidence=args.voice_confidence,
    )
    try:
        listener.start()
    except VoiceUnavailableError as exc:
        if clap_control:
            log.warning("Voice control unavailable (%s); using gripper squeezes only.", exc)
            return None
        raise SystemExit(f"Voice control unavailable.\n{exc}") from exc
    return listener


def _wait_for_start_trigger(
    grippers: FeetechGripperSampler | FeetechGripperPair | None,
    clap_detector: DoubleClapDetector | None,
    voice: VoiceCommandListener | None,
    stop_event: threading.Event,
    *,
    dashboard: _RecordingDashboard | None = None,
) -> bool:
    """Block until voice or a right double-squeeze starts the episode.

    Either control alone is enough, so a noisy room or a busy pair of hands
    never leaves the collector without a way to start.
    """
    while not stop_event.is_set():
        if voice is not None and voice.poll() == "start":
            voice.drain()
            return True
        if clap_detector is not None:
            widths = _latest_gripper_widths(grippers)
            if dashboard is not None:
                dashboard.update_grippers(widths.left_mm, widths.right_mm)
            clap_side = clap_detector.update_side(
                widths.left_mm, widths.right_mm, time.perf_counter()
            )
            if clap_side is not None:
                log.info(
                    "Double-squeeze detected while waiting: %s "
                    "(left=%.1f mm, right=%.1f mm).",
                    clap_side.upper(),
                    widths.left_mm,
                    widths.right_mm,
                )
            if clap_side == "right":
                return True
        time.sleep(0.02)
    return False


def _start_prompt(
    voice: VoiceCommandListener | None, clap_detector: DoubleClapDetector | None
) -> str:
    options = []
    if voice is not None:
        options.append('say "start recording"')
    if clap_detector is not None:
        options.append("double-squeeze the right gripper")
    return " or ".join(options).capitalize()


def _wait_for_clap(
    grippers: FeetechGripperSampler | FeetechGripperPair | None,
    clap_detector: DoubleClapDetector,
    stop_event: threading.Event,
    *,
    side: str = "right",
) -> bool:
    """Poll widths until ``side`` double-claps (or ``stop_event`` sets)."""
    while not stop_event.is_set():
        widths = _latest_gripper_widths(grippers)
        triggered = clap_detector.update_side(
            widths.left_mm, widths.right_mm, time.perf_counter()
        )
        if triggered == side:
            return True
        time.sleep(0.02)
    return False


def _latest_gripper_widths(
    grippers: FeetechGripperSampler | FeetechGripperPair | None,
) -> GripperWidths:
    if grippers is None:
        return zero_gripper_widths()
    if isinstance(grippers, FeetechGripperSampler):
        sample = grippers.latest()
        return zero_gripper_widths() if sample is None else sample.widths
    return grippers.read_normalized_widths()


def _selected_camera_names(args: argparse.Namespace) -> list[str]:
    return list(args.cameras)


def _capture_sources_metadata(
    camera_specs: list[dict[str, object]],
    cameras: Sequence[object | None],
    grippers: object | None,
    *,
    audio_enabled: bool = False,
) -> dict[str, object]:
    """Store source enablement once instead of repeating it on every row."""
    return {
        "tracking": {"enabled": True},
        "feetech": {"enabled": grippers is not None},
        "audio": {"enabled": audio_enabled},
        "cameras": {
            str(spec["name"]): {"enabled": camera is not None}
            for spec, camera in zip(camera_specs, cameras, strict=True)
        },
    }


def _robot_metadata(name: str, config_dir: Path = ROBOT_CONFIG_DIR) -> dict[str, object]:
    from handumi.robots.registry import robot_config_metadata

    return robot_config_metadata(name, config_dir)


def _recording_tcp_calibration_metadata(
    *,
    robot_metadata: dict[str, object],
    device: str,
    explicit_path: Path | None,
) -> tuple[dict[str, object], str]:
    """Resolve and snapshot the robot/gripper TCP setup used for recording."""
    robot = str(robot_metadata["name"])
    configuration = robot_metadata.get("configuration")
    if not isinstance(configuration, dict):
        raise SystemExit(f"Robot {robot!r} has invalid configuration metadata.")

    configured = configuration.get("controller_tcp_calibrations") or {}
    configured_path_value = configured.get(device) if isinstance(configured, dict) else None
    associated_with_robot_tool = configured_path_value is not None
    if explicit_path is not None:
        calibration_path = explicit_path
        source = f"explicit {calibration_path} for {robot}/{device}"
    elif configured_path_value is not None:
        calibration_path = Path(str(configured_path_value))
        source = f"configured {robot}/{device}: {calibration_path}"
    else:
        calibration_path = calibration_path_for_device(device)
        source = f"legacy device fallback {device}: {calibration_path}"
        log.warning(
            "Robot %s has no controller_tcp_calibrations.%s entry; using %s "
            "without treating it as a verified robot/gripper pairing.",
            robot,
            device,
            calibration_path,
        )

    tool = configuration.get("handumi_tool") or {}
    if not isinstance(tool, dict):
        raise SystemExit(f"Robot {robot!r} handumi_tool must be a mapping.")
    gripper = str(tool["gripper"]) if tool.get("gripper") else None
    controller_mount = (
        str(tool["controller_mount"]) if tool.get("controller_mount") else None
    )
    if associated_with_robot_tool and (gripper is None or controller_mount is None):
        raise SystemExit(
            f"Robot {robot!r} configures a {device} Controller->TCP calibration "
            "but is missing handumi_tool.gripper or handumi_tool.controller_mount."
        )

    metadata = controller_tcp_calibration_metadata(
        calibration_path,
        applied_to_state=False,
        source_robot=robot,
        source_gripper=gripper if associated_with_robot_tool else None,
        tracking_device=device,
        controller_mount=controller_mount if associated_with_robot_tool else None,
    )
    return metadata, source


def _validate_unique_camera_ids(
    camera_names: list[str],
    camera_ids: list[int | str],
) -> None:
    duplicates = {
        camera_id
        for camera_id in camera_ids
        if camera_ids.count(camera_id) > 1
    }
    if duplicates:
        mappings = ", ".join(
            f"{name}={camera_id}" for name, camera_id in zip(camera_names, camera_ids)
        )
        raise SystemExit(
            f"Selected cameras must use distinct devices ({mappings}). "
            "Fix the cameras section in configs/rig.yaml."
        )


_LEROBOT_DEFAULT_FEATURE_KEYS = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}


def _resume_handumi_metadata(
    *,
    args: argparse.Namespace,
    camera_specs: list[dict[str, object]],
    calibration_metadata: dict[str, object],
    spatial_session_metadata: dict[str, object] | None,
    robot_metadata: dict[str, object],
) -> dict[str, object]:
    """Build the session-wide metadata that must remain stable when appending."""
    embedded = getattr(args, "_resume_handumi", None)
    embedded_sources = embedded.get("sources") if isinstance(embedded, dict) else None
    sources = (
        embedded_sources
        if bool(getattr(args, "resume", False)) and isinstance(embedded_sources, dict)
        else {
            "tracking": {"enabled": True},
            "feetech": {"enabled": not args.skip_feetech},
            "audio": {"enabled": bool(getattr(args, "record_audio", False))},
            "cameras": {
                str(spec["name"]): {"enabled": True} for spec in camera_specs
            },
        }
    )
    is_legacy_resume = bool(getattr(args, "resume", False)) and isinstance(embedded, dict)

    def stable_value(key: str, current: object) -> object:
        if is_legacy_resume and isinstance(embedded, dict) and key not in embedded:
            return None
        return current

    return {
        "recording_device": args.device,
        "active_sides": list(getattr(args, "active_sides", ("left", "right"))),
        "audio": stable_value(
            "audio", audio_metadata(bool(getattr(args, "record_audio", False)))
        ),
        "camera_fps": stable_value("camera_fps", getattr(args, "cam_fps", None)),
        "tracking_loss_timeout_s": stable_value(
            "tracking_loss_timeout_s", getattr(args, "tracking_loss_timeout_s", None)
        ),
        "tracking_schema": HANDUMI_TRACKING_SCHEMA,
        "tracking_workspace": (
            "table" if spatial_session_metadata is not None else "hmd_recentered"
        ),
        "state_semantics": HANDUMI_STATE_SEMANTICS,
        "capture_schema": HANDUMI_CAPTURE_SCHEMA,
        "sync_lag_s": args.sync_lag_s,
        "max_sync_skew_s": args.max_sync_skew_s,
        "camera_stale_timeout_s": args.camera_stale_timeout_s,
        "gripper_stale_timeout_s": args.gripper_stale_timeout_s,
        "sensor_loss_timeout_s": stable_value(
            "sensor_loss_timeout_s", getattr(args, "sensor_loss_timeout_s", None)
        ),
        "feetech_sample_hz": stable_value(
            "feetech_sample_hz", getattr(args, "feetech_sample_hz", None)
        ),
        "cameras": [
            {
                "name": spec["name"],
                "index_or_path": spec["id"],
                "type": spec.get("type", "opencv"),
                "capture_width": spec.get(
                    "width", getattr(args, "cam_width", 640)
                ),
                "capture_height": spec.get(
                    "height", getattr(args, "cam_height", 480)
                ),
                "output_width": camera_output_size(
                    spec,
                    default_width=getattr(args, "cam_width", 640),
                    default_height=getattr(args, "cam_height", 480),
                )[0],
                "output_height": camera_output_size(
                    spec,
                    default_width=getattr(args, "cam_width", 640),
                    default_height=getattr(args, "cam_height", 480),
                )[1],
                "fps": spec.get("fps", getattr(args, "cam_fps", 30)),
            }
            for spec in camera_specs
        ],
        "sources": sources,
        "controller_tcp_calibration": calibration_metadata,
        "spatial_session_calibration": spatial_session_metadata,
        "target_robot": robot_metadata,
    }


def _canonical_feature(feature: object) -> tuple[object, tuple[object, ...], object]:
    if not isinstance(feature, dict):
        return None, (), None
    shape = feature.get("shape")
    names = feature.get("names")
    return (
        feature.get("dtype"),
        tuple(shape) if isinstance(shape, (list, tuple)) else (),
        tuple(names) if isinstance(names, (list, tuple)) else names,
    )


def _canonical_video_codec(vcodec: str) -> str:
    if vcodec == "libsvtav1":
        return "av1"
    if "h264" in vcodec:
        return "h264"
    if "hevc" in vcodec:
        return "hevc"
    return vcodec


def _metadata_fingerprint(value: object, keys: tuple[str, ...]) -> object:
    if value is None:
        return None
    if not isinstance(value, dict):
        return value
    return {key: value.get(key) for key in keys}


def _validate_resume_target(
    root: Path,
    *,
    fps: int,
    features: dict[str, dict],
    vcodec: str | None = None,
    handumi: dict[str, object],
) -> None:
    """Reject incomplete or session-incompatible datasets before hardware starts."""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"Cannot resume: dataset directory does not exist: {root}")
    try:
        _validate_finalized_lerobot_dataset(root)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Cannot resume incomplete dataset at {root}: {exc}") from exc

    try:
        info = json.loads((root / "meta" / "info.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot resume: invalid meta/info.json in {root}: {exc}") from exc

    mismatches: list[str] = []
    if int(info.get("fps", 0)) != int(fps):
        mismatches.append(f"fps: dataset={info.get('fps')!r}, requested={fps!r}")
    if info.get("robot_type") != "handumi_raw":
        mismatches.append(
            f"robot_type: dataset={info.get('robot_type')!r}, requested='handumi_raw'"
        )

    actual_features = info.get("features") or {}
    if not isinstance(actual_features, dict):
        mismatches.append("features: dataset metadata is not a mapping")
        actual_features = {}
    actual_custom_keys = set(actual_features) - _LEROBOT_DEFAULT_FEATURE_KEYS
    expected_keys = set(features)
    missing = sorted(expected_keys - actual_custom_keys)
    extra = sorted(actual_custom_keys - expected_keys)
    if missing:
        mismatches.append(f"features missing from dataset: {', '.join(missing)}")
    if extra:
        mismatches.append(f"unexpected dataset features: {', '.join(extra)}")
    for key in sorted(expected_keys & actual_custom_keys):
        actual = _canonical_feature(actual_features[key])
        expected = _canonical_feature(features[key])
        if actual != expected:
            mismatches.append(f"feature {key}: dataset={actual!r}, requested={expected!r}")
        feature = actual_features[key]
        if (
            vcodec not in (None, "auto")
            and isinstance(feature, dict)
            and feature.get("dtype") == "video"
        ):
            video_info = feature.get("info") or {}
            actual_codec = video_info.get("video.codec") if isinstance(video_info, dict) else None
            requested_codec = _canonical_video_codec(vcodec)
            if actual_codec is not None and actual_codec != requested_codec:
                mismatches.append(
                    f"feature {key} codec: dataset={actual_codec!r}, "
                    f"requested={requested_codec!r}"
                )

    actual_handumi = info.get("handumi")
    if not isinstance(actual_handumi, dict):
        mismatches.append("handumi metadata is missing")
        actual_handumi = {}
    if dataset_active_sides(actual_handumi) != dataset_active_sides(handumi):
        mismatches.append("handumi.active_sides: cannot change recorded sides when resuming")
    simple_keys = (
        "recording_device",
        "camera_fps",
        "tracking_loss_timeout_s",
        "tracking_schema",
        "tracking_workspace",
        "state_semantics",
        "capture_schema",
        "sync_lag_s",
        "max_sync_skew_s",
        "camera_stale_timeout_s",
        "gripper_stale_timeout_s",
        "sensor_loss_timeout_s",
        "feetech_sample_hz",
        "cameras",
        "sources",
        "audio",
    )
    for key in simple_keys:
        if actual_handumi.get(key) != handumi.get(key):
            mismatches.append(
                f"handumi.{key}: dataset={actual_handumi.get(key)!r}, "
                f"requested={handumi.get(key)!r}"
            )

    fingerprint_fields = {
        "controller_tcp_calibration": (
            "schema_version",
            "sha256",
            "applied_to_state",
            "source_robot",
            "source_gripper",
            "tracking_device",
            "controller_mount",
        ),
        "spatial_session_calibration": (
            "spatial_calibration_sha256",
            "tracking_device",
            "workspace_frame",
            "board_id",
        ),
        "target_robot": ("name", "sha256"),
    }
    for key, fields in fingerprint_fields.items():
        actual = _metadata_fingerprint(actual_handumi.get(key), fields)
        expected = _metadata_fingerprint(handumi.get(key), fields)
        if actual != expected:
            mismatches.append(f"handumi.{key}: dataset={actual!r}, requested={expected!r}")

    if mismatches:
        details = "\n  - ".join(mismatches)
        raise SystemExit(
            f"Cannot resume {root}: recording configuration is incompatible:\n  - {details}"
        )


def _record_session_history(
    metadata: dict[str, object],
    *,
    previous: dict[str, object] | None,
    first_episode: int,
    end_episode: int,
) -> None:
    """Keep immutable calibration snapshots for half-open episode ranges."""
    previous = previous or {}
    sessions = list(previous.get("recording_sessions") or [])
    original = previous.get("spatial_session_calibration")
    if not sessions and first_episode:
        sessions.append({
            "episode_from": 0,
            "episode_to": first_episode,
            "spatial_session_calibration": original,
        })
    if end_episode > first_episode:
        sessions.append({
            "episode_from": first_episode,
            "episode_to": end_episode,
            "spatial_session_calibration": metadata.get("spatial_session_calibration"),
        })
    metadata["recording_sessions"] = sessions
    # Retain the original snapshot for legacy consumers; per-run snapshots are
    # authoritative for session transforms. Raw rows already store table poses.
    if first_episode:
        metadata["spatial_session_calibration"] = original


def _update_info_json(
    root: Path, handumi: dict[str, object]
) -> dict[str, object] | None:
    path = root / "meta" / "info.json"
    if not path.exists():
        log.warning("Cannot write HandUMI metadata; missing %s", path)
        return None
    info = json.loads(path.read_text())
    info["handumi"] = {**info.get("handumi", {}), **handumi}
    # Image dimensions are authoritative in each feature and in the per-camera
    # snapshot.  The former global field could only describe one resolution and
    # was incorrect whenever a camera-specific rig entry overrode the fallback.
    info["handumi"].pop("camera_resolution", None)
    path.write_text(json.dumps(info, indent=4) + "\n")
    return info


def _dataset_card_kwargs(task: str) -> dict[str, str]:
    return {
        "dataset_description": (
            "Bimanual HandUMI demonstration data recorded in LeRobot v3 format. "
            f"Task: {task}"
        ),
        "url": "https://github.com/murobotics-ai/handumi-sw",
    }


def _write_dataset_readme(
    root: Path,
    *,
    repo_id: str,
    task: str,
    license_id: str,
) -> dict[str, str]:
    """Create the same LeRobot dataset card locally that Hub upload uses."""
    from lerobot.datasets.utils import create_lerobot_dataset_card

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    kwargs = _dataset_card_kwargs(task)
    card = create_lerobot_dataset_card(
        tags=["HandUMI"],
        dataset_info=info,
        license=license_id,
        repo_id=repo_id,
        **kwargs,
    )
    card.save(root / "README.md")
    return kwargs


def _validate_finalized_lerobot_dataset(root: Path) -> None:
    """Reject incomplete LeRobot v3 datasets before upload or reported success."""
    import pyarrow.parquet as pq

    info_path = root / "meta" / "info.json"
    readme_path = root / "README.md"
    if not info_path.is_file() or not readme_path.is_file():
        raise RuntimeError("Dataset is missing meta/info.json or README.md.")
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError(
            f"Expected LeRobot v3.0, got {info.get('codebase_version')!r}."
        )

    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    if total_episodes <= 0 or total_frames <= 0:
        raise RuntimeError("Dataset contains no completed episodes.")

    required_meta = (root / "meta" / "stats.json", root / "meta" / "tasks.parquet")
    if not all(path.is_file() for path in required_meta):
        raise RuntimeError("Dataset is missing stats.json or tasks.parquet.")

    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    data_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not episode_files or not data_files:
        raise RuntimeError("Dataset is missing episode metadata or data Parquet files.")

    episode_indices: set[int] = set()
    for path in episode_files:
        try:
            table = pq.read_table(path, columns=["episode_index"])
        except Exception as exc:
            raise RuntimeError(f"Invalid episode metadata Parquet: {path}.") from exc
        episode_indices.update(int(value.as_py()) for value in table["episode_index"])
    expected_indices = set(range(total_episodes))
    if episode_indices != expected_indices:
        raise RuntimeError(
            f"Episode metadata mismatch: expected {sorted(expected_indices)}, "
            f"found {sorted(episode_indices)}."
        )

    try:
        parquet_frames = sum(
            pq.ParquetFile(path).metadata.num_rows for path in data_files
        )
    except Exception as exc:
        raise RuntimeError("One or more data Parquet files are incomplete.") from exc
    if parquet_frames != total_frames:
        raise RuntimeError(
            f"Frame count mismatch: info.json={total_frames}, parquet={parquet_frames}."
        )

    video_keys = [
        key
        for key, feature in (info.get("features") or {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    missing_videos = [
        key for key in video_keys if not list((root / "videos" / key).glob("chunk-*/*.mp4"))
    ]
    if missing_videos:
        raise RuntimeError(f"Dataset is missing videos for: {', '.join(missing_videos)}.")

    handumi = info.get("handumi") or {}
    audio = handumi.get("audio") if isinstance(handumi, dict) else None
    if isinstance(audio, dict) and audio.get("enabled"):
        validate_audio_files(
            root,
            total_episodes,
            chunks_size=int(info.get("chunks_size", 1000)),
        )


def _normalize_camera_list(value: object) -> list[str]:
    if isinstance(value, str):
        names = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        names = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise SystemExit("Recording cameras must be a list or comma-separated string.")
    if not names:
        raise SystemExit("At least one recording camera is required.")
    if len(names) != len(set(names)):
        raise SystemExit("Recording camera names must not be repeated.")
    return names


def _camera_list_arg(value: str) -> list[str]:
    return _normalize_camera_list(value)


if __name__ == "__main__":
    main()
