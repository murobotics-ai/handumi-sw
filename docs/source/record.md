# Record Demonstrations

Recording uses HandUMI directly and does not require a robot arm. The raw
controller poses, camera streams, gripper widths, and calibration metadata
remain available for later retargeting to any supported embodiment.

## Before Recording

Confirm that:

- Each active gripper width responds correctly from closed to fully open.
- Each active controller reports valid tracking.
- Camera intrinsics and controller-camera mounts still match the hardware.
- The current session/table calibration was created for the same `--device`
  and has been visualized in Rerun.
- The Controller-to-TCP calibration matches the installed HandUMI tool.

See [Setup and Calibration](setup.md) if any check fails.

For a single Panda controlled with the right HandUMI, use `--robot franka_emika_panda`.
The recorder selects right tracking/encoder input automatically. See
[the Panda workflow](workflows/franka_emika_panda.md) for configuration and conversion.

Start with a short pilot:

```bash
handumi doctor --device meta
handumi record --output-dir outputs/handumi-demo \
  --task "pick and place" \
  --session-calibration outputs/calibration/session.yaml \
  --rerun \
  --episodes 3 \
  --episode-time-s 30
```

Each episode begins when you say "start recording" and ends on "stop
recording"; see [Controls](#controls) for the full set.

The usual device, cameras, resolution, FPS and target robot belong in the
optional `recording:` section of `configs/rig.yaml`. CLI values override those
defaults. Add `--dry-run` to resolve the complete plan, probe the encoder and
exit before opening any hardware.

Camera keys are user-defined and are preserved as dataset image names. Each
logical view chooses its own `opencv` or `zedmini` backend. Use `role` when a
name is not one of the recognized aliases (`left`/`left_wrist`,
`right`/`right_wrist`, and `top`/`workspace`):

```yaml
cameras:
  scene:
    role: workspace
    type: zedmini
    index_or_path: 4
    width: 1344
    height: 376
    fps: 30
recording:
  cameras: [left, right, scene]
```

The ZED Mini capture is stereo `1344×376`; only its left `672×376` half is
stored. Feature shapes and camera metadata record that output resolution.
Use `--cameras scene` to select that logical view; the command does not
need to know which backend it uses.

Use `--device pico` and a PICO-created `--session-calibration` for PICO. Add
`--push-to-hub` only after confirming the pilot locally.

Do not connect or configure a robot arm for this step. A target embodiment can
be selected later during conversion or replay without modifying the raw
recording.

## Resume a Recording

Append more episodes to a finalized local dataset without repeating its
recording configuration:

```bash
handumi record --output-dir outputs/handumi-demo --resume \
  --episodes 20 \
  --task "pick and place"
```

`--episodes` is the number of additional episodes, not the new total.
Resume requires an intact dataset from a previous graceful finalization and
loads the device, cameras, FPS, image format, tool calibration, Feetech state and
robot profile from its `meta/info.json` snapshot. Explicit incompatible
overrides are rejected before hardware starts, including FPS, cameras, image
shapes, tracking schemas, calibrations, or target-robot metadata. The task text
may change so the same dataset can contain multiple tasks.

The table/session calibration is read from the **current file** on every run,
including resume. Selection order is `--session-calibration`,
`recording.session_calibration` in the rig, then the last session's saved path.
A missing file is an error; resume never silently reuses an embedded transform.
Reusing the same file is valid while the VR origin is unchanged. After a new
tracking session, reset, or relocalization, run `calibrate spatial ... session`
and `verify` before resuming. A new session transform is accepted when its
spatial calibration, device, and board remain compatible.

`handumi.recording_sessions` in `meta/info.json` preserves calibration snapshots
with `episode_from` (inclusive) and `episode_to` (exclusive). The top-level
`spatial_session_calibration` remains the original snapshot for compatibility.
Each raw row also stores `workspace_from_device_pose` and original device poses;
`observation.state` already contains controller poses in the common table frame,
which replay/conversion uses without applying the session transform again.
Legacy datasets receive an initial history entry for their existing episodes;
this records their original calibration, not a correction of old measurements.

If the PICO device timestamp stops advancing for one second, recording aborts,
discards the active episode and finalizes previously saved episodes. A stationary
controller with advancing sample timestamps is valid. Restart tracking and
verify/recalibrate the session before recording again.

Image dimensions are stored in each image feature and in the per-camera
snapshot. This supports rigs whose cameras have different output resolutions;
the `recording.width` and `recording.height` values remain fallbacks only for
camera entries that do not declare their own dimensions.

Both `handumi record` and `handumi teleop-record` require `--output-dir`.
For example, `--output-dir outputs/handumi-demo` stores the dataset in that
directory, and `--resume` verifies and reads the same finalized dataset before
recording additional episodes.

`teleop-record` has a dedicated real-robot gesture protocol: double-squeeze
right while `READY` to start from home, right again while recording to save,
or left while recording to discard. Save and discard synchronously return the
robot home and leave the next episode waiting, so the operator can reset the
physical task. Double-squeeze both to discard and finish. See the
[real-robot episode gesture guide](physical_robots/real_teleoperation.md#episode-gestures)
for the complete behavior.

Unlike the earlier preview-only behavior, `teleop-record` stores every selected
camera as a LeRobot v3 `video` feature in the same dataset as the real-robot
joint observations and actions. Its dataset writer runs on a dedicated thread;
camera devices and MP4 encoders also have their own workers, so disk and codec
work does not execute in the robot-control loop. The terminal reports the
writer queue alongside the live control timing.

## Streaming Video Encoding

Video is encoded continuously while an episode is recorded. HandUMI probes the
local PyAV/FFmpeg encoders with a real MP4 before starting the tracking and
camera hardware, then reports the concrete selection, for example:

```text
Encoder: h264_nvenc (hardware, streaming, codec-managed threads).
```

The default `--encoder auto` tries a working hardware encoder first (NVIDIA
NVENC, Intel/AMD VAAPI or Quick Sync, or macOS VideoToolbox) and falls back to
H.264 on CPU. CPU encoding reserves one logical core and limits the threads
assigned to each camera so encoding does not starve capture.

Use `--encoder cpu` to force software encoding or `--encoder gpu` to require
hardware acceleration. `--vcodec <codec>` remains an advanced explicit
override; do not combine an explicit codec with `--encoder cpu` or
`--encoder gpu`.

Streaming writes frames directly to MP4 and calculates image statistics as
frames arrive instead of writing and rereading temporary PNG files. If an
encoder crashes, its queue overflows, a video is empty, or its frame count does
not match the episode, HandUMI discards the episode before appending its rows to
Parquet. `--encoder-threads` and `--encoder-queue-size` are advanced diagnostic
overrides; increasing the queue does not fix an encoder that is consistently
slower than capture.

`teleop-record` uses this same transactional streaming path automatically. If
its bounded background dataset queue fills, the active episode is discarded to
preserve exact alignment between video, state, and action instead of slowing
robot control.

## Controls

Episodes are driven by voice by default, so the collector never has to put the
shells down to reach a keyboard:

- "start recording": begin the episode.
- "stop recording": end and save the current episode.
- "restart": discard the current attempt and record it again.

Recognition is offline and its vocabulary is closed to exactly those three
phrases, so nothing else said in the room can trigger a command. The first
`handumi record` downloads a ~40 MB speech model to `~/.cache/handumi/vosk/`;
after that no network is needed. Audio comes from the system's default input,
which means plugging in a headset moves voice control to its microphone with
no flags to change.

Add `--clap-control` to *also* accept gripper squeezes. Both controls stay live
at once, which is what to use in a room too noisy to be heard reliably:

- Right double squeeze: start or save the current episode.
- Left double squeeze: discard and restart the current episode.

Always available:

- `Esc` or `Ctrl+C`: discard an active partial episode and stop.
- `--episode-time-s`: maximum episode length; it still applies as a safety
  limit while voice or clap control is running.

Pass `--no-voice-control` to record on the timer alone, or `--manual-control`
to use the PICO buttons (which turns voice off automatically). Since voice is
the default control, `handumi record` stops with an install hint if no
microphone or speech model is available — unless `--clap-control` gives it
another hands-free path, in which case it warns and continues. `handumi doctor`
reports the microphone and model state before a session.

The recorder waits for valid controllers and discards an episode after sustained tracking, camera, or encoder failure.

:::{dropdown} Tuning voice recognition
`--voice-device` selects a microphone by name or index when the system default
is not the one you want. `--voice-confidence` (default 0.7) raises or lowers
the bar a phrase must clear to count. The recorder deafens the microphone while
it speaks its own announcements, so "Stop recording" spoken by the machine is
never heard as the command.
:::

:::{dropdown} Synchronization and health gates
Every row uses one shared `observation.sync.target_time_ns`. Cameras, tracking,
and Feetech readings are selected from their native buffers against that
target. The default target is 40 ms behind real time (`--sync-lag-s 0.04`).

An episode is discarded after sustained controller loss
(`--tracking-loss-timeout-s`, default 1 second), or sustained camera/encoder
failure (`--sensor-loss-timeout-s`, default 1 second). Sources must also remain
inside `--max-sync-skew-s`.

Short failures remain visible in the raw dataset through timestamps and
`healthy` flags; they are not silently replaced. Use these options only when
diagnosing a known sensor-latency problem:

```bash
handumi record --help-advanced
```
:::

## Browser Dashboard

`--web-dashboard` serves a status page from the recording workstation. Use it
when the collector wears the headset and watches a laptop through passthrough,
or when the workstation has no microphone. The page shows:

- the recorder state, episode, frame count and saved episodes,
- live gripper widths and encoder ticks, with the double-squeeze thresholds,
- whether the Quest stream, headset and each controller are tracked,
- a preview of every camera with the age of its newest frame, outlined in red
  when a camera stops delivering frames.

The page keeps updating while the recorder waits for a trigger or for tracking,
because it reads the gripper, tracking and camera samplers directly.

The server listens on `localhost:8890` only. From a laptop, forward the port and
open the page there:

```bash
ssh -L 8890:localhost:8890 user@workstation
handumi record --output-dir outputs/my-dataset --task "..." --web-dashboard --no-sounds
# then open http://localhost:8890 on the laptop
```

Click **Enable microphone** once. The page streams the laptop microphone to the
workstation as 16 kHz audio, and the workstation runs the offline Vosk
recognizer on it, so the workstation needs neither a microphone nor PortAudio.
Say "start", "stop" or "discard" (the longer "start recording", "stop recording"
and "restart" also work). A command counts only when it ends the utterance:
"okay stop" stops, "stop for lunch" is ignored. The page also has buttons for
the same three commands, shows the last phrase it heard and why it was ignored,
and announces state changes through the laptop speaker. Browsers allow
microphone access only on `localhost` or HTTPS pages, which is why the page is
reached through the SSH tunnel.

With `--web-dashboard`, the terminal prints plain log lines instead of the
full-screen status panel, so tracking and camera warnings stay visible.

## Validate the Pilot

```bash
handumi validate \
  outputs/handumi-demo --strict
```

Review `meta/handumi_quality.json`. Fix rejected captures before increasing `--episodes`.

Hard rejection checks include insufficient duration, excessive tracking loss,
unhealthy cameras or encoders, synchronization errors, frozen source
timestamps or poses, implausible translation/rotation jumps, and invalid state
values. A stationary hand or constant gripper width is only a warning by
default. Thresholds live in `configs/quality.yaml`.

Common additions:

- `--pico-wifi`: stream PICO over Wi-Fi.
- `--skip-feetech`: record without gripper widths.
- `--dataset-license <id>`: set the dataset-card license.
- `--no-video`: store image frames instead of encoded video.
- `--encoder cpu`: force H.264 software encoding for reproducible CPU testing.
- `--encoder gpu`: require hardware encoding instead of falling back to CPU.

Run `handumi record --help` for the normal interface or
`handumi record --help-advanced` for synchronization, hardware and encoder
diagnostic overrides. Physical camera IDs belong only in `configs/rig.yaml`.
