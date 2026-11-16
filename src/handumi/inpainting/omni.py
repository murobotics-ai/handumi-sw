"""Gemini Omni Flash video editing for the context camera.

One call edits one clip. The bytes it returns are written to disk before
anything else touches them: the call is already paid for by the time they
arrive, so losing them to a crash or a bad resample means paying twice.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from google.genai.interactions import Interaction

MODEL = "gemini-omni-1.2-flash"
API_KEY_ENV = "GEMINI_API_KEY"

# Video output dominates the bill (output tokens cost ~12x input ones), and it is
# priced per second by resolution. The context camera is 672x376, so anything
# above 360p is paid for and then thrown away by the resample back onto the
# dataset grid.
DEFAULT_RESOLUTION = "360p"


@dataclass(frozen=True)
class EditResult:
    interaction_id: str
    status: str
    latency_s: float
    raw_video: Path


def _wait_active(client, file):
    while file.state.name == "PROCESSING":
        time.sleep(5)
        file = client.files.get(name=file.name)
    if file.state.name == "FAILED":
        raise RuntimeError(f"upload failed: {file.name}")
    return file


def edit_clip(
    clip: Path,
    references: list[Path],
    prompt: str,
    raw_output: Path,
    *,
    model: str = MODEL,
    resolution: str = DEFAULT_RESOLUTION,
    api_key: str | None = None,
) -> EditResult:
    """Edit ``clip`` with ``prompt`` and save the returned video to ``raw_output``."""
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "google-genai is required for context inpainting. Install with: uv sync --extra inpaint"
        ) from exc

    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} is not set (put it in .env).")

    client = genai.Client(api_key=key)
    started = time.time()

    uploads = [_wait_active(client, client.files.upload(file=str(p))) for p in [clip, *references]]
    response = client.interactions.create(
        model=model,
        input=[
            *({"type": "document", "uri": upload.uri} for upload in uploads),
            {"type": "text", "text": prompt},
        ],
        response_format={"type": "video", "delivery": "uri", "resolution": resolution},
        # Keep this literal explicit.  The SDK overload returns a Stream when
        # stream=True and an Interaction (which owns output_video) otherwise.
        stream=False,
    )
    # google-genai 2.20's generated fallback overload still includes Stream
    # when ``model`` is a configurable string.  stream=False guarantees the
    # non-streaming response at runtime, so narrow that SDK typing artifact.
    interaction = cast("Interaction", response)
    latency = time.time() - started

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    video = interaction.output_video
    if video is None:
        raise RuntimeError(
            f"interaction returned no video (status={interaction.status}, "
            f"errors={interaction.errors})"
        )

    data = video.data
    if data:
        raw_output.write_bytes(base64.b64decode(data))
    else:
        uri = video.uri
        if not uri:
            raise RuntimeError("interaction video contains neither inline data nor a URI")

        name = "files/" + uri.split("/files/")[-1].split(":")[0]
        while True:
            info = client.files.get(name=name)
            state = info.state
            if state is None:
                raise RuntimeError(f"generated file {name} has no state")
            if state.name == "ACTIVE":
                break
            if state.name == "FAILED":
                raise RuntimeError("generation failed")
            time.sleep(5)
        raw_output.write_bytes(client.files.download(file=uri))

    interaction_id = interaction.id
    if not interaction_id:
        raise RuntimeError("interaction response has no id")

    return EditResult(
        interaction_id=interaction_id,
        status=str(interaction.status),
        latency_s=round(latency, 1),
        raw_video=raw_output,
    )
