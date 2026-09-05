"""RM-351: build the 720p clip the provider benchmark is measured on.

The committed provider evidence ran on a 160x96 sixteen-frame fixture driven
with a fixed subtitle region, so the detector never ran and the inpainter had
almost nothing to do. On that clip the CUDA lane measures slightly SLOWER
than the CPU one, because what it measures is numpy bookkeeping rather than
inference. A user deciding which build to download learns nothing from it.

This renders a 1280x720 clip long enough for the inference paths to dominate:
a moving textured background so the inpainter has real work to reconstruct,
and rendered subtitles that change through the clip so automatic detection
has real work to find. It is deterministic, so the sha256 recorded beside the
evidence is reproducible from this script rather than trusted.

It is not a substitute for real footage. Acquiring redistributable real-world
clips is RM-342's job, and the benchmark evidence says so.

    python scripts/generate_benchmark_clip.py

"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np

from backend.safe_image import safe_imwrite

ROOT = Path(__file__).resolve().parent.parent
# Not tests/clips/: that directory is the reference corpus, and every file
# in it must carry metric floors in manifest.json. A benchmark input has
# none. The clip is committed rather than regenerated on demand because
# x264 output is not bit-reproducible across FFmpeg builds, and the
# sha256 recorded beside the evidence has to stay checkable.
DEFAULT_OUTPUT = ROOT / "tests" / "benchmarks" / "benchmark_720p.mkv"

WIDTH = 1280
HEIGHT = 720
FPS = 24
SECONDS = 12
FRAME_COUNT = FPS * SECONDS

# Four captions, three seconds each. Ordinary sentences at ordinary lengths,
# because a detector's cost tracks how much text is on screen.
CAPTIONS = (
    "The tide came in faster than the map said it would.",
    "Nobody had thought to bring a second lamp.",
    "We waited for the wind to drop, and it did not.",
    "By morning the boat was gone and the rope was cut.",
)


def _background(index: int) -> np.ndarray:
    """A moving textured plate. Deterministic in `index` alone."""
    rows = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 1.0, WIDTH, dtype=np.float32)[None, :]
    drift = index / float(FRAME_COUNT)

    # Two travelling wave fields plus a slow vertical ramp. Structured
    # enough that a naive fill is visibly wrong, smooth enough that the
    # temporal path has something coherent to reconstruct from.
    first = np.sin((cols * 9.0) + (drift * 6.283)) * 0.5 + 0.5
    second = np.cos((rows * 7.0) - (drift * 4.712)) * 0.5 + 0.5
    ramp = rows * 0.6 + 0.2

    # A zero plate so every channel broadcasts to the full frame: `second`
    # and `ramp` are both column vectors, and adding them alone yields a
    # column rather than an image.
    plate = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    blue = (plate + first * 0.45 + ramp * 0.55) * 255.0
    green = (plate + second * 0.50 + ramp * 0.50) * 255.0
    red = (plate + (first * second) * 0.65 + ramp * 0.35) * 255.0
    frame = np.dstack((blue, green, red)).astype(np.uint8)

    # A hard edge that sweeps across, so the inpainter cannot get away with
    # blurring: the background behind the caption genuinely changes.
    edge = int((0.15 + 0.7 * drift) * WIDTH)
    frame[:, edge:edge + 6] = np.clip(
        frame[:, edge:edge + 6].astype(np.int16) + 60, 0, 255).astype(np.uint8)
    return frame


def _caption_for(index: int) -> str:
    return CAPTIONS[min(index // (FPS * 3), len(CAPTIONS) - 1)]


def _draw_caption(frame: np.ndarray, text: str) -> None:
    """Render the caption the way a burned-in subtitle actually looks."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.1
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, scale, thickness)
    origin = ((WIDTH - text_w) // 2, HEIGHT - 70)

    # A dark outline under white fill, which is what almost every burned-in
    # subtitle does and what the detector is tuned for.
    cv2.putText(frame, text, origin, font, scale, (0, 0, 0),
                thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, font, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)
    del text_h, baseline


def _write_frames(directory: Path) -> None:
    for index in range(FRAME_COUNT):
        frame = _background(index)
        _draw_caption(frame, _caption_for(index))
        safe_imwrite(directory / f"{index:05d}.png", frame)


def _encode(frames_dir: Path, output: Path) -> None:
    """Encode small enough to commit, and lossy on purpose.

    FFV1 gives a byte-reproducible clip and a 37 MB file, which is no use in
    a repository. x264 at CRF 18 lands around a megabyte, and the compression
    artifacts around the caption edges are what a real subtitled source
    carries anyway, so the detector sees a more honest input than a lossless
    render would give it.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-framerate", str(FPS),
        "-i", str(frames_dir / "%05d.png"),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        # Tag the colour signalling explicitly. Left unspecified, the output
        # contract check has nothing on the source to compare the output's
        # tags against and reports the range as not preserved, which fails
        # the run before any timing is recorded.
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-color_range", "tv",
        str(output),
    ]
    # subprocess-policy-exempt: this is a developer script that builds a test
    # fixture, run by hand from a checkout. The argument list is fixed here,
    # nothing in it comes from user input, and no shell is involved.
    subprocess.run(command, check=True, timeout=900)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--check", action="store_true",
        help="report the existing clip's digest without rebuilding it")
    args = parser.parse_args(argv)
    output = Path(args.output)

    if args.check:
        if not output.is_file():
            print(f"missing: {output}")
            return 1
        print(f"{_sha256(output)}  {output.name}")
        return 0

    with tempfile.TemporaryDirectory(prefix="vsr-benchmark-clip-") as tmp:
        _write_frames(Path(tmp))
        _encode(Path(tmp), output)

    print(f"{output} ({output.stat().st_size} bytes)")
    print(f"sha256 {_sha256(output)}")
    print(f"{WIDTH}x{HEIGHT} {FPS}fps {SECONDS}s ({FRAME_COUNT} frames)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
