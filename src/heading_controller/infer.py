"""
Inference entry-point for the vision-based heading controller.

This thin wrapper exists so that external callers (e.g. the integration
pipeline in src/pipeline/run_pipeline.py) can import a single function
without having to deal with argparse.

For CLI usage run  main.py  directly:

    python src/heading_controller/main.py \\
        --video  path/to/video.mp4 \\
        --masks  path/to/masks/ \\
        --output results/controller_output.csv

For programmatic usage call  run_inference()  below.
"""

import sys
from pathlib import Path

# Ensure sibling modules are importable
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from heading_controller import DEFAULT_CONFIG
from main import run_pipeline


def run_inference(
    video_path: str,
    masks_dir: str | None = None,
    output_csv: str = "results/controller_output.csv",
    visualise: bool = False,
    flow_every: int = 1,
    config: dict | None = None,
) -> None:
    """
    Run the full heading-controller pipeline on a video file.

    Parameters
    ----------
    video_path  : path to the .mp4 input video
    masks_dir   : directory with pre-computed .npy masks, or None for a
                  dummy all-road mask (useful for unit-testing)
    output_csv  : where to write the per-frame CSV log
    visualise   : show the OpenCV preview window
    flow_every  : compute optical flow every N frames (1 = every frame)
    config      : override any key from heading_controller.DEFAULT_CONFIG
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    run_pipeline(
        video_path=video_path,
        masks_dir=masks_dir,
        output_csv=output_csv,
        visualise=visualise,
        flow_every=flow_every,
        config=cfg,
    )


if __name__ == "__main__":
    # Minimal smoke-test: pass a video path as the first positional argument.
    if len(sys.argv) < 2:
        print("Usage: python infer.py <video.mp4> [masks_dir] [output.csv]")
        sys.exit(1)

    _video = sys.argv[1]
    _masks = sys.argv[2] if len(sys.argv) > 2 else None
    _out   = sys.argv[3] if len(sys.argv) > 3 else "results/controller_output.csv"

    run_inference(
        video_path=_video,
        masks_dir=_masks,
        output_csv=_out,
        visualise=True,
    )
