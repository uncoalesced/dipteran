"""Render the map marker's fly sprite from the flybody MuJoCo body model.

The HUD's vehicle marker was a plain SVG dot. This renders a real *Drosophila*
instead, dorsally, from TuragaLab/flybody's anatomical MuJoCo model (Apache-2.0
-- see DATA_LICENSE.md for the attribution the derived asset carries).

Three things make this a sprite renderer rather than a simulation:

  * No physics is stepped. `mj_forward` once on the model's rest pose is all a
    map marker wants -- a standing fly with its wings folded, not a snapshot
    of a fly mid-stumble.
  * The camera looks straight down the body's dorsal axis with an orthographic
    projection, so the silhouette is a true top-down outline and does not
    change shape as the marker rotates.
  * The output is cropped to the body's own alpha bounding box and squared
    about its centre, so CSS `rotate()` pivots on the fly rather than on dead
    margin.

The render is head-up (the fly's anterior points toward the top of the image).
`index.html` rotates the marker by the telemetry `heading` field, which is
degrees clockwise from north, so head-up is exactly what that binding expects
and no offset is needed.

    MUJOCO_GL=egl python scripts/render_fly_sprite.py

EGL because this runs headless under WSL2: MuJoCo's default GLFW backend wants
an X display and there is none.

Engineered by uncoalesced.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must be set before mujoco is imported, not after -- the rendering backend is
# chosen at import time and a later assignment is silently ignored.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_MODEL = Path("/opt/dipteran/flybody/flybody/fruitfly/assets/fruitfly.xml")
DEFAULT_OUT = REPO / "assets" / "sprites" / "fly-dorsal.png"

# Rendered large and downsampled, which is cheaper than fighting MuJoCo's
# multisampling settings and gives clean edges at map-marker size.
RENDER_PX = 1024
OUTPUT_PX = 256


def render(model_path: Path, size: int = RENDER_PX) -> np.ndarray:
    """RGBA array of the model's rest pose, seen from directly above."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)          # rest pose; no stepping

    model.vis.global_.offwidth = size
    model.vis.global_.offheight = size
    # Orthographic: a plan view with no perspective foreshortening of the far
    # wing, so the silhouette stays constant as the marker rotates.
    model.vis.global_.orthographic = 1

    # In orthographic mode MuJoCo reads `fovy` as the view's half-height in
    # *model units*, not as an angle in degrees, so the 45 it defaults to is
    # meaningless here. The model spans 0.29 x 0.38 x 0.15, and geom_xpos
    # bounds geom centres rather than mesh extents, so the framing needs
    # generous margin -- `crop_to_content` tightens it afterwards anyway.
    positions = data.geom_xpos
    low, high = positions.min(axis=0), positions.max(axis=0)
    model.vis.global_.fovy = float((high - low).max()) * 1.25

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # elevation -90 looks straight down the dorsal (Z) axis. azimuth 0 is what
    # puts the head at the top of the frame -- checked by rendering all four
    # quadrants, not assumed: 90 and 270 lay the fly sideways and 180 is
    # head-down. Head-up is what `index.html`'s rotate(heading) binding wants,
    # heading being degrees clockwise from north.
    camera.elevation = -90.0
    camera.azimuth = 0.0
    camera.lookat[:] = (low + high) / 2.0
    camera.distance = 1.0          # ignored under orthographic; fovy frames it

    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    # Joint axes and actuators are simulation scaffolding, not the fly.
    option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False

    with mujoco.Renderer(model, height=size, width=size) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        rgb = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera, scene_option=option)
        seg = renderer.render().copy()

    # Segmentation gives an exact alpha mask: every pixel belonging to a geom
    # is fly, everything else is background. Chroma-keying the background
    # colour instead would also punch holes in any part of the body that
    # happened to match it.
    alpha = np.where(seg[:, :, 0] >= 0, 255, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


def crop_to_content(rgba: np.ndarray, pad: int = 2) -> np.ndarray:
    """Trim to the alpha bounding box, then square it about the centre."""
    rows = np.any(rgba[:, :, 3] > 0, axis=1)
    cols = np.any(rgba[:, :, 3] > 0, axis=0)
    if not rows.any() or not cols.any():
        raise RuntimeError(
            "render is entirely transparent -- the camera is not pointed at "
            "the model, or the segmentation pass returned no geoms"
        )
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1 = min(rgba.shape[0] - 1, y1 + pad)
    x1 = min(rgba.shape[1] - 1, x1 + pad)

    # CSS rotates about the image centre, so a non-square crop would make the
    # marker wobble as it turns.
    side = max(y1 - y0 + 1, x1 - x0 + 1)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    top, left = cy - side // 2, cx - side // 2
    out = np.zeros((side, side, 4), dtype=np.uint8)
    src_top, src_left = max(0, top), max(0, left)
    src_bottom = min(rgba.shape[0], top + side)
    src_right = min(rgba.shape[1], left + side)
    out[src_top - top:src_bottom - top,
        src_left - left:src_right - left] = rgba[src_top:src_bottom,
                                                 src_left:src_right]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--size", type=int, default=OUTPUT_PX)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"{args.model} not found. Sparse-clone the model first:\n"
            "  git clone --depth 1 --filter=blob:none --sparse "
            "https://github.com/TuragaLab/flybody.git\n"
            "  cd flybody && git sparse-checkout set flybody/fruitfly/assets"
        )

    from PIL import Image

    rgba = crop_to_content(render(args.model))
    image = Image.fromarray(rgba, mode="RGBA").resize(
        (args.size, args.size), Image.LANCZOS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    opaque = int((np.asarray(image)[:, :, 3] > 0).sum())
    print(f"[sprite] wrote {args.out} ({args.size}x{args.size}, "
          f"{opaque:,} opaque px, {args.out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
