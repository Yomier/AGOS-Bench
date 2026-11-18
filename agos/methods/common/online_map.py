from __future__ import annotations

import hashlib

import io

import math

import copy

import numpy as np

from PIL import Image, ImageDraw

from agos.perception.uav_grid_geometry import pixel_to_world, world_to_pixel

from agos.perception.public_road_elevation import PublicRoadElevation

class OnlineVisualMap:
    def __init__(self, region, resolution_m=0.5):
        self.region = {k: float(region[k]) for k in ("x_min", "x_max", "y_min", "y_max")}
        self.resolution_m = float(resolution_m)
        if not all(math.isfinite(v) for v in self.region.values()) or not (
                math.isfinite(self.resolution_m) and self.resolution_m > 0):
            raise ValueError("online-map bounds and resolution must be finite")
        self.width = math.ceil((self.region["y_max"]-self.region["y_min"])/self.resolution_m)
        self.height = math.ceil((self.region["x_max"]-self.region["x_min"])/self.resolution_m)
        if self.width <= 0 or self.height <= 0 or self.width*self.height > 16_000_000:
            raise ValueError("invalid or excessive online-map raster extent")
        self.reset()

    def reset(self):
        self._patches = []
        self._frames = set()
        self._ugv_history = []

    def _xy_pixel(self, xy):
        return ((float(xy[1])-self.region["y_min"])/self.resolution_m,
                (self.region["x_max"]-float(xy[0]))/self.resolution_m)

    def note_ugv(self, available_at, xy, trajectory):
        """Only the UGV's executed GPS path; no UAV or candidate map markers."""
        event = (float(available_at), [float(xy[0]), float(xy[1])],
                 [[float(p[0]), float(p[1])] for p in trajectory])
        if not self._ugv_history or self._ugv_history[-1] != event:
            self._ugv_history.append(event)

    def update(self, frame, available_at):
        try:
            values = [float(getattr(frame, k)) for k in (
                "x", "y", "z", "yaw_deg", "pitch_deg", "fov_deg", "timestamp_s")]
            if (not frame.valid or frame.frame_id is None or not frame.image_bytes
                    or not all(math.isfinite(v) for v in values)
                    or not math.isfinite(float(available_at)) or frame.z <= 0
                    or abs(frame.yaw_deg % 360) > 1e-6 or abs(frame.pitch_deg+90) > 1e-6
                    or frame.fov_deg != 90 or (frame.width, frame.height) != (1024, 576)):
                raise ValueError("missing, invalid, or unsupported camera metadata")
            if frame.frame_id in self._frames:
                return {"status": "duplicate_frame", "frame_id": frame.frame_id}
            source = np.asarray(Image.open(io.BytesIO(frame.image_bytes)).convert("RGB"))
            if source.shape != (frame.height, frame.width, 3):
                raise ValueError("RGB dimensions do not match bound camera metadata")
            args = (frame.x, frame.y, frame.z, frame.yaw_deg, frame.fov_deg, frame.width, frame.height)
            corners = [pixel_to_world(*args, px, py) for px, py in (
                (0, 0), (frame.width, 0), (frame.width, frame.height), (0, frame.height))]
            map_corners = [self._xy_pixel(p) for p in corners]
            c0 = max(0, math.floor(min(p[0] for p in map_corners)))
            c1 = min(self.width, math.ceil(max(p[0] for p in map_corners)))
            r0 = max(0, math.floor(min(p[1] for p in map_corners)))
            r1 = min(self.height, math.ceil(max(p[1] for p in map_corners)))
            if c0 >= c1 or r0 >= r1:
                return {"status": "outside_region", "frame_id": frame.frame_id}
            # Derive the inverse affine scale from the shared geometry module.
            cp = world_to_pixel(*args, frame.x, frame.y)
            ep = world_to_pixel(*args, frame.x, frame.y+1.0)
            np_ = world_to_pixel(*args, frame.x+1.0, frame.y)
            wy = self.region["y_min"]+(np.arange(c0, c1)+0.5)*self.resolution_m
            wx = self.region["x_max"]-(np.arange(r0, r1)+0.5)*self.resolution_m
            sx = cp[0]+(wy-frame.y)*(ep[0]-cp[0])
            sy = cp[1]+(wx-frame.x)*(np_[1]-cp[1])
            valid = ((sy[:, None] >= 0) & (sy[:, None] < frame.height)
                     & (sx[None, :] >= 0) & (sx[None, :] < frame.width))
            pixels = source[np.clip(sy.astype(int), 0, frame.height-1)[:, None],
                            np.clip(sx.astype(int), 0, frame.width-1)[None, :]].copy()
            info = {"frame_id": int(frame.frame_id), "sensor_timestamp_s": float(frame.timestamp_s),
                    "available_at": float(available_at), "capture_pose": {
                        "x": float(frame.x), "y": float(frame.y), "z": float(frame.z),
                        "yaw_deg": float(frame.yaw_deg), "pitch_deg": float(frame.pitch_deg)},
                    "raw_rgb_sha256": hashlib.sha256(frame.image_bytes).hexdigest()}
            self._patches.append((float(available_at), c0, c1, r0, r1, pixels, valid, info))
            self._frames.add(frame.frame_id)
            return {"status": "updated", **info}
        except (AttributeError, TypeError, ValueError, OSError) as exc:
            return {"status": "invalid_frame", "error": str(exc)}

    def render(self, as_of, overlays=False):
        canvas = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        observed = np.zeros((self.height, self.width), dtype=bool)
        visible = sorted((p for p in self._patches if p[0] <= float(as_of)), key=lambda p: p[0])
        for _, c0, c1, r0, r1, pixels, valid, _ in visible:
            canvas[r0:r1, c0:c1][valid] = pixels[valid]
            observed[r0:r1, c0:c1] |= valid
        count = int(observed.sum())
        meta = {"status": "ready" if visible else "empty", "source": "raw_uav_rgb",
                "as_of_event_time": float(as_of), "update_count": len(visible),
                "source_frames": [p[-1]["frame_id"] for p in visible],
                "source_packets": [p[-1] for p in visible], "observed_pixels": count,
                "coverage_fraction": count/(self.width*self.height),
                "unknown_fraction": 1-count/(self.width*self.height),
                "resolution_m": self.resolution_m,
                "origin": {"x_max": self.region["x_max"], "y_min": self.region["y_min"]},
                "width": self.width, "height": self.height,
                "projection": "north_up_ground_plane_approximation", "overlays": []}
        image = Image.fromarray(canvas)
        ugv_events = sorted((e for e in self._ugv_history if e[0] <= float(as_of)), key=lambda e: e[0])
        if overlays and ugv_events:
            _, xy, trajectory = ugv_events[-1]
            draw = ImageDraw.Draw(image)
            if len(trajectory) > 1:
                draw.line([self._xy_pixel(p) for p in trajectory], fill=(255, 210, 0), width=2)
            px, py = self._xy_pixel(xy)
            draw.ellipse((px-4, py-4, px+4, py+4), fill=(255, 210, 0), outline=(20, 20, 20))
            meta["overlays"] = ["ugv_position", "ugv_trajectory"]
            meta["ugv_position"] = xy
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue(), meta
