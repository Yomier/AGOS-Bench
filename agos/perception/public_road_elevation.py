"""Read-only PUBLIC navigation heights, using the shared map's 5m lattice."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PublicRoadElevation:
    x_max: float
    y_min: float
    resolution_m: float
    height: int
    width: int
    stride: int
    elevations: tuple

    @classmethod
    def from_provider(cls, region, resolution_m, provider):
        resolution = float(resolution_m)
        height = math.ceil((region['x_max'] - region['x_min']) / resolution)
        width = math.ceil((region['y_max'] - region['y_min']) / resolution)
        stride = max(1, round(5 / resolution))
        rows = []
        for r in range(0, height, stride):
            row = []
            for c in range(0, width, stride):
                x = region['x_max'] - (r + .5) * resolution
                y = region['y_min'] + (c + .5) * resolution
                try:
                    value = provider(x, y) if provider is not None else None
                    value = float(value) if value is not None else None
                    if value is not None and not math.isfinite(value):
                        value = None
                except Exception:
                    value = None
                row.append(value)
            rows.append(tuple(row))
        return cls(float(region['x_max']), float(region['y_min']), resolution,
                   height, width, stride, tuple(rows))

    def __call__(self, x, y):
        row = math.floor((self.x_max - float(x)) / self.resolution_m)
        col = math.floor((float(y) - self.y_min) / self.resolution_m)
        if not (0 <= row < self.height and 0 <= col < self.width):
            return None
        return self.elevations[row // self.stride][col // self.stride]

    def raster(self):
        import numpy as np
        coarse = np.array([[float('nan') if z is None else z for z in row]
                           for row in self.elevations], dtype=float)
        values = coarse.repeat(self.stride, axis=0).repeat(self.stride, axis=1)
        values = values[:self.height, :self.width]
        known = np.isfinite(values)
        return np.where(known, values, 0.), known
