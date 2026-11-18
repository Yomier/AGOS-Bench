"""Nadir camera orientation, calibration and grid/world round trips."""
from agos.perception import uav_grid_geometry as g


def test_north_south_direction():
    """图像上半=北（world_x > uav_x），下半=南"""
    north = g.pixel_to_world(0, 0, 80, 0, 90, 1024, 576, 512, 100)
    south = g.pixel_to_world(0, 0, 80, 0, 90, 1024, 576, 512, 476)
    assert north[0] > 20, "图像上半应映射到北侧, got %s" % (north,)
    assert south[0] < -20, "图像下半应映射到南侧, got %s" % (south,)
    assert abs(north[1]) < 1e-6 and abs(south[1]) < 1e-6


def test_east_west_direction():
    """图像右半=东（world_y > uav_y），左半=西"""
    east = g.pixel_to_world(0, 0, 80, 0, 90, 1024, 576, 900, 288)
    west = g.pixel_to_world(0, 0, 80, 0, 90, 1024, 576, 124, 288)
    assert east[1] > 50, "图像右侧应映射到东侧, got %s" % (east,)
    assert west[1] < -50, "图像左侧应映射到西侧, got %s" % (west,)


def test_magnitude():
    """80m 高 / FOV90 → 地面足迹 160m x 90m；边缘像素偏移量级正确"""
    edge = g.pixel_to_world(0, 0, 80, 0, 90, 1024, 576, 1024, 288)
    assert 79 < edge[1] < 81, "右边缘应约 +80m, got %s" % (edge,)


def test_roundtrip_pixel_world():
    for px, py in [(100, 100), (900, 480), (512, 288), (300, 200)]:
        wx, wy = g.pixel_to_world(37.5, -88.25, 80, 0, 90, 1024, 576, px, py)
        rpx, rpy = g.world_to_pixel(37.5, -88.25, 80, 0, 90, 1024, 576, wx, wy)
        assert abs(rpx - px) < 0.5 and abs(rpy - py) < 0.5, (px, py, rpx, rpy)


def test_center_cell_centered_on_uav():
    poly, center, unc = g.cell_label_to_world_polygon(50, 60, 80, 0, 90, 1024, 576, "C3")
    assert abs(center[0] - 50) < 0.01 and abs(center[1] - 60) < 0.01
    assert 20 <= unc <= 45


def test_world_to_grid_cell_rows():
    """目标在北侧 → A/B 行；南侧 → D/E 行"""
    cell, _, _ = g.world_to_grid_cell(0, 0, 80, 0, 90, 1024, 576, 36.0, 0.0)
    assert cell[0] in "AB", "北侧目标应落 A/B 行, got %s" % cell
    cell, _, _ = g.world_to_grid_cell(0, 0, 80, 0, 90, 1024, 576, -36.0, 0.0)
    assert cell[0] in "DE", "南侧目标应落 D/E 行, got %s" % cell
    cell, _, _ = g.world_to_grid_cell(0, 0, 80, 0, 90, 1024, 576, 0.0, 55.0)
    assert cell[1] in "45", "东侧目标应落 4/5 列, got %s" % cell


def test_calibration_guard_rejects_mirrored_matrix():
    """07-20 式镜像矩阵必须被 _validate_calibration 拒绝"""
    mirrored = {
        "height_m": 80.0, "fov_deg": 90.0, "img_w": 1024, "img_h": 576,
        "normalized_offset_to_world_xy": [[0.0, 90.0], [160.0, 0.0]],
    }
    assert g._validate_calibration(mirrored) is False


def test_shipped_calibration_is_valid_and_north_up(monkeypatch):
    monkeypatch.delenv('AGOS_ASSET_ROOT', raising=False)
    calibration = g.load_calibration()
    assert g._validate_calibration(calibration)
    assert calibration['normalized_offset_to_world_xy'][0][1] < 0


def test_perception_delegates():
    from agos.perception.perception import project_cell_to_world, compute_grid_cell_from_world
    cc = g.cell_label_to_pixel_polygon("C3")
    poly, center, unc = project_cell_to_world(10, 20, 80, 0, -90, 90, 1024, 576, cc)
    assert abs(center[0] - 10) < 0.01 and abs(center[1] - 20) < 0.01
    cell, _, _ = compute_grid_cell_from_world(0, 0, 80, 0, 90, 1024, 576, 36.0, 0.0)
    assert cell[0] in "AB"
