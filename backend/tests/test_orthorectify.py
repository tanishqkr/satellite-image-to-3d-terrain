"""
Unit and Integration Tests for Oblique & Off-Nadir Rectification (§2.2).
Covers:
1. Camera geometry & orthonormal basis calculation (pitch, roll, yaw).
2. Pass A: Pre-inference de-obliquing using DTM proxy surface.
3. Pass B: Post-inference true-orthorectification with Z-buffer occlusion handling.
4. XMP/EXIF drone metadata parsing for camera pose.
5. End-to-end API pipeline upload with oblique camera parameters.
"""
import sys
import io
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.orthorectify import (
    CameraPose,
    get_camera_basis,
    project_ground_to_camera,
    pre_rectify,
    post_rectify,
)
from app.services.geospatial import parse_camera_pose, read_image


def test_camera_basis_and_orthonormality():
    print("Testing camera pose & orthonormal basis calculation...")
    # 1. Oblique check threshold
    pose_nadir = CameraPose(pitch_deg=10.0)
    assert not pose_nadir.is_oblique, "Pitch <= 15 deg should not be marked oblique"

    pose_oblique = CameraPose(pitch_deg=35.0, roll_deg=5.0)
    assert pose_oblique.is_oblique, "Pitch 35 deg must be marked oblique"

    # 2. Orthonormal basis test
    C, u_c, v_c, w_c = get_camera_basis(pitch_deg=45.0, roll_deg=10.0, yaw_deg=60.0, altitude_m=120.0)
    assert np.isclose(np.linalg.norm(u_c), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(v_c), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(w_c), 1.0, atol=1e-6)
    assert np.isclose(np.dot(u_c, v_c), 0.0, atol=1e-6)
    assert np.isclose(np.dot(u_c, w_c), 0.0, atol=1e-6)
    assert np.isclose(np.dot(v_c, w_c), 0.0, atol=1e-6)
    print("  [PASS] Camera basis strictly orthonormal: PASSED")


def test_pass_a_pre_rectify():
    print("Testing Pass A: pre-inference de-obliquing...")
    H, W = 128, 128
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # Bright target in image center
    img[60:68, 60:68] = 255

    pose = CameraPose(pitch_deg=35.0, altitude_m=100.0, fov_deg=60.0)
    coarse_dtm = np.zeros((H, W), dtype=np.float32)

    deoblique = pre_rectify(img, coarse_dtm, pose, gsd=0.5)
    assert deoblique.shape == (H, W, 3)
    assert deoblique.dtype == np.uint8
    # Center region should retain brightness
    assert deoblique[60:68, 60:68].max() > 0

    # Non-oblique image should pass through unchanged
    pose_flat = CameraPose(pitch_deg=0.0)
    unchanged = pre_rectify(img, coarse_dtm, pose_flat, gsd=0.5)
    assert np.array_equal(unchanged, img)
    print("  [PASS] Pass A pre-rectify geometry and passthrough: PASSED")


def test_pass_b_zbuffering_occlusion():
    print("Testing Pass B: post-inference true-orthorectification & Z-buffering...")
    H, W = 128, 128
    img = np.full((H, W, 3), 120, dtype=np.uint8)

    # 30m tall building in center
    fine_dsm = np.zeros((H, W), dtype=np.float32)
    fine_dsm[50:78, 50:78] = 30.0

    pose = CameraPose(pitch_deg=45.0, altitude_m=100.0, fov_deg=60.0)
    ortho_rgb, occluded = post_rectify(img, fine_dsm, pose, gsd=0.5)

    assert ortho_rgb.shape == (H, W, 3)
    assert occluded.dtype == bool
    assert occluded.shape == (H, W)

    # Occlusion must be detected in the shadow/rear of the 30m structure
    occluded_count = int(occluded.sum())
    assert occluded_count > 100, f"Expected occlusion behind 30m building, got {occluded_count} pixels"
    print(f"  [PASS] Pass B Z-buffering detected {occluded_count} occluded pixels behind structure: PASSED")


def test_drone_xmp_metadata_parsing():
    print("Testing drone XMP metadata parsing...")
    xmp_sample = (
        b'prefix...<x:xmpmeta><rdf:RDF><rdf:Description '
        b'drone-dji:GimbalPitchDegree="-45.00" '
        b'drone-dji:GimbalRollDegree="+2.50" '
        b'drone-dji:GimbalYawDegree="+135.00" '
        b'drone-dji:RelativeAltitude="+125.00"/></rdf:RDF></x:xmpmeta>...suffix'
    )
    pose = parse_camera_pose(xmp_sample)
    assert pose is not None
    assert np.isclose(pose.pitch_deg, 45.0), f"Expected 45.0, got {pose.pitch_deg}"
    assert np.isclose(pose.roll_deg, 2.5), f"Expected 2.5, got {pose.roll_deg}"
    assert np.isclose(pose.yaw_deg, 135.0), f"Expected 135.0, got {pose.yaw_deg}"
    assert np.isclose(pose.altitude_m, 125.0), f"Expected 125.0, got {pose.altitude_m}"
    assert pose.is_oblique, "45 degree pitch must be marked oblique"
    print("  [PASS] Drone XMP telemetry parsed correctly: PASSED")


def test_api_upload_with_oblique_pose():
    print("Testing API /upload with manual oblique camera pose parameters...")
    from fastapi.testclient import TestClient
    from app.main import app

    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    with TestClient(app) as client:
        resp = client.post(
            "/api/upload?camera_pitch=35.0&camera_roll=2.0&camera_altitude=80.0",
            files={"file": ("oblique_drone.png", buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 200, f"Upload failed: {resp.text}"
        data = resp.json()
        assert "rectification" in data
        assert data["rectification"] is not None
        assert data["rectification"]["applied"] is True
        assert np.isclose(data["rectification"]["pitch_deg"], 35.0)
        print("  [PASS] API /upload with oblique rectification completed successfully: PASSED")


if __name__ == "__main__":
    test_camera_basis_and_orthonormality()
    test_pass_a_pre_rectify()
    test_pass_b_zbuffering_occlusion()
    test_drone_xmp_metadata_parsing()
    test_api_upload_with_oblique_pose()
    print("\nALL ORTHORECTIFICATION (§2.2) TESTS PASSED!")
