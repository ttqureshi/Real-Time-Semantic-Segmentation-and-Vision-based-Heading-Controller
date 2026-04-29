"""
Estimates camera intrinsics (K, dist) from checkerboard images.
Square size is set to 1.0 (arbitrary units) — does not affect K or dist.

Usage:
    python src/pipeline/calibrate_camera.py
    --calib_dir ./data/raw/calibration_images
    --output    ./data/calibration
    --nx        8
    --ny        5
"""

import cv2
import numpy as np
import glob
import os
import argparse
import sys


def collect_calibration_points(image_paths, nx, ny):
    # Unit square spacing — scale does not affect K or distortion coefficients
    objp = np.zeros((nx * ny, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Sub-pixel refinement termination criteria:
    # Stop when either max 30 iterations are reached OR corner moves < 0.001 px.
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints, imgpoints, valid_paths = [], [], []
    image_size = None

    print(f"\nProcessing {len(image_paths)} images for {nx}×{ny} pattern...\n")

    for i, fpath in enumerate(image_paths):
        img = cv2.imread(fpath)
        if img is None:
            print(f"  [{i+1:02d}] SKIP  — unreadable: {os.path.basename(fpath)}")
            continue

        if image_size is None:
            image_size = (img.shape[1], img.shape[0])

        # Convert to grayscale for faster corner detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, (nx, ny),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not ret:
            print(f"  [{i+1:02d}] FAIL  — corners not found: {os.path.basename(fpath)}")
            continue

        # Refine corner locations using sub-pixel accuracy
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), subpix_criteria
        )

        objpoints.append(objp)
        imgpoints.append(corners_refined)
        valid_paths.append(fpath)
        print(f"  [{i+1:02d}] OK    — {os.path.basename(fpath)}")

    print(f"\n{len(valid_paths)}/{len(image_paths)} images succeeded.")

    return objpoints, imgpoints, image_size, valid_paths


def run_calibration(objpoints, imgpoints, image_size):
    print("\nRunning calibration optimisation...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    return K, dist, rms, rvecs, tvecs


def compute_reprojection_errors(objpoints, imgpoints, K, dist, rvecs, tvecs):
    errors = []
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
        errors.append(err)
    return errors


def save_results(K, dist, rms, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "camera_matrix.npy"), K)
    np.save(os.path.join(output_dir, "dist_coeffs.npy"), dist)
    np.savez(
        os.path.join(output_dir, "calibration_data.npz"),
        camera_matrix=K,
        dist_coeffs=dist,
        rms_reprojection_error=rms
    )
    print(f"\nSaved calibration data to: {output_dir}/")


def save_undistortion_check(image_path, K, dist, output_dir):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_16SC2)
    undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    line_y = h // 2
    cv2.line(img,         (0, line_y), (w, line_y), (0, 0, 255), 2)
    cv2.line(undistorted, (0, line_y), (w, line_y), (0, 0, 255), 2)

    cv2.putText(img,         "ORIGINAL",    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(undistorted, "UNDISTORTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    out_path = os.path.join(output_dir, "undistortion_check.jpg")
    cv2.imwrite(out_path, np.hstack([img, undistorted]))
    print(f"Visual check saved to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Camera calibration from checkerboard images.")
    parser.add_argument("--calib_dir", type=str, required=True)
    parser.add_argument("--output",    type=str, default="./data/calibration")
    parser.add_argument("--nx",        type=int, default=8, help="Inner corners — horizontal")
    parser.add_argument("--ny",        type=int, default=5, help="Inner corners — vertical")
    return parser.parse_args()


def main():
    args = parse_args()

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_paths = sorted(
        p for pat in patterns
        for p in glob.glob(os.path.join(args.calib_dir, pat))
    )

    if not image_paths:
        print(f"ERROR: No images found in '{args.calib_dir}'")
        sys.exit(1)

    print(f"Found {len(image_paths)} images | corners: {args.nx}×{args.ny} | units: arbitrary")

    objpoints, imgpoints, image_size, valid_paths = collect_calibration_points(
        image_paths, args.nx, args.ny
    )

    if len(valid_paths) < 5:
        print("\nERROR: Too few valid images. Check --nx/--ny or image quality.")
        sys.exit(1)

    K, dist, rms, rvecs, tvecs = run_calibration(objpoints, imgpoints, image_size)

    per_image_errors = compute_reprojection_errors(objpoints, imgpoints, K, dist, rvecs, tvecs)
    worst_idx  = int(np.argmax(per_image_errors))
    worst_err  = per_image_errors[worst_idx]
    worst_file = os.path.basename(valid_paths[worst_idx])

    print("\n" + "=" * 50)
    print("CALIBRATION RESULTS")
    print("=" * 50)
    print(f"RMS reprojection error : {rms:.4f} px  ", end="")
    print("(EXCELLENT)" if rms < 0.5 else "(GOOD)" if rms < 1.0 else "(POOR — remove blurry images)")
    print(f"Worst single image     : {worst_err:.4f} px  ({worst_file})")
    print(f"\nCamera Matrix K:\n{K}")
    print(f"\nDistortion coefficients [k1, k2, p1, p2, k3]:\n{dist}")
    print(f"\nfx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    print(f"Image size: {image_size[0]}×{image_size[1]} px")

    if worst_err > 2.0:
        print(f"\nWARNING: '{worst_file}' has high reprojection error ({worst_err:.3f} px).")
        print("Consider removing it and re-running calibration.")

    save_results(K, dist, rms, args.output)
    save_undistortion_check(valid_paths[0], K, dist, args.output)

    print(f"\nDone. Heading controller should load: {os.path.join(args.output, 'calibration_data.npz')}")


if __name__ == "__main__":
    main()