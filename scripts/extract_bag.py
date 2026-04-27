"""Extract paired left/right images and K.txt from a ROS2 bag for FoundationStereo."""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")
from Utils import set_logging_format  # noqa: E402


def decode_image(raw: bytes, encoding: str, height: int, width: int, step: int) -> np.ndarray | None:
    """Decode a sensor_msgs/Image payload into a BGR (or mono) numpy array suitable for cv2.imwrite."""
    buf = np.frombuffer(raw, dtype=np.uint8)
    enc = encoding.lower()

    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(height, step)[:, : width * 3].reshape(height, width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == "rgb8" else img
    if enc in ("rgba8", "bgra8"):
        img = buf.reshape(height, step)[:, : width * 4].reshape(height, width, 4)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if enc == "rgba8" else cv2.COLOR_BGRA2BGR)
        return bgr
    if enc == "mono8":
        return buf.reshape(height, step)[:, :width]
    if enc == "mono16":
        buf16 = np.frombuffer(raw, dtype=np.uint16)
        return buf16.reshape(height, step // 2)[:, :width]

    logging.warning("Unsupported image encoding: %s — skipping frame", encoding)
    return None


def pair_by_nearest(left_stamps: list[int], right_stamps: list[int], tol_ns: int) -> list[tuple[int, int]]:
    """Two-pointer pairing of left/right indices by closest timestamp within tol_ns."""
    pairs: list[tuple[int, int]] = []
    j = 0
    for i, ls in enumerate(left_stamps):
        while j + 1 < len(right_stamps) and abs(right_stamps[j + 1] - ls) <= abs(right_stamps[j] - ls):
            j += 1
        if j < len(right_stamps) and abs(right_stamps[j] - ls) <= tol_ns:
            pairs.append((i, j))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", default="bag/rosbag2_2026_04_27-17_38_03", type=str, help="path to rosbag2 directory")
    parser.add_argument("--out_dir", default="bag/extracted", type=str, help="output directory")
    parser.add_argument("--left_image_topic", default="/zed/left_720p/image_rect_color", type=str)
    parser.add_argument("--right_image_topic", default="/zed/right_720p/image_rect_color", type=str)
    parser.add_argument("--left_info_topic", default="/zed/left_720p/camera_info", type=str)
    parser.add_argument("--right_info_topic", default="/zed/right_720p/camera_info", type=str)
    parser.add_argument(
        "--max_time_diff_ms", default=10.0, type=float, help="max stamp gap to accept a left/right pair"
    )
    parser.add_argument(
        "--frame_idx",
        default=None,
        type=int,
        help="if set, write only this paired frame as left.png / right.png (no per-frame folders)",
    )
    args = parser.parse_args()

    set_logging_format()

    bag_path = Path(args.bag)
    if not bag_path.exists():
        raise SystemExit(f"Bag path does not exist: {bag_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    needed = {args.left_image_topic, args.right_image_topic, args.left_info_topic, args.right_info_topic}

    left_info = None
    right_info = None
    left_msgs: list[tuple[int, object]] = []
    right_msgs: list[tuple[int, object]] = []

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        available = {c.topic for c in reader.connections}
        missing = needed - available
        if missing:
            raise SystemExit(
                f"Missing topics in bag: {sorted(missing)}\nAvailable topics:\n  " + "\n  ".join(sorted(available))
            )

        connections = [c for c in reader.connections if c.topic in needed]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == args.left_info_topic:
                if left_info is None:
                    left_info = msg
            elif connection.topic == args.right_info_topic:
                if right_info is None:
                    right_info = msg
            elif connection.topic == args.left_image_topic:
                left_msgs.append((timestamp, msg))
            elif connection.topic == args.right_image_topic:
                right_msgs.append((timestamp, msg))

    if left_info is None or right_info is None:
        raise SystemExit("Did not see at least one CameraInfo on each side — cannot build K.txt")
    if not left_msgs or not right_msgs:
        raise SystemExit("No image messages found on one or both image topics")

    left_msgs.sort(key=lambda x: x[0])
    right_msgs.sort(key=lambda x: x[0])

    tol_ns = int(args.max_time_diff_ms * 1e6)
    pairs = pair_by_nearest([t for t, _ in left_msgs], [t for t, _ in right_msgs], tol_ns)
    logging.info(
        "Bag has %d left, %d right images; matched %d pairs (tol=%.1fms)",
        len(left_msgs),
        len(right_msgs),
        len(pairs),
        args.max_time_diff_ms,
    )
    if not pairs:
        raise SystemExit("No left/right pairs within the time tolerance — try increasing --max_time_diff_ms")

    K = np.asarray(left_info.k, dtype=np.float64).reshape(3, 3)
    P_right = np.asarray(right_info.p, dtype=np.float64).reshape(3, 4)
    fx = P_right[0, 0]
    if fx == 0:
        raise SystemExit("right camera P[0,0] (fx) is zero — bag's right CameraInfo looks unrectified")
    baseline = -P_right[0, 3] / fx
    if baseline <= 0:
        logging.warning("Computed baseline is non-positive (%.6f m) — check left/right topic ordering", baseline)

    K_line = " ".join(f"{v:.10g}" for v in K.flatten())
    (out_dir / "K.txt").write_text(f"{K_line}\n{baseline:.6f}\n")
    logging.info("Wrote %s", out_dir / "K.txt")
    logging.info("K diag: fx=%.4f fy=%.4f cx=%.4f cy=%.4f, baseline=%.6f m", K[0, 0], K[1, 1], K[0, 2], K[1, 2], baseline)

    def decode_pair(li: int, ri: int):
        l_stamp, l_msg = left_msgs[li]
        _, r_msg = right_msgs[ri]
        l_img = decode_image(bytes(l_msg.data), l_msg.encoding, l_msg.height, l_msg.width, l_msg.step)
        r_img = decode_image(bytes(r_msg.data), r_msg.encoding, r_msg.height, r_msg.width, r_msg.step)
        return l_stamp, l_img, r_img

    if args.frame_idx is not None:
        if not (0 <= args.frame_idx < len(pairs)):
            raise SystemExit(f"--frame_idx {args.frame_idx} out of range [0, {len(pairs)})")
        li, ri = pairs[args.frame_idx]
        _, l_img, r_img = decode_pair(li, ri)
        if l_img is None or r_img is None:
            raise SystemExit("Failed to decode the requested frame")
        cv2.imwrite(str(out_dir / "left.png"), l_img)
        cv2.imwrite(str(out_dir / "right.png"), r_img)
        logging.info("Wrote single pair: %s and %s", out_dir / "left.png", out_dir / "right.png")
        return

    left_dir = out_dir / "left"
    right_dir = out_dir / "right"
    left_dir.mkdir(exist_ok=True)
    right_dir.mkdir(exist_ok=True)

    written = 0
    skipped = 0
    stamps_lines = []
    for li, ri in pairs:
        stamp, l_img, r_img = decode_pair(li, ri)
        if l_img is None or r_img is None:
            skipped += 1
            continue
        name = f"{written:06d}.png"
        cv2.imwrite(str(left_dir / name), l_img)
        cv2.imwrite(str(right_dir / name), r_img)
        stamps_lines.append(f"{written:06d} {stamp}")
        written += 1

    (out_dir / "timestamps.txt").write_text("\n".join(stamps_lines) + "\n")

    if written > 0:
        shutil.copyfile(left_dir / "000000.png", out_dir / "left.png")
        shutil.copyfile(right_dir / "000000.png", out_dir / "right.png")

    logging.info("Done: %d frames written, %d skipped (decode failures)", written, skipped)
    logging.info("Output: %s", out_dir.resolve())


if __name__ == "__main__":
    main()
