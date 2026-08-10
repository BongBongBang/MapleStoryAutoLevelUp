'''
Periodically capture the MapleStory game window.

Usage:
python tools/scheduledScreenshot.py [interval_seconds]
'''
# Standard import
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Library import
import cv2

# Local import
from src.utils.global_var import WINDOW_WORKING_SIZE
from src.utils.logger import logger
from src.utils.common import is_mac, load_yaml, override_cfg, screenshot

if is_mac():
    from src.input.GameWindowCapturorForMac import GameWindowCapturor
else:
    from src.input.GameWindowCapturor import GameWindowCapturor


DEFAULT_INTERVAL_SECONDS = 60.0


def load_config():
    '''Load the default, platform, and custom configurations.'''
    cfg = load_yaml("config/config_default.yaml")
    if is_mac():
        cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
    return override_cfg(cfg, load_yaml("config/config_custom.yaml"))


def get_game_window_frame(capture, cfg):
    '''Capture and normalize the game window like routeRecorder.py.'''
    frame = capture.get_frame()
    if frame is None:
        logger.warning("Failed to capture game frame.")
        return None

    title_bar_height = cfg["game_window"]["title_bar_height"]
    frame_no_title = frame[title_bar_height:, :]
    expected_size = tuple(cfg["game_window"]["size"])
    if expected_size != frame_no_title.shape[:2]:
        logger.error(
            f"Unexpected window size: {frame_no_title.shape[:2]} "
            f"(expect {expected_size}). Please use windowed mode and the smallest resolution."
        )
        return None

    return cv2.resize(
        frame_no_title,
        WINDOW_WORKING_SIZE,
        interpolation=cv2.INTER_NEAREST
    )


def run(interval_seconds):
    '''Capture immediately, then continue at the requested interval.'''
    cfg = load_config()
    logger.info("Waiting for game window to activate, please click on game window")
    capture = GameWindowCapturor(cfg)

    try:
        while True:
            started_at = time.monotonic()
            frame = get_game_window_frame(capture, cfg)
            if frame is not None:
                screenshot(frame, "scheduled")

            elapsed = time.monotonic() - started_at
            time.sleep(max(0.0, interval_seconds - elapsed))
    finally:
        capture.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Periodically capture the MapleStory game window."
    )
    parser.add_argument(
        "interval",
        nargs="?",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help=f"seconds between screenshots (default: {DEFAULT_INTERVAL_SECONDS:g})"
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be greater than 0")

    try:
        run(args.interval)
    except KeyboardInterrupt:
        logger.info("Scheduled screenshot stopped by user.")
    except Exception as error:
        logger.error(f"Scheduled screenshot failed: {error}")
        sys.exit(1)
