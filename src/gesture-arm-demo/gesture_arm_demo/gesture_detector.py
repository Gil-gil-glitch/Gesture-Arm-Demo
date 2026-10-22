#!/usr/bin/env python3
"""
gesture_detector — reads a webcam, recognizes hand gestures with MediaPipe,
and publishes what it sees.

Topics
  publishes  /gesture        std_msgs/String           stable gesture name, e.g. "Open_Palm"
  publishes  /hand/position  geometry_msgs/Point       index fingertip, x and y from 0.0 to 1.0
                                                       (x: left->right, y: top->bottom, z: unused)

MediaPipe's built-in gestures:
  None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou

A gesture is only published once it has been seen for `stable_frames` frames in
a row. This stops the arm reacting to a gesture that flickers for one frame
while your hand changes shape.
"""

import os
import time

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import String

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode = mp.tasks.vision.RunningMode

# Which of the 21 hand landmarks connect to which, for drawing the skeleton.
# (Landmark 0 is the wrist, 4 the thumb tip, 8 the index fingertip, ...)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm
]
INDEX_TIP = 8

# What the GUI panel lists. Keep the descriptions in sync with ACTIONS in
# gesture_to_motion.py if you change what a gesture does.
GESTURE_PANEL = [
    ('Open_Palm',   'Open palm',    'Open gripper'),
    ('Closed_Fist', 'Fist',         'Close gripper'),
    ('Thumb_Up',    'Thumbs up',    'Go home'),
    ('Thumb_Down',  'Thumbs down',  'Bow'),
    ('Victory',     'Peace sign',   'Wave'),
    ('ILoveYou',    'I love you',   'Reach forward'),
    ('Pointing_Up', 'Point up',     'Follow finger'),
]

# Colours (OpenCV uses BGR order)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)
DARK = (40, 40, 40)
PANEL_BG = (28, 28, 28)
ORANGE = (0, 140, 255)      # "seeing this gesture"
GREEN = (80, 200, 80)       # "the arm is doing this"
PANEL_WIDTH = 320
WINDOW_NAME = 'Gesture Arm Demo'


class GestureDetector(Node):

    def __init__(self):
        super().__init__('gesture_detector')

        default_model = os.path.join(
            get_package_share_directory('gesture_arm_demo'), 'models', 'gesture_recognizer.task')
        p = self.declare_parameter
        self.camera_index = p('camera_index', 0).value
        model_path = p('model_path', default_model).value
        self.min_score = p('min_score', 0.6).value
        self.stable_frames = p('stable_frames', 5).value
        self.show_window = p('show_window', True).value
        self.mirror = p('mirror', True).value      # True feels natural, like a mirror
        fps = p('fps', 30.0).value

        if not os.path.exists(model_path):
            raise RuntimeError(f'Model not found: {model_path}\n'
                               'Download gesture_recognizer.task (see README) into models/.')

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
        )
        self.recognizer = GestureRecognizer.create_from_options(options)

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f'Could not open camera {self.camera_index}')

        if self.show_window:
            # WINDOW_NORMAL lets you drag the window bigger for a projector.
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        self.gesture_pub = self.create_publisher(String, '/gesture', 10)
        self.position_pub = self.create_publisher(Point, '/hand/position', 10)

        self.candidate = 'None'     # gesture seen in the latest frames
        self.candidate_count = 0
        self.current = 'None'       # last gesture we actually published
        self.start_time = time.monotonic()
        self.last_timestamp_ms = -1

        self.create_timer(1.0 / fps, self._on_frame)
        self.get_logger().info('gesture_detector ready. Press q in the camera window to quit.')

    # -------------------------------------------------------------------
    def _on_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn('Camera frame not received', throttle_duration_sec=2.0)
            return
        if self.mirror:
            frame = cv2.flip(frame, 1)

        # MediaPipe wants RGB; OpenCV gives BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs a timestamp that always increases.
        timestamp_ms = int((time.monotonic() - self.start_time) * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        result = self.recognizer.recognize_for_video(image, timestamp_ms)

        name, score = 'None', 0.0
        if result.gestures:
            top = result.gestures[0][0]           # first hand, best guess
            if top.score >= self.min_score:
                name, score = top.category_name, top.score

        if result.hand_landmarks:
            tip = result.hand_landmarks[0][INDEX_TIP]
            self.position_pub.publish(Point(x=float(tip.x), y=float(tip.y), z=0.0))

        self._update_gesture(name)

        if self.show_window:
            self._draw(frame, result, name, score)

    def _update_gesture(self, name):
        """Only publish a gesture after it has been stable for a few frames."""
        if name == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate, self.candidate_count = name, 1

        if self.candidate_count >= self.stable_frames and self.candidate != self.current:
            self.current = self.candidate
            self.gesture_pub.publish(String(data=self.current))
            self.get_logger().info(f'Gesture: {self.current}')

    # GUI
    def _draw(self, frame, result, name, score):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # --- hand skeleton on the camera image ---
        for hand in result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], WHITE, 2)
            for x, y in pts:
                cv2.circle(frame, (x, y), 4, ORANGE, -1)

        # --- big label of the active gesture across the top of the camera ---
        label = self._pretty(self.current) if self.current != 'None' else 'No gesture'
        colour = GREEN if self.current != 'None' else GREY
        cv2.rectangle(frame, (0, 0), (w, 56), (0, 0, 0), -1)
        cv2.putText(frame, label, (14, 40), font, 1.2, colour, 3)

        # --- side panel ---
        panel = self._make_panel(h, name, score)
        cv2.imshow(WINDOW_NAME, cv2.hconcat([frame, panel]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

    def _make_panel(self, h, seeing, score):
        """
        This method creates a function to show the live camera feed while also showcasing what gesture is currently being detected along
        with its confidence.
        """

        font = cv2.FONT_HERSHEY_SIMPLEX
        panel = np.full((h, PANEL_WIDTH, 3), PANEL_BG, dtype=np.uint8)
        cv2.putText(panel, 'GESTURES', (16, 32), font, 0.7, WHITE, 2)
        cv2.putText(panel, 'orange = seeing   green = active', (16, 54), font, 0.42, GREY, 1)

        row_h = min(52, (h - 150) // len(GESTURE_PANEL))
        top = 70

        for i, (key, title, action) in enumerate(GESTURE_PANEL):
            y0, y1 = top + i * row_h, top + (i + 1) * row_h - 6
            active = key == self.current
            is_seen = key == seeing

            if active:
                cv2.rectangle(panel, (10, y0), (PANEL_WIDTH - 10, y1), GREEN, -1)
                text, sub = (0, 0, 0), (30, 30, 30)

            else:
                cv2.rectangle(panel, (10, y0), (PANEL_WIDTH - 10, y1), DARK, -1)
                text, sub = WHITE, GREY

            if is_seen and not active:
                cv2.rectangle(panel, (10, y0), (PANEL_WIDTH - 10, y1), ORANGE, 2)
                # Progress bar: how close this gesture is to being accepted.
                frac = min(self.candidate_count / self.stable_frames, 1.0) if seeing == self.candidate else 0
                cv2.rectangle(panel, (12, y1 - 4), (12 + int((PANEL_WIDTH - 24) * frac), y1 - 2), ORANGE, -1)

            cv2.putText(panel, title, (20, y0 + 19), font, 0.6, text, 2)
            cv2.putText(panel, '-> ' + action, (20, y0 + 35), font, 0.45, sub, 1)

        # Cnfidence bar at the bottom ---
        by = h - 50
        cv2.putText(panel, f'Confidence: {score:.2f}', (16, by - 10), font, 0.5, WHITE, 1)
        cv2.rectangle(panel, (16, by), (PANEL_WIDTH - 16, by + 14), DARK, -1)
        fill = int((PANEL_WIDTH - 32) * score)
        cv2.rectangle(panel, (16, by), (16 + fill, by + 14), ORANGE if score > 0 else DARK, -1)
        
        # Marker where min_score is: below it, gestures are ignored.
        mx = 16 + int((PANEL_WIDTH - 32) * self.min_score)
        cv2.line(panel, (mx, by - 3), (mx, by + 17), WHITE, 1)
        cv2.putText(panel, 'press q to quit', (16, h - 14), font, 0.42, GREY, 1)
        return panel

    @staticmethod
    def _pretty(key):
        for k, title, _ in GESTURE_PANEL:
            if k == key:
                return title
        return key

    def close(self):
        self.cap.release()
        self.recognizer.close()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GestureDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f'[gesture_detector] {e}')
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()