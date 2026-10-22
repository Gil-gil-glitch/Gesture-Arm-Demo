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

    def _draw(self, frame, result, name, score):
        h, w = frame.shape[:2]
        for hand in result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
            for x, y in pts:
                cv2.circle(frame, (x, y), 4, (0, 140, 255), -1)

        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(frame, f'Seeing: {name} ({score:.2f})', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f'Arm command: {self.current}', (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.imshow('Gesture detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

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