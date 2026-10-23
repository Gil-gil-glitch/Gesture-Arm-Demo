#!/usr/bin/env python3
"""
pose_mimic — the robot arm copies your right arm.

  Raise your right arm out to the side  -> the robot copies your shoulder and elbow
  Make a fist / open your hand           -> the gripper closes / opens
  Lower your arm                         -> the robot goes back to its rest pose
  Step left or right                     -> the robot base turns (optional)

How it works
  1. MediaPipe Pose finds 33 points on your body. We use three of them:
     your shoulder, elbow and wrist.
  2. We measure the angle of your upper arm and of your forearm, like
     measuring with a protractor on the camera picture.
  3. Those angles become the robot's shoulder (joint2) and elbow (joint3).
  4. MediaPipe's gesture recognizer looks at the hand nearest your wrist
     to open or close the gripper.

Only side-to-side and up-down movement counts. A single webcam can't see how
far your hand is from the camera, so reaching toward the camera does nothing.

Topics
  subscribes  /joint_states        sensor_msgs/JointState       where the robot is now
  publishes   /arm/joint_targets   std_msgs/Float64MultiArray   [base, shoulder, elbow, wrist] (rad)
  publishes   /arm/gripper         std_msgs/Float64             0.0 = open, 1.0 = closed
  calls       /arm/home            std_srvs/Trigger             when you lower your arm

Robot angle convention (OpenManipulator-X): with every joint at 0 the upper arm
points straight up and the forearm points straight forward. Positive joint2
tilts the upper arm forward; positive joint3 / joint4 tilt the forearm / gripper down.
"""

import math
import os
import time

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import Trigger

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions

# MediaPipe Pose landmark numbers (the person's own left/right)
BODY = {
    'right': {'shoulder': 12, 'elbow': 14, 'wrist': 16, 'hip': 24},
    'left':  {'shoulder': 11, 'elbow': 13, 'wrist': 15, 'hip': 23},
}
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24
HAND_WRIST, HAND_MIDDLE_KNUCKLE = 0, 9      # hand landmark numbers

# Robot link lengths in mm, only used to draw the little robot in the GUI
LINK_UPPER, LINK_FORE, LINK_GRIP = 130, 124, 126

# Colours (BGR)
WHITE, GREY, DARK = (255, 255, 255), (150, 150, 150), (45, 45, 45)
PANEL_BG = (28, 28, 28)
ORANGE, GREEN, BLUE = (0, 140, 255), (80, 200, 80), (255, 170, 60)
PANEL_WIDTH = 340
WINDOW_NAME = 'Gesture Arm Demo - Mimic'


def wrap(a):
    """Keep an angle between -180 and +180 degrees (in radians)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angle_from_up(out, up):
    """Angle of a vector measured from straight up, positive toward 'out'.
    Straight up = 0, straight out = 90 deg, straight down = 180 deg."""
    return math.atan2(out, up)


class PoseMimic(Node):

    def __init__(self):
        super().__init__('pose_mimic')
        share = get_package_share_directory('gesture-arm-demo')   # must match package_name in setup.py
        p = self.declare_parameter
        self.camera_index = p('camera_index', 0).value
        pose_model = p('pose_model_path', os.path.join(share, 'models', 'pose_landmarker_lite.task')).value
        gesture_model = p('gesture_model_path', os.path.join(share, 'models', 'gesture_recognizer.task')).value
        self.side = p('arm_side', 'right').value            # which of YOUR arms to copy
        self.wrist_mode = p('wrist_mode', 'level').value    # 'level' or 'follow'
        self.rest_angle = math.radians(p('rest_angle_deg', 135.0).value)
        self.min_visibility = p('min_visibility', 0.5).value
        self.lost_frames_to_rest = p('lost_frames_to_rest', 15).value
        self.base_range = p('base_range', 0.6).value        # rad; 0.0 = base never turns
        self.base_sign = p('base_sign', 1.0).value
        self.smoothing = p('smoothing', 0.3).value          # 0..1, lower = smoother
        self.max_step = p('max_step', 0.06).value           # rad per frame
        self.min_gesture_score = p('min_gesture_score', 0.6).value
        self.stable_frames = p('stable_frames', 4).value
        self.show_window = p('show_window', True).value
        fps = p('fps', 30.0).value

        if self.side not in BODY:
            raise RuntimeError("arm_side must be 'right' or 'left'")
        # On the raw (un-mirrored) camera picture, your right arm reaching
        # away from your body moves toward the LEFT of the image.
        self.out_sign = -1.0 if self.side == 'right' else 1.0

        for path in (pose_model, gesture_model):
            if not os.path.exists(path):
                raise RuntimeError(f'Model not found: {path}\nDownload it into models/ (see README), then rebuild.')

        self.pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model),
            running_mode=RunningMode.VIDEO, num_poses=1))
        self.hands = GestureRecognizer.create_from_options(GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=gesture_model),
            running_mode=RunningMode.VIDEO, num_hands=2))

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f'Could not open camera {self.camera_index}')
        if self.show_window:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        self.joint_pub = self.create_publisher(Float64MultiArray, '/arm/joint_targets', 10)
        self.gripper_pub = self.create_publisher(Float64, '/arm/gripper', 10)
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10)
        self.home_client = self.create_client(Trigger, '/arm/home')

        self.last_hand_landmarks = []
        self.robot_now = None          # latest joint angles from the driver
        self.target = None             # smoothed angles we are sending
        self.active = False            # True while copying your arm
        self.lost_frames = 0
        self.human = None              # (upper arm angle, elbow bend) for the GUI
        self.gripper_state = None      # 'Open_Palm' or 'Closed_Fist'
        self.candidate, self.candidate_count = None, 0
        self.seen_gesture, self.seen_score = 'None', 0.0

        self.start = time.monotonic()
        self.last_ts = -1
        self.create_timer(1.0 / fps, self._on_frame)
        self.get_logger().info(f'pose_mimic ready: raise your {self.side} arm out to the side')

    def _on_joint_states(self, msg):
        named = dict(zip(msg.name, msg.position))
        if all(f'joint{i}' in named for i in range(1, 5)):
            self.robot_now = [named[f'joint{i}'] for i in range(1, 5)]

    def _timestamp(self):
        ts = int((time.monotonic() - self.start) * 1000)
        self.last_ts = max(ts, self.last_ts + 1)
        return self.last_ts

    def _on_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn('Camera frame not received', throttle_duration_sec=2.0)
            return
        h, w = frame.shape[:2]

        # Run MediaPipe on the RAW picture so "right arm" really means your right
        # arm. The picture is only mirrored afterwards, for display.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts = self._timestamp()
        pose_result = self.pose.detect_for_video(image, ts)
        hand_result = self.hands.recognize_for_video(image, ts)

        body = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
        hand = self._pick_hand(body, hand_result)

        goal = self._arm_goal(body, hand, w, h)
        if goal is not None:
            self.lost_frames = 0
            if not self.active:
                self.active = True
                self.target = list(self.robot_now) if self.robot_now else list(goal)
                self.get_logger().info('Copying your arm')
            self._step_toward(goal)
            self.joint_pub.publish(Float64MultiArray(data=[float(a) for a in self.target]))
        else:
            self.lost_frames += 1
            if self.active and self.lost_frames >= self.lost_frames_to_rest:
                self.active = False
                self.human = None
                self._go_home()

        self._update_gripper(hand_result, hand)

        if self.show_window:
            self._draw(frame, body, hand_result, hand)

    def _arm_goal(self, body, hand_index, w, h):
        """Turn your shoulder/elbow/wrist positions into 4 robot joint angles.
        Returns None if your arm is down or can't be seen."""
        if body is None:
            return None
        ids = BODY[self.side]
        s, e, wr = body[ids['shoulder']], body[ids['elbow']], body[ids['wrist']]
        if min(s.visibility, e.visibility, wr.visibility) < self.min_visibility:
            return None

        # Vectors in pixels, rewritten as (out, up): "out" = away from your body,
        # "up" = toward the ceiling. (Image y grows downward, so we flip it.)
        def vec(a, b):
            return (self.out_sign * (b.x - a.x) * w, -(b.y - a.y) * h)

        upper = vec(s, e)
        fore = vec(e, wr)
        upper_angle = angle_from_up(*upper)     # 0 = arm up, 90 deg = out, 180 deg = down
        fore_angle = angle_from_up(*fore)

        elbow_bend = math.pi - abs(wrap(fore_angle - upper_angle))   # 180 = straight arm
        self.human = (upper_angle, elbow_bend)

        if abs(upper_angle) > self.rest_angle:  # arm hanging down -> rest
            return None

        # Your "out" is the robot's "forward". Your "up" is the robot's "up".
        shoulder = upper_angle
        # Robot forearm direction (from up) is joint2 + joint3 + 90 deg.
        elbow = wrap(fore_angle - upper_angle - math.pi / 2)

        if self.wrist_mode == 'follow' and hand_index is not None:
            lm = self.last_hand_landmarks[hand_index]
            hand_vec = vec(lm[HAND_WRIST], lm[HAND_MIDDLE_KNUCKLE])
            wrist = wrap(angle_from_up(*hand_vec) - fore_angle)
        else:
            wrist = wrap(-(shoulder + elbow))   # keep the gripper level

        # Base: turn with your body's left/right position in the picture.
        centre_x = (body[L_SHOULDER].x + body[R_SHOULDER].x) / 2.0
        base = self.base_sign * (centre_x - 0.5) * 2.0 * self.base_range

        return [base, shoulder, elbow, wrist]

    def _step_toward(self, goal):
        """Smooth and speed-limit the motion so the robot glides instead of shaking."""
        for i in range(4):
            step = self.smoothing * (goal[i] - self.target[i])
            self.target[i] += min(max(step, -self.max_step), self.max_step)

    def _go_home(self):
        self.get_logger().info('Arm lowered: going to rest pose')
        if self.home_client.service_is_ready():
            self.home_client.call_async(Trigger.Request())
        else:
            self.get_logger().warn('/arm/home not available. Is arm_driver running?')

    
    def _pick_hand(self, body, hand_result):
        """Return the index of the detected hand closest to your tracked wrist."""
        self.last_hand_landmarks = hand_result.hand_landmarks
        if body is None or not hand_result.hand_landmarks:
            return None
        wr = body[BODY[self.side]['wrist']]
        best, best_d = None, 0.15               # must be within 15% of the picture
        for i, lm in enumerate(hand_result.hand_landmarks):
            d = math.hypot(lm[HAND_WRIST].x - wr.x, lm[HAND_WRIST].y - wr.y)
            if d < best_d:
                best, best_d = i, d
        return best

    def _update_gripper(self, hand_result, hand_index):
        name, score = 'None', 0.0
        if hand_index is not None and hand_result.gestures:
            top = hand_result.gestures[hand_index][0]
            if top.score >= self.min_gesture_score:
                name, score = top.category_name, top.score
        self.seen_gesture, self.seen_score = name, score
        if name not in ('Open_Palm', 'Closed_Fist'):
            self.candidate, self.candidate_count = None, 0
            return
        if name == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate, self.candidate_count = name, 1
        if self.candidate_count >= self.stable_frames and name != self.gripper_state:
            self.gripper_state = name
            self.gripper_pub.publish(Float64(data=1.0 if name == 'Closed_Fist' else 0.0))
            self.get_logger().info('Gripper ' + ('closed' if name == 'Closed_Fist' else 'open'))

    # GUI
    def _draw(self, frame, body, hand_result, hand_index):
        frame = cv2.flip(frame, 1)              # mirror for display only
        h, w = frame.shape[:2]
        px = lambda lm: (int((1.0 - lm.x) * w), int(lm.y * h))   # mirrored pixel position

        if body is not None:
            # Faint torso outline for context
            for a, b in [(L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP)]:
                cv2.line(frame, px(body[a]), px(body[b]), GREY, 2)
            # The tracked arm, thick
            ids = BODY[self.side]
            colour = GREEN if self.active else ORANGE
            pts = [px(body[ids[k]]) for k in ('shoulder', 'elbow', 'wrist')]
            cv2.line(frame, pts[0], pts[1], colour, 6)
            cv2.line(frame, pts[1], pts[2], colour, 6)
            for pt in pts:
                cv2.circle(frame, pt, 9, WHITE, -1)
                cv2.circle(frame, pt, 9, colour, 2)
            if self.human is not None:
                cv2.putText(frame, f'{math.degrees(self.human[1]):.0f}', (pts[1][0] + 12, pts[1][1] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

        if hand_index is not None:
            lm = hand_result.hand_landmarks[hand_index]
            for pt in (px(p) for p in lm):
                cv2.circle(frame, pt, 3, BLUE, -1)

        # Status banner
        if self.active:
            text, colour = 'COPYING YOUR ARM', GREEN
        else:
            text, colour = f'Raise your {self.side} arm out to the side', GREY
        cv2.rectangle(frame, (0, 0), (w, 52), (0, 0, 0), -1)
        cv2.putText(frame, text, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)

        cv2.imshow(WINDOW_NAME, cv2.hconcat([frame, self._make_panel(h)]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

    def _make_panel(self, h):
        font = cv2.FONT_HERSHEY_SIMPLEX
        panel = np.full((h, PANEL_WIDTH, 3), PANEL_BG, dtype=np.uint8)
        y = 32
        cv2.putText(panel, 'YOU', (16, y), font, 0.7, WHITE, 2)
        y += 28
        if self.human is not None:
            cv2.putText(panel, f'Upper arm: {math.degrees(self.human[0]):5.0f} deg from up', (16, y), font, 0.5, WHITE, 1)
            cv2.putText(panel, f'Elbow bend: {math.degrees(self.human[1]):4.0f} deg', (16, y + 22), font, 0.5, WHITE, 1)
        else:
            cv2.putText(panel, 'arm not raised', (16, y), font, 0.5, GREY, 1)
        y += 56

        cv2.putText(panel, 'ROBOT', (16, y), font, 0.7, WHITE, 2)
        y += 8
        angles = self.target if (self.active and self.target) else self.robot_now
        self._draw_robot(panel, (PANEL_WIDTH // 2 - 60, y + 140), angles)
        y += 178
        if angles:
            for i, name in enumerate(['base', 'shoulder', 'elbow', 'wrist']):
                cv2.putText(panel, f'{name:>8}: {math.degrees(angles[i]):6.0f} deg', (16, y + i * 20),
                            font, 0.5, WHITE, 1)
        y += 80

        grip = {'Closed_Fist': 'CLOSED', 'Open_Palm': 'OPEN'}.get(self.gripper_state, '--')
        cv2.putText(panel, f'Gripper: {grip}', (16, y), font, 0.65, GREEN if grip != '--' else GREY, 2)
        seen = self.seen_gesture if self.seen_gesture != 'None' else 'no hand gesture'
        cv2.putText(panel, f'seeing: {seen}', (16, y + 22), font, 0.45, GREY, 1)
        cv2.putText(panel, 'fist = close   open hand = open', (16, h - 34), font, 0.42, GREY, 1)
        cv2.putText(panel, 'lower your arm = rest   q = quit', (16, h - 14), font, 0.42, GREY, 1)
        return panel

    def _draw_robot(self, panel, origin, angles):
        """Side view of the robot arm (forward = right) from its joint angles."""
        ox, oy = origin
        scale = 0.55
        cv2.rectangle(panel, (ox - 30, oy), (ox + 30, oy + 14), GREY, -1)       # base
        if not angles:
            return
        _, j2, j3, j4 = angles
        pts = [(ox, oy)]
        heading = [(math.sin(j2), math.cos(j2)),                                  # upper arm
                   (math.cos(j2 + j3), -math.sin(j2 + j3)),                       # forearm
                   (math.cos(j2 + j3 + j4), -math.sin(j2 + j3 + j4))]             # gripper
        for (fx, fy), length in zip(heading, (LINK_UPPER, LINK_FORE, LINK_GRIP)):
            x, y = pts[-1]
            pts.append((int(x + fx * length * scale), int(y - fy * length * scale)))
        colours = [GREEN, GREEN, BLUE]
        for i in range(3):
            cv2.line(panel, pts[i], pts[i + 1], colours[i], 6)
        for pt in pts[:-1]:
            cv2.circle(panel, pt, 6, WHITE, -1)

    def close(self):
        self.cap.release()
        self.pose.close()
        self.hands.close()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PoseMimic()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f'[pose_mimic] {e}')
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()