#!/usr/bin/env python3
"""
gesture_to_motion — turns gesture names into arm movements.

Topics
  subscribes  /gesture            std_msgs/String               from gesture_detector
  subscribes  /hand/position      geometry_msgs/Point           from gesture_detector
  publishes   /arm/joint_targets  std_msgs/Float64MultiArray    to arm_driver
  publishes   /arm/gripper        std_msgs/Float64              to arm_driver

To change what the arm does, edit POSES and ACTIONS below.
Joint order is always [base, shoulder, elbow, wrist], in radians.
"""

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray, String

# ---------------------------------------------------------------------------
# Named poses. Try changing the numbers! (They are clamped by the driver's
# joint limits, so a bad number won't break anything.)
# ---------------------------------------------------------------------------
POSES = {
    'home':       [0.0, -1.0, 0.3, 0.7],
    'reach':      [0.0, 0.5, -0.2, -0.3],
    'bow':        [0.0, 0.8, 0.4, 0.3],
    'wave_left':  [0.5, -0.5, -0.3, -0.2],
    'wave_right': [-0.5, -0.5, -0.3, -0.2],
    # Used by "follow" mode: hand high in the picture -> follow_high, low -> follow_low
    'follow_high': [0.0, -0.8, 0.0, 0.8],
    'follow_low':  [0.0, 0.4, 0.2, 0.2],
}

# ---------------------------------------------------------------------------
# What each gesture does. Action types:
#   ('pose', name)                      move to one pose
#   ('gripper', amount)                 0.0 = open ... 1.0 = closed
#   ('sequence', [(pose, seconds), ...]) play poses one after another
#   ('follow',)                          the arm follows your index finger
# Gestures not listed here (like "None") do nothing: the arm stays put.
# ---------------------------------------------------------------------------
ACTIONS = {
    'Open_Palm':   ('gripper', 0.0),
    'Closed_Fist': ('gripper', 1.0),
    'Thumb_Up':    ('pose', 'home'),
    'ILoveYou':    ('pose', 'reach'),
    'Thumb_Down':  ('sequence', [('bow', 1.5), ('home', 1.5)]),
    'Victory':     ('sequence', [('wave_left', 1.0), ('wave_right', 1.0),
                                 ('wave_left', 1.0), ('wave_right', 1.0),
                                 ('home', 1.5)]),
    'Pointing_Up': ('follow',),
}


class GestureToMotion(Node):

    def __init__(self):
        super().__init__('gesture_to_motion')
        p = self.declare_parameter
        self.base_range = p('follow_base_range', 0.8).value   # max base swing (rad)
        self.base_sign = p('follow_base_sign', 1.0).value     # set -1.0 if it turns the wrong way
        self.smoothing = p('follow_smoothing', 0.2).value     # 0..1, lower = smoother but laggier
        self.max_step = p('follow_max_step', 0.05).value      # max change per tick (rad)
        rate = p('rate', 20.0).value

        self.joint_pub = self.create_publisher(Float64MultiArray, '/arm/joint_targets', 10)
        self.gripper_pub = self.create_publisher(Float64, '/arm/gripper', 10)
        self.create_subscription(String, '/gesture', self._on_gesture, 10)
        self.create_subscription(Point, '/hand/position', self._on_hand, 10)

        self.mode = 'idle'              # 'idle', 'sequence' or 'follow'
        self.sequence = []              # remaining (pose, seconds) steps
        self.step_time_left = 0.0
        self.hand = None                # latest (x, y) of the index fingertip
        self.last_pose = list(POSES['home'])   # the driver starts at home
        self.follow_target = list(POSES['home'])
        self.dt = 1.0 / rate
        self.create_timer(self.dt, self._tick)
        self.get_logger().info('gesture_to_motion ready')

    def _send_pose(self, angles):
        self.last_pose = list(angles)
        self.joint_pub.publish(Float64MultiArray(data=[float(a) for a in angles]))

    def _on_gesture(self, msg):
        action = ACTIONS.get(msg.data)
        if action is None:
            if self.mode == 'follow':
                self.mode = 'idle'      # stop following when the pointing hand goes away
            return

        self.get_logger().info(f'{msg.data} -> {action[0]}')
        kind = action[0]
        if kind == 'gripper':
            # The gripper can open/close without interrupting a sequence.
            self.gripper_pub.publish(Float64(data=float(action[1])))
        elif kind == 'pose':
            self.mode = 'idle'
            self._send_pose(POSES[action[1]])
        elif kind == 'sequence':
            self.mode = 'sequence'
            self.sequence = list(action[1])
            self.step_time_left = 0.0   # start the first step on the next tick
        elif kind == 'follow':
            self.mode = 'follow'
            self.follow_target = list(self.last_pose)   # start from where the arm was sent last

    def _on_hand(self, msg):
        self.hand = (msg.x, msg.y)

    def _tick(self):
        if self.mode == 'sequence':
            self.step_time_left -= self.dt
            if self.step_time_left <= 0.0:
                if not self.sequence:
                    self.mode = 'idle'
                    return
                pose, seconds = self.sequence.pop(0)
                self._send_pose(POSES[pose])
                self.step_time_left = seconds

        elif self.mode == 'follow' and self.hand is not None:
            x, y = self.hand
            # Base: hand left/right of the picture centre -> turn left/right.
            base = self.base_sign * (0.5 - x) * 2.0 * self.base_range
            # Arm height: blend between the two follow poses using y (0 = top).
            t = min(max((y - 0.2) / 0.6, 0.0), 1.0)
            high, low = POSES['follow_high'], POSES['follow_low']
            goal = [base] + [high[i] + t * (low[i] - high[i]) for i in range(1, 4)]

            # Smooth and rate-limit so the arm glides instead of jittering.
            for i in range(4):
                step = self.smoothing * (goal[i] - self.follow_target[i])
                step = min(max(step, -self.max_step), self.max_step)
                self.follow_target[i] += step
            self._send_pose(self.follow_target)


def main(args=None):
    rclpy.init(args=args)
    node = GestureToMotion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()