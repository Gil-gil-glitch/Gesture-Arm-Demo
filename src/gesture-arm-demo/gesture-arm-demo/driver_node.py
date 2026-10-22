#!/usr/bin/env python3
"""
arm_driver — a small ROS 2 driver for a 4-DOF + gripper DYNAMIXEL arm
(XM430-W350-R motors on RS-485, connected through a U2D2).

Topics
  publishes   /joint_states        sensor_msgs/JointState        joint1..joint4 + gripper (rad)
  subscribes  /arm/joint_targets   std_msgs/Float64MultiArray    [j1, j2, j3, j4] in radians
  subscribes  /arm/gripper         std_msgs/Float64              0.0 = open ... 1.0 = closed

Services
  /arm/home    std_srvs/Trigger    move slowly to the home pose
  /arm/torque  std_srvs/SetBool    true = torque on, false = torque off (the arm goes limp!)

All motor communication happens inside this node's callbacks. ROS 2 runs them
one at a time (single-threaded executor), so the serial port is never used by
two callbacks at once.
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import SetBool, Trigger

from dynamixel_sdk import (
    COMM_SUCCESS,
    GroupSyncRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)

# XM430 control table (Protocol 2.0)
# When creating a driver for servos, use the documentation to find the following values in the control table. Keep in mind that these values
# will change depending on the servo model and desired configuration.
#
# See: https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/
#
ADDR_RETURN_DELAY = 9          # 1 byte
ADDR_OPERATING_MODE = 11       # 1 byte  (EEPROM: torque must be OFF to change)
ADDR_TORQUE_ENABLE = 64        # 1 byte
ADDR_HW_ERROR = 70             # 1 byte
ADDR_GOAL_CURRENT = 102        # 2 bytes
ADDR_PROFILE_ACCEL = 108       # 4 bytes
ADDR_PROFILE_VEL = 112         # 4 bytes
ADDR_GOAL_POSITION = 116       # 4 bytes
ADDR_PRESENT_CURRENT = 126     # 2 bytes, followed directly by
ADDR_PRESENT_VELOCITY = 128    # 4 bytes, followed directly by
ADDR_PRESENT_POSITION = 132    # 4 bytes
READ_BLOCK_START = ADDR_PRESENT_CURRENT
READ_BLOCK_LEN = 10            # current + velocity + position in one read

MODE_POSITION = 3
MODE_CURRENT_BASED_POSITION = 5

RAD_PER_TICK = 2.0 * math.pi / 4096.0   # 0.088 degrees per tick. 
RAD_S_PER_VEL_UNIT = 0.229 * 2.0 * math.pi / 60.0
AMP_PER_CURRENT_UNIT = 0.00269

HW_ERROR_BITS = {
    0: 'input voltage',
    2: 'overheating',
    3: 'motor encoder',
    4: 'electrical shock',
    5: 'overload',
}


def to_signed(value, bits):
    """
    Convert an unsigned register value to a signed integer.
    """
    
    if value >= 1 << (bits - 1):
        value -= 1 << bits
    
    return value


def to_bytes(value, size):
    """
    Split an integer into little-endian bytes (what DYNAMIXEL expects).
    """
    value &= (1 << (8 * size)) - 1
    return [(value >> (8 * i)) & 0xFF for i in range(size)]


class ArmDriver(Node):

    def __init__(self):
        super().__init__('arm_driver')

        # ---------------- parameters (see config/arm.yaml) ----------------
        p = self.declare_parameter
        self.port_name = p('port', '/dev/ttyUSB0').value
        self.baudrate = p('baudrate', 1000000).value
        self.joint_ids = list(p('joint_ids', [1, 2, 3, 4]).value)
        self.gripper_id = p('gripper_id', 5).value
        self.joint_names = list(p('joint_names', ['joint1', 'joint2', 'joint3', 'joint4']).value)
        self.zero_ticks = list(p('zero_ticks', [2048, 2048, 2048, 2048]).value)
        self.directions = list(p('directions', [1, 1, 1, 1]).value)
        self.joint_min = list(p('joint_min', [-1.5, -1.5, -1.5, -1.7]).value)
        self.joint_max = list(p('joint_max', [1.5, 1.5, 1.4, 2.0]).value)
        self.home_pose = list(p('home_pose', [0.0, -1.0, 0.3, 0.7]).value)
        self.profile_vel = p('profile_velocity', 40).value
        self.profile_acc = p('profile_acceleration', 10).value
        self.gripper_open = p('gripper_open_ticks', 2048).value
        self.gripper_closed = p('gripper_closed_ticks', 2048).value
        self.gripper_current = p('gripper_goal_current', 150).value
        self.return_delay = p('return_delay_time', 0).value
        rate = p('publish_rate', 50.0).value
        move_home = p('move_home_on_start', True).value

        self.all_ids = self.joint_ids + [self.gripper_id]
        if self.gripper_open == self.gripper_closed:
            self.get_logger().warn('gripper_open_ticks == gripper_closed_ticks, so the gripper '
                                   'will not move. Calibrate them in config/arm.yaml.')

        # ---------------- open the U2D2 ----------------
        self.port = PortHandler(self.port_name)
        self.packet = PacketHandler(2.0)
        if not self.port.openPort():
            raise RuntimeError(f'Could not open {self.port_name}. Is the U2D2 plugged in? '
                               'Are you in the "dialout" group?')
        if not self.port.setBaudRate(self.baudrate):
            raise RuntimeError(f'Could not set baud rate {self.baudrate}')
        self.get_logger().info(f'Opened {self.port_name} at {self.baudrate} bps')

        # ---------------- set up the motors ----------------
        self._check_motors()
        self._configure_motors()

        # Sync read: one packet asks all five motors for current/velocity/position.
        self.sync_read = GroupSyncRead(self.port, self.packet, READ_BLOCK_START, READ_BLOCK_LEN)
        for dxl_id in self.all_ids:
            self.sync_read.addParam(dxl_id)
        # Sync write: one packet sets goal positions on several motors at once.
        self.sync_write = GroupSyncWrite(self.port, self.packet, ADDR_GOAL_POSITION, 4)

        # ---------------- ROS interfaces ----------------
        self.state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.create_subscription(Float64MultiArray, '/arm/joint_targets', self._on_joint_targets, 10)
        self.create_subscription(Float64, '/arm/gripper', self._on_gripper, 10)
        self.create_service(Trigger, '/arm/home', self._on_home)
        self.create_service(SetBool, '/arm/torque', self._on_torque)
        self.create_timer(1.0 / rate, self._publish_state)
        self._read_failures = 0

        if move_home:
            self._move_joints(self.home_pose)
            self.get_logger().info('Moving to home pose')
        self.get_logger().info('arm_driver ready')

    # Low-level helpers for sending and receiving data
    def _write(self, dxl_id, addr, size, value):
        """
        Write one register and report any error. Returns True on success.
        """
        
        writer = {1: self.packet.write1ByteTxRx,
                  2: self.packet.write2ByteTxRx,
                  4: self.packet.write4ByteTxRx}[size]
        result, error = writer(self.port, dxl_id, addr, value)
        if result != COMM_SUCCESS:
            self.get_logger().error(f'ID {dxl_id}: {self.packet.getTxRxResult(result)}')
            return False
        
        if error != 0:
            self.get_logger().warn(f'ID {dxl_id}: {self.packet.getRxPacketError(error)}')

        return True

    def _read(self, dxl_id, addr, size):
        reader = {1: self.packet.read1ByteTxRx,
                  2: self.packet.read2ByteTxRx,
                  4: self.packet.read4ByteTxRx}[size]
        value, result, _ = reader(self.port, dxl_id, addr)

        return value if result == COMM_SUCCESS else None

    def _set_torque(self, on):
        ok = True
        for dxl_id in self.all_ids:
            ok &= self._write(dxl_id, ADDR_TORQUE_ENABLE, 1, 1 if on else 0)

        return ok

    def _rad_to_ticks(self, i, rad):
        return int(round(self.zero_ticks[i] + self.directions[i] * rad / RAD_PER_TICK))

    def _ticks_to_rad(self, i, ticks):
        return self.directions[i] * (ticks - self.zero_ticks[i]) * RAD_PER_TICK

    def _send_goals(self, goals):
        """
        goals: dict {motor_id: goal_ticks}. Sends them all in one packet.
        """
        self.sync_write.clearParam()
        
        for dxl_id, ticks in goals.items():
            self.sync_write.addParam(dxl_id, to_bytes(ticks, 4))
        
        result = self.sync_write.txPacket()
        if result != COMM_SUCCESS:
            self.get_logger().error(f'Sync write failed: {self.packet.getTxRxResult(result)}')

    # Startup
    def _check_motors(self):
        """
        Ping every motor and report hardware errors before doing anything.
        """
        missing = []
        
        for dxl_id in self.all_ids:
            _, result, _ = self.packet.ping(self.port, dxl_id)
            if result != COMM_SUCCESS:
                missing.append(dxl_id)
                continue
        
            hw = self._read(dxl_id, ADDR_HW_ERROR, 1)
        
            if hw:
                names = [name for bit, name in HW_ERROR_BITS.items() if hw & (1 << bit)]
                self.get_logger().warn(f'ID {dxl_id} hardware error: {", ".join(names)} '
                                       '(power-cycle or reboot the motor to clear)')
        
        if missing:
            raise RuntimeError(f'No response from motor ID(s) {missing}. '
                               'Check power (Kobuki on?), cables, IDs and baud rate.')
        
        self.get_logger().info(f'Found all motors: {self.all_ids}')

    def _configure_motors(self):
        """
        perating mode lives in EEPROM, so torque has to be off while we change it.
        """
        self._set_torque(False)
        for dxl_id in self.joint_ids:
            self._write(dxl_id, ADDR_RETURN_DELAY, 1, self.return_delay)
            self._write(dxl_id, ADDR_OPERATING_MODE, 1, MODE_POSITION)
            self._write(dxl_id, ADDR_PROFILE_ACCEL, 4, self.profile_acc)
            self._write(dxl_id, ADDR_PROFILE_VEL, 4, self.profile_vel)

        # The gripper uses current-based position mode: it moves to a position
        # but never pushes harder than gripper_goal_current, so it can't crush things.
        g = self.gripper_id
        self._write(g, ADDR_RETURN_DELAY, 1, self.return_delay)
        self._write(g, ADDR_OPERATING_MODE, 1, MODE_CURRENT_BASED_POSITION)
        self._write(g, ADDR_GOAL_CURRENT, 2, self.gripper_current)
        self._write(g, ADDR_PROFILE_ACCEL, 4, self.profile_acc)
        self._write(g, ADDR_PROFILE_VEL, 4, self.profile_vel)

        # Set each goal to where the motor already is, so turning torque on
        # doesn't make the arm jump to some old goal position.
        for dxl_id in self.all_ids:
            present = self._read(dxl_id, ADDR_PRESENT_POSITION, 4)
            if present is not None:
                self._write(dxl_id, ADDR_GOAL_POSITION, 4, to_signed(present, 32))
        self._set_torque(True)

    # Motion Functions
    def _move_joints(self, angles):
        goals = {}

        for i, rad in enumerate(angles):
            clamped = min(max(rad, self.joint_min[i]), self.joint_max[i])
            if clamped != rad:
                self.get_logger().warn(
                    f'{self.joint_names[i]} target {rad:.2f} rad clamped to {clamped:.2f}',
                    throttle_duration_sec=1.0)
            goals[self.joint_ids[i]] = self._rad_to_ticks(i, clamped)

        self._send_goals(goals)

    def _on_joint_targets(self, msg):

        if len(msg.data) != len(self.joint_ids):
            self.get_logger().error(f'Expected {len(self.joint_ids)} joint values, got {len(msg.data)}')
            return
        
        self._move_joints(list(msg.data))

    def _on_gripper(self, msg):
        amount = min(max(msg.data, 0.0), 1.0)   # 0 = open, 1 = closed
        ticks = int(round(self.gripper_open + amount * (self.gripper_closed - self.gripper_open)))
        self._send_goals({self.gripper_id: ticks})

    def _on_home(self, request, response):
        self._move_joints(self.home_pose)
        response.success = True
        response.message = 'Moving to home pose'
        return response

    def _on_torque(self, request, response):

        if request.data:
            # Re-sync goals to present positions before enabling, to avoid a jump.
            for dxl_id in self.all_ids:
                present = self._read(dxl_id, ADDR_PRESENT_POSITION, 4)
                if present is not None:
                    self._write(dxl_id, ADDR_GOAL_POSITION, 4, to_signed(present, 32))

        response.success = self._set_torque(request.data)
        response.message = 'Torque ON' if request.data else 'Torque OFF (arm is limp)'
        return response

    # Feedback
    def _publish_state(self):
        result = self.sync_read.txRxPacket()
        if result != COMM_SUCCESS:
            self._read_failures += 1
            self.get_logger().warn(f'Sync read failed ({self.packet.getTxRxResult(result)}). '
                                   'If this keeps happening during fast moves, check the 12 V supply.',
                                   throttle_duration_sec=2.0)
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for i, dxl_id in enumerate(self.all_ids):
            cur = to_signed(self.sync_read.getData(dxl_id, ADDR_PRESENT_CURRENT, 2), 16)
            vel = to_signed(self.sync_read.getData(dxl_id, ADDR_PRESENT_VELOCITY, 4), 32)
            pos = to_signed(self.sync_read.getData(dxl_id, ADDR_PRESENT_POSITION, 4), 32)
            if dxl_id == self.gripper_id:
                msg.name.append('gripper')
                msg.position.append((pos - 2048) * RAD_PER_TICK)
                msg.velocity.append(vel * RAD_S_PER_VEL_UNIT)
            else:
                msg.name.append(self.joint_names[i])
                msg.position.append(self._ticks_to_rad(i, pos))
                msg.velocity.append(self.directions[i] * vel * RAD_S_PER_VEL_UNIT)
            msg.effort.append(cur * AMP_PER_CURRENT_UNIT)   # amps, not Nm
        self.state_pub.publish(msg)

    def close(self):
        # Torque stays ON so the arm doesn't fall when the node stops.
        # Call /arm/torque false first if you want it limp.
        self.port.closePort()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ArmDriver()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    
    except RuntimeError as e:
        print(f'[arm_driver] {e}')

    finally:
        
        if node is not None:
            node.close()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()