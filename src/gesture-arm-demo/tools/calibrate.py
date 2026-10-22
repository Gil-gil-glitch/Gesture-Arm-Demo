#!/usr/bin/env python3
"""
Calibration helper. Turns torque OFF and prints raw motor positions so you can
move the arm by hand and read off the values for config/arm.yaml.

  python3 calibrate.py --port /dev/ttyUSB0

1. Pose the arm at its zero position (stock OpenManipulator-X: pointing straight
   up) and copy joint 1-4 values into zero_ticks.
2. Open the gripper fully -> gripper_open_ticks. Close it -> gripper_closed_ticks.
3. Move each joint to its safe extremes to choose joint_min / joint_max.

Ctrl+C to quit. Hold the arm while it runs: with torque off it will fall!
Do not run this while arm_driver is running (they would share the port).
"""

import argparse
import math
import time

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='/dev/ttyUSB0')
    ap.add_argument('--baud', type=int, default=1000000)
    ap.add_argument('--ids', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    port, packet = PortHandler(args.port), PacketHandler(2.0)
    if not port.openPort() or not port.setBaudRate(args.baud):
        raise SystemExit(f'Could not open {args.port} at {args.baud}')

    input('Torque will turn OFF and the arm will go limp. Support it, then press Enter...')
    for i in args.ids:
        packet.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 0)

    try:
        while True:
            parts = []
            for i in args.ids:
                val, res, _ = packet.read4ByteTxRx(port, i, ADDR_PRESENT_POSITION)
                if res != COMM_SUCCESS:
                    parts.append(f'ID{i}: ----')
                    continue
                if val >= 1 << 31:
                    val -= 1 << 32
                deg = (val - 2048) * 360.0 / 4096.0
                parts.append(f'ID{i}: {val:5d} ({deg:+6.1f} deg)')
            print('  '.join(parts), end='\r', flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        port.closePort()


if __name__ == '__main__':
    main()