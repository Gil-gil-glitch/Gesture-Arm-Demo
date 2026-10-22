# gesture-arm-demo

A ROS 2 Humble demo where hand gestures (detected with MediaPipe) control a
4-DOF + gripper DYNAMIXEL arm (OpenManipulator-X style, XM430-W350-R motors,
U2D2 over RS-485, powered from the TurtleBot 2 / Kobuki 12V/5A port).

## Hardware

| Motor ID | Joint       |
|---------:|-------------|
| 1        | joint1 (base) |
| 2        | joint2 (shoulder) |
| 3        | joint3 (elbow) |
| 4        | joint4 (wrist) |
| 5        | gripper     |

Baud rate 1,000,000, Protocol 2.0.

Power: Kobuki 12V/5A -> power board -> motors. Data: U2D2 RS-485 port -> power board.
Turn the Kobuki on before launching.

## Setup (once)

```bash
sudo apt install ros-humble-dynamixel-sdk
sudo usermod -aG dialout $USER          # then log out and back in
# Faster USB (do this every time the U2D2 is plugged in, or add a udev rule):
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

## Build

```bash
cd gesture-arm-demo
colcon build --symlink-install
source install/setup.bash
```

## Calibrate (once)

Close DYNAMIXEL Wizard first (only one program can use the U2D2).

```bash
python3 src/arm_driver/tools/calibrate.py
```

Copy the values it prints into `src/arm_driver/config/arm.yaml`
(`zero_ticks`, `gripper_open_ticks`, `gripper_closed_ticks`), and flip
`directions` for any joint that moves the wrong way.

## Run the driver

```bash
ros2 launch arm_driver arm.launch.py
```

## Try it from the terminal

```bash
# Watch joint angles
ros2 topic echo /joint_states

# Move the arm: [base, shoulder, elbow, wrist] in radians
ros2 topic pub --once /arm/joint_targets std_msgs/msg/Float64MultiArray "{data: [0.5, -0.5, 0.3, 0.5]}"

# Gripper: 0.0 = open, 1.0 = closed
ros2 topic pub --once /arm/gripper std_msgs/msg/Float64 "{data: 1.0}"

# Go home
ros2 service call /arm/home std_srvs/srv/Trigger

# Torque off (arm goes limp - hold it!) / on
ros2 service call /arm/torque std_srvs/srv/SetBool "{data: false}"
ros2 service call /arm/torque std_srvs/srv/SetBool "{data: true}"
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Could not open /dev/ttyUSB0` | U2D2 unplugged, wrong port, not in `dialout`, or DYNAMIXEL Wizard still open |
| `No response from motor ID(s)` | Kobuki off, loose RS-485 cable, wrong baud rate |
| Sync read failures during fast moves | 12 V supply sagging; lower `profile_velocity` |
| Joint moves the wrong way | Flip its sign in `directions` |
| Gripper doesn't move | `gripper_open_ticks` / `gripper_closed_ticks` not calibrated |
| Gripper squeezes too hard | Lower `gripper_goal_current` |