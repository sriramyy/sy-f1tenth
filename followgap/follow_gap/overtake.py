#!/usr/bin/env python3
import math
from enum import Enum, auto
from typing import Optional, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose
from ackermann_msgs.msg import AckermannDriveStamped

import numpy as np


# Faster and shouldn't hug wall
class OvertakeState(Enum):
    FOLLOW = auto()
    PREPARE = auto()
    OVERTAKE = auto()
    RETURN = auto()


class TargetVehicle:
    def __init__(self, pose: Pose, vel: float):
        self.pose = pose
        self.vel = vel


class OvertakeFollowGap(Node):
    def __init__(self):
        super().__init__('overtake_follow_gap')

        # topics
        self.scan_topic = '/scan'
        self.odom_topic = '/odom'
        self.opps_topic = '/opponents'
        self.drive_topic = '/drive'

        # subs/pubs
        qos10 = QoSProfile(depth=10)
        qos20 = QoSProfile(depth=20)

        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.lidar_callback, qos10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos20)
        self.opps_sub = self.create_subscription(PoseArray, self.opps_topic, self.opponents_callback, qos10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, qos10)

        # -------- follow-gap knobs (tuned to reduce wall-hugging) --------
        self.BUBBLE_RADIUS = 120
        self.PREPROCESS_CONV_SIZE = 3
        self.BEST_POINT_CONV_SIZE = 120
        self.MAX_LIDAR_DIST = 6.0
        self.MAX_STEER_ABS = np.deg2rad(40.0)

        # faster baseline speeds
        self.STRAIGHT_SPEED = 7.5     # was 2.5
        self.CORNER_SPEED   = 4.0     # was 1.2
        self.SPEED_MAX      = 8.0

        # center bias (reduce wall hugging) + edge guard
        self.CENTER_BIAS_ALPHA   = 0.35   # 0=no bias, 1=all center
        self.EDGE_GUARD_DEG      = 12.0   # ignore edges within ±EDGE_GUARD_DEG of FOV limits

        # side repulsion (push away from closer wall)
        self.SIDE_REPULSION_GAIN = 0.28   # radians-equivalent -> index shift scaling

        # TTC-based braking (forward safety wedge)
        self.TTC_HARD_BRAKE = 0.55  # s
        self.TTC_SOFT_BRAKE = 0.9   # s
        self.FWD_WEDGE_DEG  = 8.0

        # steering smoothing / rate limit
        self.STEER_SMOOTH_ALPHA = 0.5     # 0=no smoothing, 1=no change
        self.STEER_RATE_LIMIT   = np.deg2rad(8.0)  # max rad change per callback

        # -------- overtaking knobs --------
        self.FOLLOW_TIME_GAP = 0.6
        self.MIN_SPEED_ADV   = 0.4
        self.MIN_CLEAR_DIST  = 2.5
        self.PASS_SIDE       = 'left'        # 'left' or 'right'
        self.PASS_BIAS_DEG   = 12.0
        self.RETURN_LATENCY  = 0.8
        self.PREPARE_TIMEOUT = 2.0

        self.state = OvertakeState.FOLLOW
        self.state_ts = self.now_sec()

        self.ego_pose: Optional[Pose] = None
        self.ego_speed: float = 0.0
        self.opponents: List[TargetVehicle] = []
        self.radians_per_elem: Optional[float] = None
        self.proc_latest: Optional[np.ndarray] = None  # pre-bubble, cropped
        self.prev_steer: float = 0.0

        self.get_logger().info("OvertakeFollowGap (ROS2) started.")

    # Helpers
    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    #  Callbacks
    def odom_callback(self, msg: Odometry):
        self.ego_pose = msg.pose.pose
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.ego_speed = float(np.hypot(vx, vy))

    def opponents_callback(self, msg: PoseArray):
        self.opponents = [TargetVehicle(p, 5.0) for p in msg.poses]  # ASSUMPTIONS HERE

    def lidar_callback(self, scan: LaserScan):
        if self.ego_pose is None:
            return

        ranges_full = np.array(scan.ranges, dtype=np.float32)
        proc = self.preprocess_lidar(ranges_full)
        self.proc_latest = proc.copy()  # store before the bubble mapping

        # state machine
        target = self.select_front_target()
        self.step_fsm(target)

        # follow-gap core
        closest_idx = int(np.argmin(proc))
        proc = self.mask_bubble(proc, closest_idx, self.BUBBLE_RADIUS)
        gap_start, gap_end = self.find_max_gap(proc)
        best_idx = self.find_best_point(gap_start, gap_end, proc)

        # 1) edge guard: avoid hugging FOV edges
        best_idx = self.apply_edge_guard(best_idx, proc.size)

        # 2) center bias (pull toward scan center)
        best_idx = self.apply_center_bias(best_idx, proc.size, self.CENTER_BIAS_ALPHA)

        # 3) overtaking bias during PREPARE/OVERTAKE
        if self.state in (OvertakeState.PREPARE, OvertakeState.OVERTAKE):
            bias_idx = self.index_bias(proc.size, np.deg2rad(self.PASS_BIAS_DEG))
            best_idx = int(np.clip(best_idx + bias_idx, 0, proc.size - 1))

        # 4) side repulsion away from walls
        repel_shift = self.side_repulsion_shift(proc)
        best_idx = int(np.clip(best_idx + repel_shift, 0, proc.size - 1))

        # steer + speed
        steer_raw = self.index_to_steer(best_idx, proc.size)
        steer_cmd = self.smooth_and_limit_steer(steer_raw)
        speed_cmd = self.speed_policy(steer_cmd)

        self.publish_drive(speed_cmd, steer_cmd)

    # FSM
    def step_fsm(self, target: Optional[TargetVehicle]):
        now = self.now_sec()
        t_in_state = now - self.state_ts

        if target is None:
            if self.state != OvertakeState.FOLLOW:
                self.set_state(OvertakeState.RETURN)
            return

        dist = self.longitudinal_gap(self.ego_pose, target.pose)
        rel_v = self.ego_speed - target.vel
        desired_gap = self.FOLLOW_TIME_GAP * max(self.ego_speed, 0.1)

        if self.state == OvertakeState.FOLLOW:
            if dist < desired_gap and rel_v < self.MIN_SPEED_ADV:
                self.set_state(OvertakeState.PREPARE)
        elif self.state == OvertakeState.PREPARE:
            if self.clear_on_pass_side():
                self.set_state(OvertakeState.OVERTAKE)
            elif t_in_state > self.PREPARE_TIMEOUT:
                self.set_state(OvertakeState.FOLLOW)
        elif self.state == OvertakeState.OVERTAKE:
            if self.passed_target(target):
                self.set_state(OvertakeState.RETURN)
        elif self.state == OvertakeState.RETURN:
            if t_in_state > self.RETURN_LATENCY:
                self.set_state(OvertakeState.FOLLOW)

    def set_state(self, s: OvertakeState):
        if s != self.state:
            self.state = s
            self.state_ts = self.now_sec()
            self.get_logger().info(f"State -> {self.state.name}")

    def select_front_target(self) -> Optional[TargetVehicle]:
        if self.ego_pose is None or not self.opponents:
            return None
        best = None
        best_s = float('inf')
        for o in self.opponents:
            s = self.longitudinal_gap(self.ego_pose, o.pose)
            if s > 0 and s < best_s:
                best, best_s = o, s
        return best

    def longitudinal_gap(self, ego: Pose, other: Pose) -> float:
        dx = other.position.x - ego.position.x
        dy = other.position.y - ego.position.y
        yaw = self.yaw_of(ego)
        return math.cos(yaw) * dx + math.sin(yaw) * dy

    def passed_target(self, target: TargetVehicle) -> bool:
        return self.longitudinal_gap(target.pose, self.ego_pose) > 1.5

    # LIDAR stuff
    def preprocess_lidar(self, ranges: np.ndarray) -> np.ndarray:
        n = len(ranges)
        self.radians_per_elem = (2.0 * np.pi) / n if n > 0 else None
        # keep a wide-ish front (±135°) but handle small scans gracefully
        proc = ranges[135:-135].copy() if n > 270 else ranges.copy()
        k = self.PREPROCESS_CONV_SIZE
        if k > 1 and proc.size > 0:
            kernel = np.ones(k, dtype=np.float32) / float(k)
            proc = np.convolve(proc, kernel, mode='same')
        if proc.size > 0:
            np.clip(proc, 0.0, self.MAX_LIDAR_DIST, out=proc)
        return proc

    def mask_bubble(self, arr: np.ndarray, center: int, radius: int) -> np.ndarray:
        a = arr.copy()
        if a.size == 0:
            return a
        lo = max(0, center - radius)
        hi = min(a.size, center + radius + 1)
        a[lo:hi] = 0.0
        return a

    def find_max_gap(self, a: np.ndarray):
        if a.size == 0:
            return 0, 0
        masked = np.ma.masked_where(a == 0.0, a)
        slices = np.ma.notmasked_contiguous(masked)
        if not slices:
            return 0, a.size
        best = max(slices, key=lambda sl: (sl.stop - sl.start))
        return best.start, best.stop

    def find_best_point(self, start: int, stop: int, a: np.ndarray) -> int:
        if a.size == 0:
            return 0
        window = self.BEST_POINT_CONV_SIZE
        if stop <= start + 1:
            return start
        segment = a[start:stop]
        if window > 1 and segment.size > 0:
            kernel = np.ones(window, dtype=np.float32) / float(window)
            smoothed = np.convolve(segment, kernel, mode='same')
        else:
            smoothed = segment
        return int(np.argmax(smoothed)) + start

    def apply_edge_guard(self, idx: int, length: int) -> int:
        if self.radians_per_elem is None or length == 0:
            return 0
        guard = int(round(np.deg2rad(self.EDGE_GUARD_DEG) / (self.radians_per_elem or 1e-6)))
        lo = max(0, guard)
        hi = max(0, length - guard - 1)
        return int(np.clip(idx, lo, hi))

    def apply_center_bias(self, idx: int, length: int, alpha: float) -> int:
        if length == 0:
            return 0
        center = (length - 1) / 2.0
        biased = (1.0 - alpha) * float(idx) + alpha * center
        return int(np.clip(round(biased), 0, length - 1))

    def side_repulsion_shift(self, proc: np.ndarray) -> int:
        if proc.size == 0 or self.radians_per_elem is None:
            return 0
        L = proc.size
        band = max(6, int(0.05 * L))   # small band near edges
        left_avg  = float(np.mean(proc[:band])) if band < L else float(proc[0])
        right_avg = float(np.mean(proc[-band:])) if band < L else float(proc[-1])
        diff = right_avg - left_avg
        per = (self.radians_per_elem or 1e-6)
        max_shift_idx = int(round(self.SIDE_REPULSION_GAIN / per))
        shift = int(np.clip(np.sign(diff) * min(abs(diff), 1.0) * max_shift_idx, -max_shift_idx, max_shift_idx))
        return shift

    def index_to_steer(self, idx: int, length: int) -> float:
        if self.radians_per_elem is None or length == 0:
            return 0.0
        lidar_angle = (idx - (length / 2.0)) * self.radians_per_elem
        steer = float(lidar_angle / 2.0)
        return float(np.clip(steer, -self.MAX_STEER_ABS, self.MAX_STEER_ABS))

    def smooth_and_limit_steer(self, steer: float) -> float:
        s = (1.0 - self.STEER_SMOOTH_ALPHA) * steer + self.STEER_SMOOTH_ALPHA * self.prev_steer
        delta = np.clip(s - self.prev_steer, -self.STEER_RATE_LIMIT, self.STEER_RATE_LIMIT)
        s_limited = self.prev_steer + float(delta)
        self.prev_steer = s_limited
        return s_limited

    def forward_clearance(self) -> float:
        if self.proc_latest is None or self.radians_per_elem is None or self.proc_latest.size == 0:
            return self.MAX_LIDAR_DIST
        a = self.proc_latest
        L = a.size
        center = L // 2
        half = int(round(np.deg2rad(self.FWD_WEDGE_DEG) / (self.radians_per_elem or 1e-6)))
        lo = max(0, center - half)
        hi = min(L, center + half + 1)
        if hi <= lo:
            return self.MAX_LIDAR_DIST
        return float(np.min(a[lo:hi]))

    def speed_policy(self, steer: float) -> float:
        base = self.CORNER_SPEED if abs(steer) > np.deg2rad(10.0) else self.STRAIGHT_SPEED
        base = float(np.clip(base, 0.0, self.SPEED_MAX))
        fwd = self.forward_clearance()
        v   = max(self.ego_speed, 0.05)
        ttc = fwd / v if v > 0.05 else 999.0
        if ttc < self.TTC_HARD_BRAKE:
            return 0.0
        elif ttc < self.TTC_SOFT_BRAKE:
            scale = (ttc - self.TTC_HARD_BRAKE) / (self.TTC_SOFT_BRAKE - self.TTC_HARD_BRAKE)
            return float(np.clip(scale * base, 0.0, base))
        if abs(steer) < np.deg2rad(6.0) and fwd > 3.0:
            base = min(self.SPEED_MAX, base + 0.5)
        return base

    def index_bias(self, arr_len: int, bias_rad: float) -> int:
        per = self.radians_per_elem or 1e-6
        shift = int(round(bias_rad / per))
        return -abs(shift) if self.PASS_SIDE == 'left' else abs(shift)

    def clear_on_pass_side(self) -> bool:
        if self.proc_latest is None or self.radians_per_elem is None or self.proc_latest.size == 0:
            return False
        a = self.proc_latest
        L = a.size
        center = L // 2
        wedge_center = np.deg2rad(15.0)
        wedge_half   = np.deg2rad(8.0)
        per = self.radians_per_elem
        off_idx  = int(round(wedge_center / per))
        half_idx = int(round(wedge_half   / per))
        cidx = center - off_idx if self.PASS_SIDE == 'left' else center + off_idx
        lo = max(0, cidx - half_idx)
        hi = min(L, cidx + half_idx + 1)
        if hi <= lo:
            return False
        wedge_min = float(np.min(a[lo:hi]))
        return wedge_min >= self.MIN_CLEAR_DIST

    def yaw_of(self, pose: Pose) -> float:
        q = pose.orientation
        return math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))

    # Output
    def publish_drive(self, speed: float, steer: float):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(np.clip(steer, -self.MAX_STEER_ABS, self.MAX_STEER_ABS))
        msg.drive.speed = float(max(0.0, speed))
        self.drive_pub.publish(msg)


def main():
    rclpy.init()
    node = OvertakeFollowGap()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

