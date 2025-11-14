import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np

# --- TUNING PARAMETERS ---
STEERING_GAIN_KP = 1.0  # turn aggresiveness
MAX_STEER = 0.35        # Maximum steering angle (radians, approx 20 degrees)
MAX_SPEED = 3.0         # Maximum speed the car is allowed to go
# ---

class FollowerNode(Node):

    def __init__(self):
        super().__init__('follower_node')
        
        # Internal state
        self.self_pose = None
        self.self_orientation_q = None

        # This is the topic our car's controller will listen to
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # need to know own position to calculate how to follow
        self.self_odom_subscription = self.create_subscription(
            Odometry,
            '/pf/pose/odom',  # Subscribes to OWN odometry
            self.self_odom_callback,
            10)

        # subscribes to the opponent's odometry
        self.opponent_subscription = self.create_subscription(
            Odometry,
            'opponent_data', 
            self.opponent_odom_callback,
            10)
            
        self.get_logger().info('Follower Node is running: Listening for opponent and own odom...')

    def self_odom_callback(self, msg):
        """ This function is called every time we receive own odometry data. """
        self.self_pose = msg.pose.pose.position
        self.self_orientation_q = msg.pose.pose.orientation

    def opponent_odom_callback(self, msg):
        """ This function is called every time we receive the OPPONENT'S odometry data. """
        
        # --- 1. Log the data (as requested) ---
        opp_pos = msg.pose.pose.position
        opp_vel = msg.twist.twist.linear
        self.get_logger().info(f'HEARD Opponent: Pos=({opp_pos.x:.2f}, {opp_pos.y:.2f}) | Speed=({opp_vel.x:.2f})')

        # --- 2. Safety Check ---
        # If we don't know where WE are, we can't follow.
        if self.self_pose is None or self.self_orientation_q is None:
            self.get_logger().warn("Waiting for own odometry data... cannot follow.")
            return

        # --- 3. The "Follow" Algorithm (Proportional Control) ---
        
        # Convert our own orientation from quaternion to a simple yaw angle
        self_yaw = self.quaternion_to_yaw(self.self_orientation_q)
        
        # X and Y distance to the opponent
        dx = opp_pos.x - self.self_pose.x
        dy = opp_pos.y - self.self_pose.y
        
        # calc the angle from our car to the opponent
        target_angle = np.arctan2(dy, dx)
        
        # calc the error: the difference between our heading and the angle to the target
        alpha = target_angle - self_yaw
        
        # "Wrap" the angle to be between -pi and +pi (to handle crossing 360 degrees)
        if alpha > np.pi:
            alpha -= 2.0 * np.pi
        if alpha < -np.pi:
            alpha += 2.0 * np.pi

        # --- 4. Calculate Drive Command ---
        
        # Steering: Proportional to the error angle.
        steering_angle = STEERING_GAIN_KP * alpha
        
        # Speed: match the opponent's speed.
        speed = opp_vel.x
        
        # Clamp the vals to safe vals
        steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
        speed = np.clip(speed, 0.0, MAX_SPEED)
        
        # --- 5. Publish the command ---
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = steering_angle
        
        self.drive_publisher.publish(drive_msg)

    def quaternion_to_yaw(self, q):
        """ Helper function to convert a ROS2 Quaternion to a Yaw angle (in radians). """
        # sin(yaw)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        # cos(yaw)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    follower_node = FollowerNode()
    rclpy.spin(follower_node)
    
    follower_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()