import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class OdomSender(Node):

    def __init__(self):
        super().__init__('odom_sender')
        
        # TOPIC TO PUBLISH TO
        # publishes to the shared topic that the other car will listen to (opponent_data)
        self.publisher_ = self.create_publisher(Odometry, 'opponent_data', 10)
        
        # TOPIC TO LISTEN TO
        # subscribes to OWN odometry topic.
        self.subscription = self.create_subscription(
            Odometry,
            '/pf/pose/odom',  # <-- odom
            self.odom_callback,
            10)
            
        self.get_logger().info('Odom Sender is running. Forwarding /pf/pose/odom to /opponent_data...')

    def odom_callback(self, msg):
        """Function called when a message is received from odom, send to opponent"""

        # republish to /opponent_data topic
        self.publisher_.publish(msg)

        # get the message to print to our own console
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear

        # log to console
        self.get_logger().info(f'Sending Data: Pos=({pos.x:.2f}, {pos.y:.2f}) | Speed=({vel.x:.2f})')


def main(args=None):
    rclpy.init(args=args)
    odom_sender = OdomSender()
    rclpy.spin(odom_sender)
    
    odom_sender.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()