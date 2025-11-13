import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class OdomReceiver(Node):

    def __init__(self):
        super().__init__('odom_receiver')
        
        # subscribes to the shared topic (opponent_data)
        self.subscription = self.create_subscription(
            Odometry,
            'opponent_data',
            self.listener_callback,
            10)
        self.get_logger().info('Odom Receiver is running and listening for opponent...')

    def listener_callback(self, msg):
        """Called everytime a message is received"""    

        # get the position and speed
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        
        # log it
        self.get_logger().info(f'HEARD Opponent: Pos=({pos.x:.2f}, {pos.y:.2f}) | Speed=({vel.x:.2f})')

def main(args=None):
    rclpy.init(args=args)
    odom_receiver = OdomReceiver()
    rclpy.spin(odom_receiver)
    
    odom_receiver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()