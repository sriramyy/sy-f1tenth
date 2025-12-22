#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import json
import time
import numpy as np

from .genetic_ml import GeneticML

class GeneticSupervisor(Node):
    def __init__(self):
        super().__init__("genetic_supervisor")
        
        # init Genetic Manager
        self.ga = GeneticML(population_size=20, generations=50)
        
        # Publishers
        self.gene_pub = self.create_publisher(String, "/genetic_genes", 10)
        self.reset_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        
        # Subscribers (To detect crashes/lap completion)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        # Listen for crashes
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)

        # State Variables
        self.start_time = time.time()
        self.is_running = False
        self.max_lap_time = 40.0 # Seconds before we kill a slow run
        
        self.get_logger().info("Genetic Supervisor Initiated. Starting...")
        
        # Give ROS a moment to connect, then start
        self.create_timer(2.0, self.start_first_run_callback)
        self.timer_started = False

    def start_first_run_callback(self):
        """simple oneshot timer to start loop"""
        if not self.timer_started:
            self.start_next_run()
            self.timer_started = True

    def start_next_run(self):
        """loads the next genome and begins tracking"""
        if self.ga.is_generation_complete():
            self.ga.evolve_next_generation()
            
        # get params for next car
        params = self.ga.get_next_genome_params()
        
        # convert params object to Dictionary -> JSON
        param_dict = vars(params) 
        msg = String()
        msg.data = json.dumps(param_dict)
        
        # publish new genes to the car
        self.gene_pub.publish(msg)
        
        # reset simulator
        self.reset_simulation()

        # reset trackers
        self.start_time = time.time()
        self.is_running = True

        idx = self.ga.current_genome_index
        gen = self.ga.current_generation
        
        print(f"Running Genome {idx} (Gen {gen})")

    def reset_simulation(self):
        """teleports car back to (0,0) in rviz"""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        # --- START LINE COORDINATES ---
        # ADJUST IF MAP START ISNT (0,0)
        msg.pose.pose.position.x = 0.0 
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = 0.0
        
        # Orientation: (0,0,0,1) = Facing East (Positive X)
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        self.reset_pub.publish(msg)
        time.sleep(0.1)

    def scan_callback(self, msg):
        """Detects crashes via LiDAR data."""
        if not self.is_running: return

        # Convert to numpy
        ranges = np.array(msg.ranges)
        
        # Filter out invalid values (inf/nan)
        valid_ranges = ranges[np.isfinite(ranges)]
        
        if valid_ranges.size == 0: return

        min_dist = np.min(valid_ranges)
        
        # CRASH THRESHOLD: 0.15 meters (15cm)
        if min_dist < 0.15: 
            self.get_logger().warn(f"CRASH! Min Dist: {min_dist:.3f}m")
            
            # Calculate time driven
            run_time = time.time() - self.start_time
            
            # Report failure
            self.finish_run(run_time, crash_count=1)

    def odom_callback(self, msg):
        """Checks for timeouts (or lap completion w/ waypoints)."""
        if not self.is_running: return
        
        current_time = time.time() - self.start_time
        
        # TIMEOUT CHECK
        if current_time > self.max_lap_time:
            self.get_logger().warn("TIMEOUT! Car too slow.")
            self.finish_run(current_time, crash_count=0) # No crash, just slow

    def finish_run(self, lap_time, crash_count):
        """finalizes the current car's attempt."""
        self.is_running = False
        
        # Send data to Manager
        self.ga.report_lap_result(lap_time, crash_count)
        
        # Loop immediately to next
        self.start_next_run()
        

def main():
    rclpy.init()
    node = GeneticSupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()