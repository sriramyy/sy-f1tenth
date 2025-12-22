//
// Created by srira on 12/10/2025.
//

#ifndef PARAMETER_H
#define PARAMETER_H
#include <iostream>
#include <string>


class Parameters {
public:
  // Follow the Gap
  int bubbleRadius = 100;          // size of safety bubble
  int preprocessConvSize = 3;      // smoothing window for raw lidar
  int bestPointConvSize = 120;     // smoothing window for finding best target
  float maxLidarDist = 7.0f;       // max lookahead distance
  float maxSteerAbs = 0.7f;        // max steering angle (rad) ~40deg

  // Speed
  float straightSpeed = 5.0f;
  float cornerSpeed = 2.0f;
  float maxSpeed = 8.0f;

  // Handling
  float centerBiasAlpha = 0.35f;   // bias to stay in center
  float edgeGuardDeg = 12.0f;      // dont aim at the very edge of a gap
  float fwdWedgeDeg = 8.0f;        // width of cone to check for forward collisions
  float steerSmoothAlpha = 0.5f;   // low pass filter on steering
  float steerRateLimit = 0.2f;     // max change in steering per update

  // returns a formatted string with all parameters
  std::string formatParams();
};



#endif //PARAMETER_H
