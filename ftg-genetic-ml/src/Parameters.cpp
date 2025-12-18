//
// Created by srira on 12/10/2025.
//

#include "Parameters.h"


std::string Parameters::formatParams() {
    std::string s = "Parameters: " + std::to_string(bubbleRadius) +
                        ", " + std::to_string(preprocessConvSize) +
                        ", " + std::to_string(bestPointConvSize) +
                        ", " + std::to_string(maxLidarDist) +
                        ", " + std::to_string(maxSteerAbs) +
                        ", " + std::to_string(straightSpeed);

    return s;
}
