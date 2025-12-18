//
// Created by srira on 12/10/2025.
//

#ifndef FOLLOWTHEGAP_H
#define FOLLOWTHEGAP_H
#include "Parameters.h"


class FollowTheGap {

public:
    FollowTheGap();

    // callbacks
    void lidar_callback();
    void odom_callback();

    // core ftg logic
    void preprocess_lidar();
    void mask_safety_bubble();




};



#endif //FOLLOWTHEGAP_H
