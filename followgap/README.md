## Follow Gap Package
Allows a car to follow the gap in any general circuit. Does not require any raceline implementation or knowledge of the track.
\
\
**Uses:**
\
`ros2 run gt_follow_gap gt_follow_gap` - Standard follow the gap package
\
\
`ros2 run gt_follow_gap overtake` - Overtaking mode for follow the gap, much faster and assumes opponent is at constant speed.
\
**Overtaking Packages:**
* `overtake1` - original overtaking package
* `overtake2` - updated overtaking package
* `overtake3` - most recently used overtaking package