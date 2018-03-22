# Indoor Monocular UGV driving system
Building up all-in-one driving system (lane detection, position calibration, obstacle avoidance) for indoor spaces using just one camera. 
# Approach
Main idea is overlay different cv systems so to have the better possible definition of the scene, meant only to discover/follow one path towards the path-end. The system is designed to correct directions by using VP (*vanishing point*). It would use:
- lane detection
- obstacle detection
to achive the better path possible. Each steps influences the VP definition, which defines the ROI (*Region of Interest*). So first VP cohordinates are expressed by lane search. If no lanes are found, UGV would assume it's on some rough path (or internal floor), so it would rather rely on *Texture Flow Estimation* [1] . Once we have VP, then we can define it as the vertex of one triangle whose other vertex are == (0, image Height), (image Width, image Height). Cutting the top part of it by using horizon line or camera tilt setup, we have the road extraction part as one trapezoid. 
This trapezoid is then used to look for obstacle detection, thus furthing changing VP cohordinates. 
