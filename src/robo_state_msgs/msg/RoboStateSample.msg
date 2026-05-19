std_msgs/Header header

robo_state_msgs/PolicyState policy_state
robo_state_msgs/RobotLowState robot_state
sensor_msgs/Imu imu

float32[29] target_joint_pos
float32[45] aligned_target_pos
float32[29] action
float32[1545] stepit_observation
float32 observation_l2_error

string[] missing_optional_fields
