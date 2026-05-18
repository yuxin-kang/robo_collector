# Joint names are StepIt base joint names with the _joint/_cmd/_gain suffix removed.
string[29] joint_names

float32[29] joint_pos
float32[29] joint_vel
float32[29] joint_torque

float32[29] cmd_joint_pos
float32[29] cmd_joint_vel
float32[29] cmd_joint_torque

float32[29] kp
float32[29] kd
float32[29] desired_torque

string[] foot_names
float32[] foot_force
