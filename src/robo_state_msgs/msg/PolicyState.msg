# Training policy input fields published by StepIt.
float32[90] relative_ori_6d
float32[45] motion_anchor_lin_vel_b
float32[45] motion_anchor_ang_vel_b
float32[30] ang_vel_history
float32[30] gravity_history
float32[290] joint_pos_rel_history
float32[290] joint_vel_history
float32[290] action_history

# Concatenation of the fields above in message order.
float32[1110] flattened
