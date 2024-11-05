# Robo Motion Control

A repository for various robot motion control.

## Installation

To install the required packages, you can use the following command:

```bash
pip install .
```

Ensure you have Python and pip installed on your system. The `requirements.txt` file should contain all the dependencies needed to run the project.

## Running the IK Example

To run the code, use the following command:

### H1 Robot:

```bash
python mujoco_robot_control/examples/inverse_kinematic_h1.py
```
### UR5e Robot:
```bash
python mujoco_robot_control/examples/inverse_kinematic_ur5e.py
```

## Video Demo

A video demonstration of the "UR5e" end effector control can be found here:

![UR5e Task Space Control](https://github.com/parsa25b/robo_motion_control/blob/main/images/ur5e_task_space_control.gif)

A video demonstration of the "h1" right arm task space control can be found here:

![h1 right arm Task Space Control](https://github.com/parsa25b/robot_task_space_control/blob/main/images/h1_right_arm_task_space_control.gif)

## Acknowledgments

Special thanks to the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main) for the robot models.

## References

- Khatib, O., 1987. A unified approach for motion and force control of robot manipulators: The operational space formulation. IEEE Journal on Robotics and Automation, 3(1), pp.43-53.
- [kevinzakka mjctrl](https://github.com/kevinzakka/mjctrl/tree/main) 
