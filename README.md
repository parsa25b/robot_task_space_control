# Robo Motion Control

A repository for various robot motion control.

## Installation

To install the required packages, you can use the following command:

```bash
pip install -r requirements.txt
```

Ensure you have Python and pip installed on your system. The `requirements.txt` file should contain all the dependencies needed to run the project.

## Running the Code

To run the code, use the following command:

```bash
python simulate.py --robot-model <robot_model_name>
```

The `--robot-model` argument is optional and defaults to `"universal_robots_ur5e"`.

For example, to run the script with the default robot model, use:

```bash
python robot_control.py
```

Or to specify a different robot model:

```bash
python robot_control.py --robot-model h1
```

## Video Demo

A video demonstration of the UR5e robot end effector control can be found here:
![UR5e Task Space Control](https://github.com/parsa25b/robo_motion_control/blob/main/images/ur5e_task_space_control.gif)

A video demonstration of the h1 right arm task space control can be found here:
![h1 right arm Task Space Control](https://github.com/parsa25b/robo_motion_control/blob/main/images/ur5e_task_space_control.gif)

## Acknowledgments

Special thanks to the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main) for the robot models.

## References

- Khatib, O., 1987. A unified approach for motion and force control of robot manipulators: The operational space formulation. IEEE Journal on Robotics and Automation, 3(1), pp.43-53.
- [kevinzakka mjctrl](https://github.com/kevinzakka/mjctrl/tree/main) 
