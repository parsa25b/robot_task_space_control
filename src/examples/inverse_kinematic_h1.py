import argparse
from pathlib import Path
import numpy as np
import sys

# Add the desired path
path = "src"
if path not in sys.path:
    sys.path.append(path)
# from src.ik_gradient_descent import GradientDescentIK
from ik_levenberg_marquardt import LevenbegMarquardtIK
from ik_quadratic_programming import QuadraticProgrammingIK
from robot_env import RobotEnv


def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
    as a function of time t and frequency f."""
    x = r * np.cos(2 * np.pi * f * t) + h
    y = r * np.sin(2 * np.pi * f * t) + k
    z = 0.5
    return np.array([x, y, z])


def simulate(args):
    """
    Simulates the robot control loop.

    Args:
        args (object): The arguments object containing the simulation parameters.
    """

    unlocked_joints = [
        "torso",
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
    ]

    env = RobotEnv(
        model_path=Path("src/assets/" + args.robot_model + "/scene.xml"),
        unlocked_joint_name=unlocked_joints,
    )
    # ik = GradientDescentIK(env)
    # ik = LevenbegMarquardtIK(env)
    ik = QuadraticProgrammingIK(env)

    # End-effector we wish to control.
    frame_name = "left_elbow_link"
    # frame_name = "end_effector"
    mocap_id = env.model.body("target").mocapid[0]

    i = 0
    weight = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    while True:
        env.add_marker(
            env.data.body(env.body_name_to_id(frame_name)).xpos,
            label="left_elbow",
            marker_size=0.05,
        )

        env.data.mocap_pos[mocap_id][:2] = circle(i * env.timestep, 0.1, 0.1, 0.3, 0.5)[
            :2
        ]
        ee_reference_pose = np.concatenate(
            (env.data.mocap_pos[mocap_id], env.data.mocap_quat[mocap_id])
        )

        qpos = ik.calculate(ee_reference_pose, frame_name, weight * 1, type="body")
        env.set_qpos(qpos)
        env.forward_dynamics()

        env.render()
        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-model",
        type=str,
        default="unitree_h1",
        help="Name of the robot model",
    )
    args = parser.parse_args()
    simulate(args)
