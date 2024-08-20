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

    unlcoked_joints = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    env = RobotEnv(
        model_path=Path("src/assets/" + args.robot_model + "/scene.xml"),
        unlocked_joint_name=unlcoked_joints,
    )

    # ik = GradientDescentIK(env)
    ik = LevenbegMarquardtIK(env)

    # End-effector we wish to control.
    # frame_name = "wrist_3_link"
    frame_name = "end_effector"
    mocap_id = env.model.body("target").mocapid[0]

    i = 0
    while True:
        env.data.mocap_pos[mocap_id][:2] = circle(i * env.timestep, 0.1, 0.5, -0.2, 1)[
            :2
        ]
        ee_reference_pose = np.concatenate(
            (env.data.mocap_pos[mocap_id], env.data.mocap_quat[mocap_id])
        )

        qpos = ik.calculate(ee_reference_pose, frame_name, type="site")
        env.set_qpos(qpos)
        env.forward_dynamics()

        env.render()
        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-model",
        type=str,
        default="universal_robots_ur5e",
        help="Name of the robot model",
    )
    args = parser.parse_args()
    simulate(args)
