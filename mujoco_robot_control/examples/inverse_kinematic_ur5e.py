import argparse
from pathlib import Path
import numpy as np
from mujoco_robot_control.ik_levenberg_marquardt import LevenbegMarquardtIK
from mujoco_robot_control.ik_quadratic_programming import QuadraticProgrammingIK
from mujoco_robot_control.robot_env import RobotEnv


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
        model_path=Path(args.urdf_path),
        unlocked_joint_name=unlcoked_joints,
    )

    if args.ik == "LevenbegMarquardtIK":
        ik = LevenbegMarquardtIK(env)
    elif args.ik == "QuadraticProgrammingIK":
        ik = QuadraticProgrammingIK(env)
    else:
        raise ValueError(f"Unknown IK method: {args.ik}")

    frame_dict = {"frame_name": "wrist_3_link", "frame_type": "body"}
    # frame_dict = {"frame_name": "end_effector", "frame_type": "site"}

    mocap_id = env.model.body("target").mocapid[0]
    env.set_timestep(0.005)

    i = 0
    while True:
        i += 1
        env.data.mocap_pos[mocap_id][:2] = circle(
            i * env.timestep, 0.1, 0.5, -0.2, 0.5
        )[:2]
        ee_reference_pose = np.concatenate(
            (env.data.mocap_pos[mocap_id], env.data.mocap_quat[mocap_id])
        )

        qpos, qvel = ik.calculate(
            ee_reference_pose,
            frame_dict["frame_name"],
            frame_type=frame_dict["frame_type"],
            number_of_iterations=1,
            only_position=False,
            weight=np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]),
        )

        env.set_state(qpos, qvel)
        env.forward_dynamics()
        env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="mujoco_robot_control/assets/universal_robots_ur5e/scene.xml",
        help="Path to the urdf file",
    )
    parser.add_argument(
        "--ik", type=str, default="QuadraticProgrammingIK", help="IK method"
    )

    args = parser.parse_args()
    simulate(args)
