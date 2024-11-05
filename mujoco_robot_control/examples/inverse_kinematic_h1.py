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

    unlocked_joints = [
        # "torso",
        # "left_shoulder_pitch",
        # "left_shoulder_roll",
        # "left_shoulder_yaw",
        # "left_elbow",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
    ]

    env = RobotEnv(
        model_path=Path(args.urdf_path),
        unlocked_joint_name=unlocked_joints,
    )

    if args.ik == "LevenbegMarquardtIK":
        ik = LevenbegMarquardtIK(env)
    elif args.ik == "QuadraticProgrammingIK":
        ik = QuadraticProgrammingIK(env)
    else:
        raise ValueError(f"Unknown IK method: {args.ik}")

    # End-effector we wish to control.
    frame_dict = {"frame_name": "end_effector", "frame_type": "site"}
    # frame_id = env.body_name_to_id(frame_dict["frame_name"])

    mocap_id = env.model.body("target").mocapid[0]
    env.set_timestep(0.005)

    weight = np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])

    i = 0
    while True:
        i += 1
        env.data.mocap_pos[mocap_id][:2] = circle(
            i * env.timestep, 0.1, 0.4, -0.3, 0.5
        )[:2]

        env.add_marker(
            getattr(env.data, frame_dict["frame_type"])(frame_dict["frame_name"]).xpos,
            label=frame_dict["frame_name"],
            marker_size=0.05,
        )

        ee_reference_pose = np.concatenate(
            (env.data.mocap_pos[mocap_id], env.data.mocap_quat[mocap_id])
        )

        # Inverse Kinematics calculations
        qpos, qvel = ik.calculate(
            ee_reference_pose,
            frame_name=frame_dict["frame_name"],
            weight=weight,
            frame_type=frame_dict["frame_type"],
            only_position=True,
        )

        env.set_state(qpos=qpos, qvel=qvel)
        env.forward_dynamics()
        env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="mujoco_robot_control/assets/unitree_h1/scene.xml",
        help="Path to the urdf file",
    )
    parser.add_argument(
        "--ik",
        type=str,
        default="QuadraticProgrammingIK",
        help="IK methods",
    )
    args = parser.parse_args()
    simulate(args)
