from setuptools import setup, find_packages

setup(
    name="mujoco_robot_control",
    version="0.0.1",
    description="A mujoco-based robot control library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    author="Parsa Bakhshandeh",
    author_email="parsa7676@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "mujoco==3.1.5",
        "mujoco-python-viewer==0.1.4",
        "pandas",
        "scipy",
        "osqp",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",  # Adjust license as necessary
    ],
)
