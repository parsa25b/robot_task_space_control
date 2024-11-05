def add_mujoco_section_text(urdf_file, output_file):
    # Load the URDF file as text
    with open(urdf_file, "r") as file:
        urdf_text = file.read()

    if "<mujoco>" in urdf_text:
        return

    # Define the MuJoCo block as a string with proper indentation
    mujoco_block = """
  <mujoco>
    <compiler angle="radian" meshdir="meshes/urdf/obj_combined_hulls" autolimits="true" balanceinertia="false" discardvisual="true" fusestatic="false" inertiafromgeom="false"/>
    <option integrator="implicitfast"/>
  </mujoco>"""

    # Find the position to insert the MuJoCo block
    robot_tag = "<robot"
    insertion_index = (
        urdf_text.find(robot_tag) + urdf_text[urdf_text.find(robot_tag) :].find(">") + 1
    )

    # Insert the MuJoCo block after the robot name
    modified_urdf_text = (
        urdf_text[:insertion_index] + mujoco_block + urdf_text[insertion_index:]
    )

    # Save the modified URDF file
    with open(output_file, "w") as file:
        file.write(modified_urdf_text)


if __name__ == "__main__":
    # Usage
    input_urdf = "src/mujoco_robot_control/assets/phoenix/urdfs/gpr-gen8.1/robot_obj_combined_hulls.urdf"
    output_urdf = "output_file.urdf"
    add_mujoco_section_text(input_urdf, output_urdf)
