# GraspGen Gripper Description

version: `v.1.0.0`

A gripper is specified by its unique name "gripper_name", which is an argument in the training script as well as the basename for the config files.

Each gripper need to have three assets defined
1) YAML Config `{gripper_name}.yaml`
2) Python Config `{gripper_name}.py`
3) URDF for just the gripper

### YAML Config 

`{gripper_name}.yaml` has the following key variables used for data generation and training:


- `width`: Distance between the fingers at its max joint angle in the open configuration
- `depth`: Extents of the gripper along the z-axis, from base link frame to TCP frame
- `transform_offset_from_asset_to_graspgen_convention`: Expressed as a list of translation and quaternion e.g. `[[xyz],[xyzw]]`

Our gripper definition follows the following convention shown below. The approach direction is the positive Z-axis while the gripper finger closing direction is along the X-axis. 

<img src="../fig/graspgen_coordinate_convention.png" width="380" height="250" title="doc1">

If your gripper has a different convention, please use the `transform_offset_from_asset_to_graspgen_convention` variable to bring your gripper to this convention. The original URDF does not need to be modified.

Examples of the frames overlaid on the 3 gripper models we provided in the dataset (Suction, Robotiq 2f-140 and Franka-Panda) are shown below:

<img src="../fig/graspgen_coordinate_convention_examples.png" width="500" height="250" title="doc2">

### Python Config
The following variables and functions need to be defined in the python script `{gripper_name}.py`:

- class `GripperModel`: Specifies how the collision and visual meshes of the gripper can be loaded from the gripper URDF
- func `load_control_points_for_visualization`: Loads the control points needed for visualization in meshcat
— func `load_control_points`: Loads the control points needed for applying training metrics


Optional:
get_transform_from_base_link_to_tool_tcp: 
If this function is not specified, this is assumed to be a fixed positive Z-offset based on the `depth` value.

If you have feedback about this convetion, please add a GitHub issue!