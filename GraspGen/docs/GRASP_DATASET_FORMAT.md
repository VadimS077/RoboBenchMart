# GraspGen Gripper Description

version: `v.1.0.0`

GraspGen expects the dataset to be in the following format.

1. **Splits**: The objects are separated into the training and validation/test sets. Each line in the `*.txt` file can be a uuid (in the case of Objaverse) or a relative path to the object mesh (obj/stl) file (relative to the root of the object dataset). If you are using the same object for both training and testing, include them in both lists.
```
path/to/splits/
    train.txt
    valid.txt
```

2. **Grasp Dataset**: The grasps are specified in separate directory
```
path/to/grasp/data/
    *.json
```

Each json file inside grasp dataset has following information:
```
{
    "object": {
        "file": # relative path to object asset in the object dataset
        "scale": # scale for the object mesh at which the grasps were sampled and evaluated
    }, 
    "grasps": {
        "transforms": # 4x4 homogeous transformation matrix of the base link of gripper
        "object_in_gripper": # mask to distinguish successful vs. unsuccessful grasps
    }
}
```

The json file can be loaded in python as follows:
```
import json
import numpy as np
grasps_dict = json.load(open("/path/to/json/file", "r"))
object_file = grasps_dict["object"]["file"]
object_scale = grasps_dict["object"]["scale"]
grasps = np.array(grasps_dict["grasps"]["transforms"])
grasp_mask = np.array(grasps_dict["grasps"]["object_in_gripper"])
positive_grasps = grasps[grasp_mask]
negative_grasps = grasps[~grasp_mask]
```
