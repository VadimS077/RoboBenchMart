import argparse
import json
import numpy as np
import gymnasium as gym
import trimesh
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
import sapien

import sys
sys.path.append('.')
from dsynth.envs import *
from dsynth.robots import *

@register_env("BuggyPush", max_episode_steps=50)
class BuggyPush(PushCubeEnv):
    def _load_scene(self, options: dict):
        super()._load_scene(options)
       
        # load ycb object
        model_id='002_master_chef_can'
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{model_id}",
        )
        builder.set_initial_pose(sapien.Pose(p = [0.2, 0.2, 0.071]))
        self.can = builder.build(name=model_id)

        # hide goal_region
        self._hidden_objects.append(self.goal_region)



def save_pointcloud_from_maniskill(env_id="DarkstoreContinuousBaseEnv",robot_uids="ds_fetch_basket",scene_dir="",num_envs=1, out_path="scene.json", cam_width=1024, cam_height=1024, selected_id_arg="[ENV#0]_food.BEER.DuffBeerCan:0:1:0:0", shader='default', gui=True):
    """Сохраняет pointcloud из ManiSkill среды в JSON формате, совместимом с GraspGen"""
    sensor_configs = dict()
    sensor_configs["width"] = cam_width
    sensor_configs["height"] = cam_height

    #env = BuggyPush(obs_mode="pointcloud",reward_mode="none", sensor_configs=sensor_configs)
    parallel_in_single_scene = num_envs > 1 and gui
    env = gym.make(env_id, 
                   robot_uids=robot_uids, 
                   config_dir_path=scene_dir,
                   num_envs=num_envs, 
                   viewer_camera_configs={'shader_pack': shader}, 
                   human_render_camera_configs={'shader_pack': shader},
                   render_mode="human" if gui else "rgb_array", 
                   control_mode=None,
                   enable_shadow=True,
                   sim_config={'spacing': 20},
                   obs_mode="pointcloud",
                   sim_backend='auto',
                   parallel_in_single_scene=parallel_in_single_scene,
                   sensor_configs=sensor_configs
                   )

    obs, _ = env.reset(seed=0,options={'reconfigure': True})
    # Получаем облако точек и цвета
    xyz = obs["pointcloud"]["xyzw"][0, ..., :3].cpu().numpy()

    rgb = obs["pointcloud"]["rgb"][0].cpu().numpy()
    #rgb = (rgb.astype(np.float32) / 255.0).round(4)

    seg_raw = obs["pointcloud"]["segmentation"][0].cpu().numpy()

    # Определяем selected_id (int) на основе аргумента selected_id_arg
    selected_id = None
    if selected_id_arg is not None:
        # если передали цифру
        if isinstance(selected_id_arg, str) and selected_id_arg.isdigit():
            selected_id = int(selected_id_arg)
        elif isinstance(selected_id_arg, int):
            selected_id = selected_id_arg
        else:
            # попробуем найти имя в карте env.unwrapped.segmentation_id_map
            seg_map = getattr(env.unwrapped, "segmentation_id_map", None)
            
            if seg_map is not None:
                reverse_name_to_id = {obj.name: obj_id for obj_id, obj in seg_map.items()}
                print(reverse_name_to_id)
                if selected_id_arg in reverse_name_to_id:
                    selected_id = reverse_name_to_id[selected_id_arg]
                    
                else:
                    # попытка: если передали Link/Actor имя с префиксом
                    for obj_id, obj in seg_map.items():
                        try:
                           
                            if selected_id_arg in obj.name:
                                selected_id = obj_id
                                break
                        except Exception:
                            continue

    if seg_raw is None:
        # Нет сегментации — object == scene
        mask_obj = np.ones(len(xyz), dtype=bool)
    else:
        mask_obj = (seg_raw == selected_id) if selected_id is not None else np.ones(len(xyz), dtype=bool)

    # Применяем маску
    mask_obj = mask_obj.reshape(-1)
    xyz_obj = xyz[mask_obj]
    rgb_obj = rgb[mask_obj]
    
    data = {
        "object_info": {
            "pc": xyz_obj.tolist(),
            "pc_color": rgb_obj.tolist(),
        },
        "scene_info": {
            "pc_color": xyz.tolist(),
            "img_color": rgb.tolist(),
        },
        "grasp_info": {
            "grasp_poses": [],
            "grasp_conf": [],
        },
    }

    with open(out_path, "w") as f:
        json.dump(data, f)

    print(f"✅ Saved ManiSkill scene from {env_id}")
    print(f"📦 File: {out_path}")
    print(f"🟢 Points: {len(xyz)}")
    env.close()


def view_pointcloud_from_json(json_path):
    """Открывает сохранённый JSON-файл и визуализирует облака точек объекта и сцены"""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Проверяем наличие необходимых данных
    if "object_info" not in data or "pc" not in data["object_info"]:
        raise ValueError(f"{json_path} does not contain valid object pointcloud data!")
    
    if "scene_info" not in data or "pc_color" not in data["scene_info"]:
        raise ValueError(f"{json_path} does not contain valid scene pointcloud data!")

    # Загружаем данные объекта
    obj_xyz = np.array(data["object_info"]["pc"], dtype=np.float32)
    obj_rgb = np.array(data["object_info"]["pc_color"], dtype=np.float32)

    # Загружаем данные сцены
    scene_xyz = np.array(data["scene_info"]["pc_color"], dtype=np.float32)
    scene_rgb = np.array(data["scene_info"]["img_color"], dtype=np.float32)

    print(f"📂 Loaded scene: {json_path}")
    print(f"📊 Object points: {len(obj_xyz)}")
    print(f"📊 Scene points: {len(scene_xyz)}")

    # Создаем облака точек
    obj_pcd = trimesh.points.PointCloud(vertices=obj_xyz, colors=obj_rgb)
    scene_pcd = trimesh.points.PointCloud(vertices=scene_xyz, colors=scene_rgb)

    # Визуализируем отдельно объект и сцену
    print("🔵 Showing OBJECT point cloud...")
    trimesh.Scene([obj_pcd]).show()

    print("🟢 Showing SCENE point cloud...") 
    trimesh.Scene([scene_pcd]).show()

    # Дополнительно: показываем объединенную сцену с выделенным объектом
    print("🌈 Showing COMBINED scene (object in red)...")
    # Делаем копию сцены и выделяем объект красным цветом
    red_obj_rgb = np.ones_like(obj_rgb) * np.array([1.0, 0.0, 0.0])  # объект полностью красный

    combined_xyz = np.vstack([scene_xyz, obj_xyz])
    combined_rgb = np.vstack([scene_rgb, red_obj_rgb])

    combined_pcd = trimesh.points.PointCloud(vertices=combined_xyz, colors=combined_rgb)
    trimesh.Scene([combined_pcd]).show()

def parse_args():
    parser = argparse.ArgumentParser(description="ManiSkill scene save/view tool")
    parser.add_argument(
        "--mode",
        choices=["save", "view"],
        required=True,
        help="Mode: 'save' to capture ManiSkill scene, 'view' to open JSON scene",
    )
    parser.add_argument("scene_dir", help="Путь к директории с JSON конфигом сцены")
    parser.add_argument(
        "--env-id",
        type=str,
        default="DarkstoreContinuousBaseEnv",   
        help="Environment ID for ManiSkill (used in save mode)",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="scene.json",
        help="Output JSON path (used in save mode)",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="scene.json",
        help="Input JSON path (used in view mode)",
    )
    parser.add_argument(
        "--cam-width",
        type=int,
        default=1024,  # Высокое разрешение по умолчанию
        help="Camera width for high resolution point cloud",
    )
    parser.add_argument(
        "--cam-height", 
        type=int,
        default=1024,  # Высокое разрешение по умолчанию
        help="Camera height for high resolution point cloud",
    )
    parser.add_argument("-r", "--robot-uids", type=str, default="ds_fetch_basket", help=f"Robot id")
    parser.add_argument("-s", "--seed", type=int, nargs='+', default=42)
    parser.add_argument('--shader',
                        default='default',
                        const='default',
                        nargs='?',
                        choices=['rt', 'rt-fast', 'rt-med', 'default', 'minimal'],)
    parser.add_argument('--gui',
                        action='store_true',
                        default=True)
    parser.add_argument('--episode_length', type=int, default=10)
    parser.add_argument('--video',
                        action='store_true',
                        default=False)
    parser.add_argument(
        "--selected-id",
        type=str,
        default="[ENV#0]_food.CRACKERS_COOKIES.OreoLemonCremeSandwichCookies:0:1:0:0",#"[ENV#0]_food.HOUSEHOLD.VanishStainRemover:0:3:1:0",
        #default='[ENV#0]_food.ENERGY_DRINKS.MonsterEnergyDrink:0:0:2:0',
        help="Selected object ID for segmentation (used in save mode)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "save":
        save_pointcloud_from_maniskill(
            env_id=args.env_id,
            robot_uids=args.robot_uids,
            scene_dir=Path(args.scene_dir),
            num_envs=args.num_envs,
            out_path=args.out_path,
            cam_width=args.cam_width,
            cam_height=args.cam_height,
            selected_id_arg=args.selected_id,
            shader=args.shader,
            gui=args.gui
        )
    elif args.mode == "view":
        view_pointcloud_from_json(args.json_path)


if __name__ == "__main__":
    #usage python main.py save /path/to/scene/dir --out-path output.json --env-id DarkstoreContinuousBaseEnv
    main()