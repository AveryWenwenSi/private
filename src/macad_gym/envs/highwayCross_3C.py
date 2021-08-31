#!/usr/bin/env python
import time

from macad_gym.carla.multi_env import MultiCarlaEnv

# from env.carla.multi_env import get_next_actions

# config_file = open("urban_2_car_1_ped.json")
# configs = json.load(config_file)

HC3C_CONFIGS = {
    "scenarios": "HC3C_TOWN4",
    "env": {
        "server_map": "/Game/Carla/Maps/Town04",
        "render": True,
        "render_x_res": 800,
        "render_y_res": 600,
        "x_res": 84,
        "y_res": 84,
        "framestack": 1,
        "discrete_actions": True,
        "squash_action_logits": False,
        "verbose": False,
        "use_depth_camera": False,
        "send_measurements": False,
        "enable_planner": False,
        "spectator_loc": [145, 34, 20],
        "sync_server": True,
        "fixed_delta_seconds": 0.05,
    },
    "actors": {
        "car1": {
            "type": "vehicle_4W",
            "enable_planner": False,
            "convert_images_to_video": False,
            "early_terminate_on_collision": True,
            "reward_function": "hiway_lane_change",
            "scenarios": "HC3C_TOWN4_CAR1",
            "manual_control": False,
            "auto_control": False,
            "camera_type": "rgb",
            "collision_sensor": "on",
            "lane_sensor": "off",
            "log_images": False,
            "log_measurements": False,
            "render": True,
            #"render_x_res": 800,
            #"render_y_res": 600,
            "x_res": 84,
            "y_res": 84,
            "use_depth_camera": False,
            "send_measurements": False,
        },
        "car2": {
            "type": "vehicle_4W",
            "enable_planner": False,
            "convert_images_to_video": False,
            "early_terminate_on_collision": True,
            "reward_function": "hiway_lane_change",
            "scenarios": "HC3C_TOWN4_CAR2",
            "manual_control": False,
            "auto_control": False,
            "camera_type": "rgb",
            "collision_sensor": "on",
            "lane_sensor": "off",
            "log_images": False,
            "log_measurements": False,
            "render": True,
            #"render_x_res": 800,
            #"render_y_res": 600,
            "x_res": 84,
            "y_res": 84,
            "use_depth_camera": False,
            "send_measurements": False,
        },
        "car3": {
            "type": "vehicle_4W",
            "enable_planner": False,
            "convert_images_to_video": False,
            "early_terminate_on_collision": True,
            "reward_function": "hiway_lane_change",
            "scenarios": "HC3C_TOWN4_CAR3",
            "manual_control": False,
            "auto_control": False,
            "camera_type": "rgb",
            "collision_sensor": "on",
            "lane_sensor": "off",
            "log_images": False,
            "log_measurements": False,
            "render": True,
            #"render_x_res": 800,
            #"render_y_res": 600,
            "x_res": 84,
            "y_res": 84,
            "use_depth_camera": False,
            "send_measurements": False,
        },
    },
}


class HighwayCross3Car(MultiCarlaEnv):
    """A 4-way signalized intersection Multi-Agent Carla-Gym environment"""
    def __init__(self):
        self.configs = HC3C_CONFIGS
        super(HighwayCross3Car, self).__init__(self.configs)


if __name__ == "__main__":
    env = HighwayCross3Car()
    configs = env.configs
    for ep in range(2):
        obs = env.reset()

        total_reward_dict = {}
        action_dict = {}

        env_config = configs["env"]
        actor_configs = configs["actors"]
        for actor_id in actor_configs.keys():
            total_reward_dict[actor_id] = 0
            if env._discrete_actions:
                import random
                t = random.random()
                if t<0.5:
                    action_dict[actor_id] = 6  # Forward right
                else:
                    action_dict[actor_id] = 5  # Forward left
            else:
                action_dict[actor_id] = [1, 0]  # test values

        start = time.time()
        i = 0
        done = {"__all__": False}
        while not done["__all__"]:
            # while i < 20:  # TEST
            i += 1
            obs, reward, done, info = env.step(action_dict)
            # action_dict = get_next_actions(info, env.discrete_actions)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
            print(":{}\n\t".join(["Step#", "rew", "ep_rew",
                                  "done{}"]).format(i, reward,
                                                    total_reward_dict, done))

            time.sleep(0.1)

        print("{} fps".format(i / (time.time() - start)))
