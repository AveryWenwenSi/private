import numpy as np


class Reward(object):
    def __init__(self):
        self.reward = 0.0
        self.prev = None
        self.curr = None

    def compute_reward(self, prev_measurement, curr_measurement, flag):
        self.prev = prev_measurement
        self.curr = curr_measurement

        if flag == "corl2017":
            return self.compute_reward_corl2017()
        elif flag == "lane_keep":
            return self.compute_reward_lane_keep()
        elif flag == "custom":
            return self.compute_reward_custom()
        elif flag == "hiway_lane_change":
            return self.compute_reward_hiway_lane_change()

    def compute_reward_custom(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
        self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) / 10
        new_damage = (
            self.curr["collision_vehicles"] +
            self.curr["collision_pedestrians"] + self.curr["collision_other"] -
            self.prev["collision_vehicles"] -
            self.prev["collision_pedestrians"] - self.prev["collision_other"])
        if new_damage:
            self.reward -= 100.0

        self.reward -= self.curr["intersection_offroad"] * 0.05
        self.reward -= self.curr["intersection_otherlane"] * 0.05

        if self.curr["next_command"] == "REACH_GOAL":
            self.reward += 100

        return self.reward

    def compute_reward_corl2017(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        # Distance travelled toward the goal in m
        self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
        # Change in speed (km/h)
        self.reward += 0.05 * (
            self.curr["forward_speed"] - self.prev["forward_speed"])
        # New collision damage
        self.reward -= .00002 * (
            self.curr["collision_vehicles"] +
            self.curr["collision_pedestrians"] + self.curr["collision_other"] -
            self.prev["collision_vehicles"] -
            self.prev["collision_pedestrians"] - self.prev["collision_other"])

        # New sidewalk intersection
        self.reward -= 2 * (self.curr["intersection_offroad"] -
                            self.prev["intersection_offroad"])

        # New opposite lane intersection
        self.reward -= 2 * (self.curr["intersection_otherlane"] -
                            self.prev["intersection_otherlane"])

        return self.reward

    def compute_reward_lane_keep(self):
        self.reward = 0.0
        # Speed reward, up 30.0 (km/h)
        self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) / 10
        # New collision damage
        new_damage = (
            self.curr["collision_vehicles"] +
            self.curr["collision_pedestrians"] + self.curr["collision_other"] -
            self.prev["collision_vehicles"] -
            self.prev["collision_pedestrians"] - self.prev["collision_other"])
        if new_damage:
            self.reward -= 100.0
        # Sidewalk intersection
        self.reward -= self.curr["intersection_offroad"]
        # Opposite lane intersection
        self.reward -= self.curr["intersection_otherlane"]

        return self.reward

    # adding new reward function for hiway different scenarios 
    def compute_reward_hiway_lane_change(self):
        self.reward = 0.0

        ## !! Here, spped also should not taken into considerations
        # Speed reward, up 30.0 (km/h)   ---on highway no limits on speed
        # self.reward +=  self.curr["forward_speed"] / 100.0
        # self.reward += 0.05 * (
        #     self.curr["forward_speed"] - self.prev["forward_speed"])     #instead using v, use a(acceleraion)

        # New collision damage
        new_damage = self.curr["collision_vehicles"]  - self.prev["collision_vehicles"] 
        if new_damage:
            self.reward -= 100.0          #?? Maybe set collision damage smaller, the cars can behave more aggresively? Since they are collecting the most reward
        
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        # Distance travelled toward the goal in m

        # when only forward movement is enabled : we can delete reward for distance going straight
        # self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0) * 0.1
       
        # add more rewards for Y axis change
        cur_dist_y = self.curr["y_to_goal"]
        prev_dist_y = self.prev["y_to_goal"]

        cur_dist_x = self.curr["x_to_goal"]
        prev_dist_x = self.prev["x_to_goal"]
        # Distance at y axis travelled toward the goal in m
        

        # self.reward += np.clip(prev_dist_y - cur_dist_y, -10.0, 10.0) * 0.5
        
        ############
        y_diff = np.clip(prev_dist_y - cur_dist_y, -10.0, 10.0) +0.001
        x_diff = np.clip(prev_dist_x - cur_dist_x, -10.0, 10.0) +0.001
        #self.reward += np.tanh(y_diff/x_diff)
        #self.reward += np.tanh(y_diff/(cur_dist / 5.0))

        #--------------------------
        #y_diff = np.clip(prev_dist_y - cur_dist_y, -10.0, 10.0) 
        #if y_diff>0:
        #    self.reward += np.clip(prev_dist_y - cur_dist_y, -10.0, 10.0) * 5
        #else:
        #    self.reward += np.clip(prev_dist_y - cur_dist_y, -10.0, 10.0) * 10.0

        #if self.curr["next_command"] == "REACH_GOAL":
        #    self.reward += 100

        #cur_action = self.curr["previous_action"]
        #prev_action = self.prev["previous_action"]
        #if cur_action == prev_action:     # if actions not changing so much, not constantly changing on the road
        #    self.reward+=0.01
        #if cur_action != prev_action:
        #    self.reward -=0.01             # penalizing constant change of action

        goal_y = self.curr["goal_y"]
        y = self.curr["y"]
        start_y = self.curr["start_y"]

        #
        if (y - start_y)/ (goal_y - start_y) < -0.0 :     # not going in the same direction
            self.reward -= 10.0
        else:                                              # going in the same direction 

            if (goal_y - y)* (goal_y - start_y) > 0.0 :     # going inside the expected trajectory range
                if np.abs(goal_y - y) <= 0.1:
                    self.reward += 10.0
                    if self.curr["car_id"] ==1:
                        print("closer in 0.1 range")
                elif np.abs(goal_y - y) <= 0.5:
                    self.reward += 5.0
                    if self.curr["car_id"] ==1:
                        print("closer in 0.5 range")
                elif np.abs(goal_y - y) <= 1.0:
                    self.reward += 1.0
                    if self.curr["car_id"] ==1:
                        print("closer in 1.0 range")
            else:
                if np.abs(goal_y - y) > 0.1:
                    self.reward -= 1.0
                    if self.curr["car_id"] ==1:
                        print("----out in 0.1 range")
                elif np.abs(goal_y - y) > 0.5:
                    self.reward -= 5.0
                    if self.curr["car_id"] ==1:
                        print("----out in 0.5 range")
                elif np.abs(goal_y - y) > 1.0:
                    self.reward -= 10.0
                    if self.curr["car_id"] ==1:
                        print("-----out in 1.0 range")
                
        
        if self.curr["car_id"] ==1:
            print("car ", self.curr["car_id"], " has reward of: ", self.reward)
            print("cyrrennt y position is: ", y, "  goal y pos is:  ", goal_y)
        
        return self.reward


    def destory(self):
        pass
