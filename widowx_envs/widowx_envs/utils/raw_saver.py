import os
import shutil
import pickle as pkl
import cv2
import copy
from widowx_envs.utils.utils import AttrDict
import glob
import json
import datetime

import numpy as np

def count_trajs(save_dir):

    base_folder = '/'.join(str.split(save_dir, '/')[:-1])
    trajs = glob.glob(base_folder + '/*/raw/traj_group*/*')
    return len(trajs)

class RawSaver():
    def __init__(self, save_dir, ngroup=1000):
        self.save_dir = save_dir
        self.ngroup = ngroup
        self.current_group = None
        self.current_group_folder = None

    def save_traj(self, itr, agent_data=None, obs_dict=None, policy_outputs=None, reward_data=None):

        print("##################################")
        print('saving trajectory number {}'.format(count_trajs(self.save_dir) + 1))
        print("##################################")

        igrp = itr // self.ngroup
        group_folder = os.path.join(self.save_dir , 'raw/traj_group{}'.format(igrp))

        # Handle group metadata when starting a new group
        if self.current_group != igrp:
            self.current_group = igrp
            self.current_group_folder = group_folder
            
            if not os.path.exists(group_folder):
                os.makedirs(group_folder)
                print("##################################")
                print(f"Saving trajectory group in {group_folder}")
                print("##################################")
                print("IMPORTANT!!"*25)
                print(f"After collecting trajectories, if there are any notes you want to add, you can do so in the post_collection_notes field in the group_metadata.json file.")
                print(f"That file is located at {group_folder}/group_metadata.json")
                print(f"For example, if you mess up the final trajectory or soemthing, just add that info in the post_collection_notes field for this trajectory group.")
                print("\nStarting new trajectory group {}".format(igrp))
                print(f"IMPORTANT!! IMPORTANT!! IMPORTANT!!")
                print("IMPORTANT!!: Read the info above please!")
                print("Enter a text prompt for this group of trajectories:")
                group_prompt = input().strip()
                print("Enter any additional info (collector name, notes, etc):")
                group_info = input().strip()
                
                if group_prompt or group_info:
                    with open(os.path.join(group_folder, 'group_metadata.json'), 'w') as f:
                        json.dump({
                            'group_id': igrp,
                            'prompt': group_prompt,
                            'additional_info': group_info,
                            'start_time': datetime.datetime.now().isoformat(),
                            'post_collection_notes': '',
                            'trajectories': []
                        }, f, indent=2)

        traj_folder = os.path.join(group_folder , 'traj{}'.format(itr))
        if os.path.exists(traj_folder):
            print('trajectory folder {} already exists, deleting the folder'.format(traj_folder))
            shutil.rmtree(traj_folder)

        os.makedirs(traj_folder)
        print('creating: ', traj_folder)

        if 'images' in obs_dict:
            images = obs_dict['images']
            T, n_cams = images.shape[:2]
            for i in range(n_cams):
                os.mkdir(traj_folder + '/images{}'.format(i))
            for t in range(T):
                for i in range(n_cams):
                    args = AttrDict(i=i, images=images, t=t, traj_folder=traj_folder)
                    save_single(args)
        if 'depth_images' in obs_dict:
            depth_images = obs_dict['depth_images']
            T, n_cams = depth_images.shape[:2]
            for i in range(n_cams):
                os.mkdir(traj_folder + '/depth_images{}'.format(i))
            for t in range(T):
                for i in range(n_cams):
                    cv2.imwrite('{}/depth_images{}/im_{}.png'.format(traj_folder, i, t), depth_images[t, i])

        if agent_data is not None:
            with open('{}/agent_data.pkl'.format(traj_folder), 'wb') as file:
                pkl.dump(agent_data, file)
        if obs_dict is not None:
            obs_dict_no_image = copy.deepcopy(obs_dict)
            obs_dict_no_image.pop('images')
            if "depth_images" in obs_dict_no_image:
                obs_dict_no_image.pop('depth_images')
            with open('{}/obs_dict.pkl'.format(traj_folder), 'wb') as file:
                pkl.dump(obs_dict_no_image, file)
        if policy_outputs is not None:
            with open('{}/policy_out.pkl'.format(traj_folder), 'wb') as file:
                pkl.dump(policy_outputs, file)
        if reward_data is not None:
            with open('{}/reward_data.pkl'.format(traj_folder), 'wb') as file:
                pkl.dump(reward_data, file)

class RawSaverRailRL(RawSaver):
    def __init__(self, save_dir, ngroup=1000):
        super(RawSaverRailRL, self).__init__(save_dir, ngroup=1000)
        self.i_traj = 0

    def save(self, path):
        obs = convert_listofdicts2dictoflists(path['full_observations'])
        self.save_traj(self.i_traj,
                       {'agent_infos': path['agent_infos'], 'env_infos': path['env_infos']},
                       obs, path['actions'], path['rewards'])
        print('saving traj {} done'.format(self.i_traj))
        self.i_traj += 1


def convert_listofdicts2dictoflists(list):
    obs_dict = {}
    for key in list[0].keys():
        vecs = []
        for tstep in list:
            vecs.append(tstep[key])
        obs_dict[key] = np.stack(vecs, 0)
    return obs_dict


def save_single(arg):
    cv2.imwrite('{}/images{}/im_{}.jpg'.format(arg.traj_folder, arg.i, arg.t), arg.images[arg.t, arg.i, :, :, ::-1])
    return True


