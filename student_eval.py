import os
import shutil
from tkinter.tix import IMAGE, Tree
from legged_gym import LEGGED_GYM_ROOT_DIR

from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import torch
import numpy as np
from isaacgym.torch_utils import *
from PIL import Image as im
from isaacgym import gymtorch, gymapi, gymutil
import time
from collections import deque
from torchvision.transforms import Resize

from imitateTSmdn import *


RESIZE = Resize((64, 64))
NORM = torch.tensor([1., 3., 3.], device=DEVICE)
CLIP = torch.tensor([[-1.28,  0.4],
                     [-7.76, 13.12],
                     [-3.2,  3.6]], device=DEVICE)
# from config import *
EXPORT_POLICY = False
RECORD_FRAMES = False
MOVE_CAMERA = False
NUM_ENVS = 32

TRAIN_VAL_SPLIT = 0.8  # train set ratio
EPOCHS = 20  # number of epochs
BATCH_SIZE = 256  # mb size
DATA_DIR = "./"
MODEL_FILE = "ImitateModel.pt"
NOISE_SCALE = 0.1
LR = 0.0001
TSITERS = 1000
IMAGE_SHAPE = (1, 240, 424)
ADD_NOISE = False
HOLE_P = 0.1
DEVICE = "cuda:0"


def _get_depth_image(env, camera_tensors, DEVICE):
    # get depth images from camera
    ##############
    # This block costs 0.1x s time
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    ##############

    # ###############
    # # This block costs 0.1x s
    env.gym.start_access_image_tensors(env.sim)
    depth_obs = torch.stack(camera_tensors, dim=0)
    env.gym.end_access_image_tensors(env.sim)
    # ###############

    depth_near, depth_far = 0.105, 5
    depth_obs = torch.clamp(-depth_obs, min=depth_near,
                            max=depth_far).to(DEVICE)
    if ADD_NOISE:
        depth_obs[torch.rand_like(depth_obs) < HOLE_P] = depth_far
    depth_obs = depth_obs / depth_far
    depth_obs = RESIZE(depth_obs[:, None, :, :])
    return depth_obs

    # return depth_camera_scan


def play(args, DEVICE="cuda:0"):

    IMAGE_HISTORY = deque(maxlen=IMAGE_SHAPE[0])

    for _ in range(3):
        IMAGE_HISTORY.append(torch.zeros(
            [NUM_ENVS, IMAGE_SHAPE[1], IMAGE_SHAPE[2]], device=DEVICE))

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    num_envs = min(env_cfg.env.num_envs, NUM_ENVS)

    env_cfg.env.num_envs = num_envs

    env_cfg.env.episode_length_s = 32
    env_cfg.record.record = True
    env_cfg.record.folder = os.path.join(
        LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
    env_cfg.terrain.num_rows = 20
    env_cfg.terrain.num_cols = 20
    env_cfg.terrain.terrain_proportions = [
        0.1, 0., 0.1, 0.1, 0., 0, 0.7, 0]  # train

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device)
    ################# working ####################
    _f = "./runs/debug_dagger_concat_dataset/Model_NOW.pt"
    ##############################################

    policy2 = torch.load(_f)
    policy2.eval()
    base_quat = env.root_states[:, 3:7]
    camera_handles = []
    camera_tensors = []
    camera_properties = gymapi.CameraProperties()
    camera_properties.height = IMAGE_SHAPE[1]  # 360
    camera_properties.width = IMAGE_SHAPE[2]  # 640
    camera_properties.enable_tensors = True
    camera_properties.horizontal_fov = 87
    # supersampling_horizontal  # maybe useful   
    for i in range(num_envs):

        camera_handle = env.gym.create_camera_sensor(
            env.envs[i], camera_properties)

        local_transform = gymapi.Transform()
        # TODO: get correct parameter
        local_transform.p = gymapi.Vec3(0.3, 0.0, 0.0)
        _cam_quat = quat_mul(base_quat[i], torch.tensor(
            [0, 0, np.sin(0 / 2), np.cos(0 / 2)], device=env.device, dtype=torch.float))
        # _cam_quat = quat_mul(self.base_quat[i], torch.tensor([0., 0., 1., 0.], device=self.device, dtype=torch.float))
        local_transform.r = gymapi.Quat(
            _cam_quat[0], _cam_quat[1], _cam_quat[2], _cam_quat[3])
        actor_handle = env.gym.get_actor_handle(env.envs[i], 0)

        body_handle = env.gym.find_actor_rigid_body_handle(
            env.envs[i], actor_handle, "base")
        assert body_handle >= 0
        env.gym.attach_camera_to_body(
            camera_handle, env.envs[i], body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
        camera_handles.append(camera_handle)
        camera_buffer = env.gym.get_camera_image_gpu_tensor(
            env.sim, env.envs[i], camera_handle, gymapi.IMAGE_DEPTH)
        camera_tensor = gymtorch.wrap_tensor(camera_buffer)
        camera_tensors.append(camera_tensor)

    # for i in range(10*int(env.max_episode_length)):
    step = 0
    # time_previous = time.time()

    depth_obs = _get_depth_image(env, camera_tensors, DEVICE)

    while True:

        # actions = policy(obs.detach())
        obs = obs[:, 6:50]
        actions2 = policy2.sample(obs, depth_obs, test=True, resamples=5)
        actions2 = torch.clamp(actions2.view((-1, 3)),
                               CLIP[:, 0], CLIP[:, 1])
        actions2 = actions2.view(-1, 12)
        obs, _, rews, dones, infos = env.step(actions2.detach())

        depth_obs = _get_depth_image(env, camera_tensors, DEVICE)

        step += 1
        yield (obs, depth_obs, rews, actions2.to(DEVICE))


if __name__ == '__main__':
    args = get_args()
    # gym = gymapi.acquire_gym()
    data_generator = play(args)
    for i, data in enumerate(data_generator):
        current_frame = data
        if i > 4096:
            break
