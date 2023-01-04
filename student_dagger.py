
import os
from legged_gym import LEGGED_GYM_ROOT_DIR

from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import torch
from imitateTSmdn import *
import numpy as np
from isaacgym.torch_utils import *
from PIL import Image as im
from isaacgym import gymtorch, gymapi, gymutil
from collections import deque
from torchvision.transforms import Resize
from torch.utils.tensorboard import SummaryWriter


RESIZE = Resize((64, 64))
EXP_NAME = 'dagger_action_clip_epochs32_collect32_Reuse_5_env1024_obs44_splitdata2'
# from config import *
EXPORT_POLICY = False
RECORD_FRAMES = False
MOVE_CAMERA = False

# c1 1024 e32 l32   better than c2
# c2 1024 e64 l32

NUM_ENVS = 1024  # 512
EPOCHS = 32  # number of epochs
COLLECT_LENGTH = 32
DAGGER_RESUED = 5

BATCH_SIZE = 128  # 128  # mb size
DATA_DIR = "./"
NOISE_SCALE = 0.1
LR = 0.0001
IMAGE_SHAPE = (1, 240, 424)
ADD_NOISE = False
HOLE_P = 0.1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NORM = torch.tensor([1., 10., 3.], device=DEVICE)
CLIP = torch.tensor([[-1.28,  0.4],
                     [-7.76, 13.12],
                     [-3.2,  3.6]], device=DEVICE)


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


def play(args):
    # STEP is stand for sim step, in each sim step, NUM_ENVS agent perform on step (in sim)
    STEP = 0

    # attach img history
    # IMAGE_HISTORY = deque(maxlen=IMAGE_SHAPE[0])

    # for _ in range(3):
    #     IMAGE_HISTORY.append(torch.zeros(
    #         [NUM_ENVS, IMAGE_SHAPE[1], IMAGE_SHAPE[2]], device=DEVICE))

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    # num_envs = min(env_cfg.env.num_envs, NUM_ENVS)

    env_cfg.env.num_envs = NUM_ENVS
    env_cfg.env.episode_length_s = 32   # 32s 32*50 step
    env_cfg.record.record = False
    env_cfg.record.folder = os.path.join(
        LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
    env_cfg.terrain.num_rows = 20
    env_cfg.terrain.num_cols = 20
    env_cfg.terrain.terrain_proportions = [
        0.1, 0., 0.1, 0.1, 0., 0, 0.7, 0]  # train
    # teacher should not observe noisy state
    env_cfg.noise.add_noise = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    base_quat = env.root_states[:, 3:7]
    camera_handles = []
    camera_tensors = []
    camera_properties = gymapi.CameraProperties()
    camera_properties.height = IMAGE_SHAPE[1]  # 360
    camera_properties.width = IMAGE_SHAPE[2]  # 640
    camera_properties.enable_tensors = True
    camera_properties.horizontal_fov = 87
    # supersampling_horizontal  # maybe useful   
    for i in range(NUM_ENVS):

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

    depth_obs = _get_depth_image(env, camera_tensors, DEVICE)

    while True:
        # reoload model after training a new model (train model every COLLECT_LENGTH)
        if STEP % COLLECT_LENGTH == 0:
            policy_eval = torch.load(f'./runs/{EXP_NAME}/Model_NOW.pt').eval()

        teacher_actions = policy(obs.detach())
        obs = obs[:, 6:50]
        # obs[:, :6][obs[:, :6] >= 0] = 1
        # obs[:, :6][obs[:, :6] < 0] = -1
        #### generate student action ####
        teacher_actions = torch.clamp(teacher_actions.view((-1, 3)),
                                      CLIP[:, 0], CLIP[:, 1]).view(-1, 12)

        student_actions = policy_eval.sample(obs, depth_obs)
        # student_actions = torch.clamp(student_actions.view((-1, 3)),
        #                        CLIP[:, 0], CLIP[:, 1])
        # student_actions = student_actions.view(-1, 12)
        #################################
        obs, _, rews, dones, infos = env.step(student_actions.detach())

        depth_obs = _get_depth_image(env, camera_tensors, DEVICE)

        # print("teacher action: ", teacher_actions)
        # print("student action", student_actions)

        # add losses to tensorboard
        mape = custom_loss(student_actions, teacher_actions)

        l_dict = {f"collect/MAPE_{i}": mape[i] for i in range(3)}
        collect_loss = {
            **l_dict, "collect/collect_L1": nn.L1Loss()(student_actions, teacher_actions)}
        for k, v in collect_loss.items():
            writer.add_scalar(k, v, STEP)

        STEP += 1

        yield (obs, depth_obs.squeeze(1), rews, teacher_actions.to(DEVICE))


def train(model, val_loader, current_train_iter):
    """
    Training main method
    :param model: the network
    :param device: the cuda device
    """
    # global STEP

    loss_function = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # construct train set
    train_set = CustomTrainDataset(idx=list(range(
        0, COLLECT_LENGTH*DAGGER_RESUED)), batch_size=BATCH_SIZE, filename="./tmp_dagger_data/obs_image_rews_action_daggar")
    sampler_train = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(train_set), BATCH_SIZE, False)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               sampler=sampler_train,
                                               batch_size=None)

    for batch_now in range(EPOCHS):  # epochs
        for j in range(DAGGER_RESUED):
            train_loss = trainOnBatch(model,
                                      loss_function,
                                      optimizer,
                                      train_loader)

            test_loss = test(model, loss_function, val_loader)
            loss_logging = {**train_loss, **test_loss}
            for k, v in loss_logging.items():
                writer.add_scalar(k, v, current_train_iter*EPOCHS+batch_now+j)

        torch.save(model, f'./runs/{EXP_NAME}/Model_{current_train_iter}.pt')
    return model


def main(args):

    # load student model (random init)
    student_model = MyModel(BcModel().to(DEVICE), MixtureDensityNetwork(
        128+256, 12, n_components=12).to(DEVICE)).to(DEVICE)
    torch.save(student_model, f'./runs/{EXP_NAME}/Model_NOW.pt')

    data_generator = play(args)
    # current_train_step = 0
    # generate eval data set
    for i, data in enumerate(data_generator):
        if i == 32:
            break
        current_frame = data
        # print(current_data_ind)
        torch.save(
            current_frame, f"tmp_dagger_data_test/obs_image_rews_action_daggar_{i}")

    val_set = CustomValDataset(idx=list(
        range(0, 32)), filename="./tmp_dagger_data_test/obs_image_rews_action_daggar")
    sampler_val = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(val_set), BATCH_SIZE, False)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             sampler=sampler_val,
                                             batch_size=None)
    for i, data in enumerate(data_generator):
        current_frame = data
        # print(current_data_ind)
        torch.save(
            current_frame, f"tmp_dagger_data/obs_image_rews_action_daggar_{i%(COLLECT_LENGTH*DAGGER_RESUED)}")
        # train a new student_model once step COLLECT_LENGTH in sim
        if (i % COLLECT_LENGTH == 0) and (i >= (COLLECT_LENGTH*DAGGER_RESUED)):
            # current_train_iter   collect COLLECT_LENGTH data in sim and train, called one iter
            # in one iter, train EPOCHS times one COLLECT_LENGTH data
            student_model = train(
                student_model, val_loader, i // COLLECT_LENGTH)
            torch.save(student_model, f'./runs/{EXP_NAME}/Model_NOW.pt')


if __name__ == '__main__':
    args = get_args()
    writer = SummaryWriter(f'./runs/{EXP_NAME}')
    main(args=args)
