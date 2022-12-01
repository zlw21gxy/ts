import torch
import numpy as np
import torch.nn as nn
import torch.utils.data  # 如果不用这个就会出现pycharm不识别data的问题
import torch.optim as optim
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from mdn import MixtureDensityNetwork
from typing import List
from torchvision.transforms import Resize, RandomPerspective, RandomSolarize, GaussianBlur


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_VAL_SPLIT = 0.8  # train/val ratio
EPOCHS = 128  # number of epochs
BATCH_SIZE = 128  # mb size
MODEL_FILE = "ImitateModel.pt"
NOISE_SCALE = 0.1
LR = 0.0005
EXP_NAME = "Dagger_whole_clip_norm2_lr0.0005_128batch_imgoise_obsnoise_data1_leakyrelu"


def custom_loss(input, target):
    l = torch.abs(1000*(input.view(-1, 3) - target.view(-1, 3)) /
                  1000*target.view(-1, 3))
    l = torch.clamp(l, 0, 2)
    return l.mean(axis=0)


class CustomImgT(torch.nn.Module):
    def __init__(self, p=0.05) -> None:
        super().__init__()
        self.p = p

    def forward(self, img):
        depth_near, depth_far = 0.105, 5

        # depth_obs = torch.clamp(-depth_obs, min=depth_near, max=depth_far).view((self.num_envs, -1))
        # depth_obs = torch.clamp(-img, min=depth_near, max=depth_far)
        depth_obs = img
        depth_obs[torch.rand_like(depth_obs, device=DEVICE) < self.p] = depth_far
        # depth_obs = depth_obs / depth_far
        return depth_obs


class CustomTrainDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, idx: List = [1, 2, 3], batch_size: int = 128, filename="/home/guxy/data_backup/dataTS/obs_image_rews_action") -> None:
        self.idx = idx    # circular linked list
        self.batch_size = batch_size
        self.__resize = Resize((64, 64))
        self.img_transformations = nn.Sequential(CustomImgT(0.05))
        self.filename = filename
        self.__reload_current()
        self.__gen_noise()
        # self.current_batch_usage = 0
        # self.img_transformations = nn.Sequential(Resize((64, 64)), RandomPerspective(0.1, 0.1), CustomImgT(0.05))
        self.clip = torch.tensor([[-1.28,  0.4],
                                  [-7.76, 13.12],
                                  [-3.2,  3.6]], device=DEVICE)
        self.norm = torch.tensor([1., 3., 3.], device=DEVICE)

    def __reload_current(self):

        self.current_batch_usage = 0

        data = torch.load(self.filename + f"_{self.idx.pop(0)}")
        self.tensors_obs = data[0][:, 6:50]     # N， 44
        self.tensors_img = self.__resize(data[1])
        self.tensors_action = data[3]  # N , 12
        for idx_now in self.idx:
            data = torch.load(self.filename + f"_{idx_now}")

            self.tensors_obs = torch.concat(
                (self.tensors_obs, data[0][:, 6:50]), dim=0)
            self.tensors_img = torch.concat(
                (self.tensors_img, self.__resize(data[1])), dim=0)
            self.tensors_action = torch.concat(
                (self.tensors_action, data[3]), dim=0)


    def __gen_noise(self):
        noise_vec = torch.zeros(50, device=DEVICE)
        noise_vec[:3] = 0.
        noise_vec[3:6] = 0.
        noise_vec[6:9] = 0.05
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:24] = 0.01 * 1.0
        noise_vec[24:36] = 1.5 * 0.05
        noise_vec[36:48] = 0.  # previous actions
        noise_vec[48:50] = 0.  # terrain info
        self.noise_scale_vec = noise_vec[6:50]

    def __getitem__(self, index):

        obs = self.tensors_obs[index]
        img = self.tensors_img[index][:, None, :, :]

        # add noise to img
        img = self.img_transformations(img)
        # add noise to obs
        obs += (2 * torch.rand_like(obs, device=DEVICE) - 1) * self.noise_scale_vec

        action = self.tensors_action[index]
        action = torch.clamp(action.view((-1, 3)),
                             self.clip[:, 0], self.clip[:, 1]) / self.norm
        action = action.view(-1, 12)
        return obs, img, action

    def __len__(self):
        return self.tensors_obs.size(0)


class CustomValDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, idx: List = [1, 2, 3], batch_size: int = 128, filename="./tmp_dagger_data_test/obs_image_rews_action") -> None:
        self.idx = idx
        self.batch_size = batch_size
        self.img_transformations = nn.Sequential(
            Resize((64, 64)), CustomImgT(0.05))
        self.filename = filename
        self.clip = torch.tensor([[-1.28,  0.4],
                                  [-7.76, 13.12],
                                  [-3.2,  3.6]], device=DEVICE)

        self.norm = torch.tensor([1., 3., 3.], device=DEVICE)
        self.__reload_current()
        # self.current_batch_usage = 0
        # self.img_transformations = nn.Sequential(Resize((64, 64)), RandomPerspective(0.1, 0.1), CustomImgT(0.05))

    def __reload_current(self):

        self.current_batch_usage = 0
        data = torch.load(self.filename + f"_{self.idx.pop()}")
        self.tensors_obs = data[0][:1024, 6:50]

        self.tensors_img = self.img_transformations(data[1])
        self.tensors_action = data[3][:1024, :]

        for idx_now in self.idx:
            data = torch.load(
                self.filename + f"_{idx_now}")
            self.tensors_obs = torch.concat(
                (self.tensors_obs, data[0][:1024, 6:50]), dim=0)
            _img = self.img_transformations(data[1])
            self.tensors_img = torch.concat((self.tensors_img, _img), dim=0)
            self.tensors_action = torch.concat(
                (self.tensors_action, data[3][:1024, :]), dim=0)


    def __getitem__(self, index):

        obs = self.tensors_obs[index]
        # get depth
        img = self.tensors_img[index][:, None, :, :]

        action = self.tensors_action[index]
        action = torch.clamp(action.view((-1, 3)),
                             self.clip[:, 0], self.clip[:, 1]) / self.norm
        action = action.view(-1, 12)

        return obs, img, action

    def __len__(self):
        return self.tensors_obs.size(0)


class CnnModel(torch.nn.Module):
    def __init__(self, image_shape, activation, out_dim=256) -> None:
        super().__init__()
        self.image_shape = image_shape
        CNN_layer = []
        CNN_layer.append(nn.Conv2d(
            in_channels=image_shape[0], out_channels=32, kernel_size=8, stride=4))
        CNN_layer.append(activation)
        CNN_layer.append(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=4, stride=2))
        CNN_layer.append(activation)
        CNN_layer.append(
            nn.Conv2d(in_channels=128, out_channels=out_dim, kernel_size=3, stride=1))
        # CNN_layer.append(activation)
        # CNN_layer.append(nn.Flatten())
        CNN_layer.append(nn.AdaptiveAvgPool2d((1, 1)))
        CNN_layer.append(nn.Flatten())

        self.embedding_model = nn.Sequential(*CNN_layer)

    def forward(self, image):
        out = self.embedding_model(image)
        # out = self.linear(out)
        return out


class BcModel(torch.nn.Module):
    def __init__(self):
        super(BcModel, self).__init__()

        self.embedding_net = torch.nn.Sequential(
            torch.nn.Linear(44, 256),
            # torch.nn.ELU(),
            # torch.nn.ReLU(),
            torch.nn.LeakyReLU(),
            # torch.nn.BatchNorm1d(256),
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
        )

        self.depth_net = CnnModel(image_shape=(
            1, 64, 64), activation=nn.LeakyReLU(), out_dim=256)

        # self.action_net = torch.nn.Sequential(
        # torch.nn.Linear(128+256, 512),
        # torch.nn.ELU(),
        # torch.nn.BatchNorm1d(512),
        # torch.nn.Dropout(0.2),
        # torch.nn.Linear(512, 12),

    def forward(self, obs, img):
        '''
        obs: N, 450
        img: N, 2, 64, 64
        '''
        obs_embedding = self.embedding_net(obs)
        img_embedding = self.depth_net(img)
        # print(obs_embedding.shape, img_embedding.shape)
        # print(torch.concat((obs_embedding, img_embedding), dim=1).shape)
        emb_to_mdn = torch.concat((obs_embedding, img_embedding), dim=1)
        return emb_to_mdn



def create_datasets():
    """Create training and validation datasets"""

    # train_set = CustomTrainDataset(idx=list(range(0, 4000)))
    # train_set = CustomTrainDataset(idx=list(range(0, 128)), filename="/home/guxy/project/dog/fold/imitate/tmp_dagger_data/obs_image_rews_action_daggar")
    train_set = CustomTrainDataset(idx=list(range(
        0, 32)), filename="/home/guxy/project/dog/fold/imitate/tmp_dagger_data_test/obs_image_rews_action_daggar")
    # train_set = CustomTrainDataset2(idx=list(range(0, 32)))

    sampler_train = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(train_set), BATCH_SIZE, False)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               sampler=sampler_train,
                                               batch_size=None)

    val_set = CustomValDataset(idx=list(range(4000, 4006)))
    sampler_val = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(val_set), BATCH_SIZE, False)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             sampler=sampler_val,
                                             batch_size=None)
    return train_loader, val_loader


def trainOnBatch(model, loss_function, optimizer, data_loader):
    """Train for a single epoch"""

    # set model to training mode
    model.train()

    current_loss = 0.0
    # current_L1 = 0.
    # iterate over the training data
    for i, (obs, img, action) in enumerate(data_loader):
        # obs: N, 50
        # img: N, 2, 64, 64
        # action: N, 12
        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # forward
            loss = model.loss(obs, img, action)
            # backward
            loss.backward()
            # Qloss.backward()
            optimizer.step()

        # statistics
        current_loss += loss.item()
    current_loss = current_loss / (i + 1)
    outputs = model.sample(obs, img)
    current_MAPE = custom_loss(action, outputs)
    current_L1 = loss_function(action, outputs)
    l_dict = {f"train/MAPE_{i}": current_MAPE[i] for i in range(3)}
    # print(f'Test Loss: {total_loss}')
    return {"train/train_loss": current_loss, **l_dict, "train/train_L1": current_L1}
    # return {"train_loss": current_loss}


def train(model):
    """
    Training main method
    :param model: the network
    :param device: the cuda device
    """

    loss_function = nn.MSELoss()
    # loss_function = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_loader, val_lorder = create_datasets()  # read datasets

    # train
    with tqdm(range(EPOCHS)) as bar:
        for batch_now in bar:
            # print('batch_now {}/{}'.format(batch_now + 1, EPOCHS*4000))
            train_loss = trainOnBatch(model,
                                      loss_function,
                                      optimizer,
                                      train_loader)
            # # save model
            # model_path = os.path.join(DATA_DIR, MODEL_FILE)
            # torch.save(model.state_dict(), model_path)
            if batch_now % 1 == 0:
                test_loss = test(model, loss_function, val_lorder)
                loss_logging = {**train_loss, **test_loss}
                for k, v in loss_logging.items():
                    writer.add_scalar(k, v, batch_now)

                bar.set_postfix(
                    {"train_loss": loss_logging["train/train_loss"], "test_L1": loss_logging["test/test_L1"]})

            # if batch_now % 50 == 0:
                # torch.save(model.state_dict(), f'./runs/{EXP_NAME}/Model_{batch_now}.pt')
                torch.save(model, f'./runs/{EXP_NAME}/Model_{batch_now}.pt')


def test(model, loss_function, data_loader):
    """Test over the whole dataset"""

    model.eval()  # set model in evaluation mode

    current_loss = 0.0
    current_MAPE = torch.zeros(3, device=DEVICE)
    for i, (obs, img, action) in enumerate(data_loader):

        with torch.set_grad_enabled(False):
            # forward
            outputs = model.sample(obs, img)
            # Qloss = QuantileLoss(outputs, labels)
            loss = loss_function(outputs, action)
            mape = custom_loss(outputs, action)

        # statistics
        current_loss += loss.item()
        current_MAPE += mape

    current_MAPE = current_MAPE / (i + 1)
    current_loss = current_loss / (i + 1)
    l_dict = {f"test/MAPE_{i}": current_MAPE[i] for i in range(3)}
    # current_L1 = current_L1 / (i + 1)
    # print(f'Test Loss: {total_loss}')
    return {"test/test_L1": current_loss, **l_dict}


class MyModel(nn.Module):
    def __init__(self, emb_model, mix_model) -> None:
        super().__init__()
        self.emb_model = emb_model
        self.mix_model = mix_model

    def forward(self, obs, img):
        emb = self.emb_model.forward(obs, img)
        mixture_model = self.mix_model.forward(emb)
        return mixture_model

    def loss(self, obs, img, action):
        mixture_model = self.forward(obs, img)
        log_prob = mixture_model.log_prob(action)
        loss = -torch.mean(log_prob)
        return loss

    def sample(self, obs, img, test=False, resamples=5):
        mixture_model = self.forward(obs, img)
        sample0 = mixture_model.sample()
        if test:
            for _ in range(resamples):
                sample1 = mixture_model.sample()
                cond_filter = mixture_model.log_prob(
                    sample1) > mixture_model.log_prob(sample0)
                sample0[cond_filter] = sample1[cond_filter]
        return sample0


if __name__ == '__main__':
    writer = SummaryWriter(f'./runs/{EXP_NAME}')

    # imitate_model = QuantileModel(hidden=[512, 256, 64]).to(DEVICE)
    imitate_model = BcModel().to(DEVICE)
    mdn_model = MixtureDensityNetwork(256+128, 12, n_components=8).to(DEVICE)

    train(MyModel(imitate_model, mdn_model))
