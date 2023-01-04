import torch
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import  MultivariateNormal, MixtureSameFamily, Categorical


class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.
    [ Bishop, 1994 ]
    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components):
        super().__init__()
        self.pi_network = CategoricalNetwork(dim_in, n_components)
        self.normal_network = MixtureDiagNormalNetwork(dim_in, dim_out,
                                                       n_components)

    def forward(self, x):
        comp = self.normal_network(x)
        mix = self.pi_network(x)
        mixture_model = MixtureSameFamily(mix, comp) 

        return mixture_model

    def loss(self, x, y):
        mixture_model = self.forward(x)
        log_prob = mixture_model.log_prob(y)
        loss = -torch.mean(log_prob)
        return loss

    def sample(self, x, test=False, resamples=5):
        mixture_model = self.forward(x)
        sample0 = mixture_model.sample()
        if test:
            for _ in range(resamples):
                sample1 = mixture_model.sample()  
                cond_filter = mixture_model.log_prob(sample1) > mixture_model.log_prob(sample0)
                sample0[cond_filter] = sample1[cond_filter]
        return sample0


class MixtureDiagNormalNetwork(nn.Module):
    """
    creata MultivariateNormal distribution, which covariance_matrix is diagonal
    """
    def __init__(self, in_dim, out_dim, n_components, hidden_dim=512):
        super().__init__()
        self.n_components = n_components

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.total_outs = self.n_components * (out_dim + out_dim)  # n Multi-gaussian

        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # torch.nn.BatchNorm1d(hidden_dim),
            # torch.nn.Dropout(0.2),
            # nn.ELU(),
            # torch.nn.ReLU(),
            torch.nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.total_outs),
        )

    def forward(self, data):
        params = self.network(data)

        mean, sd = torch.split(params, self.out_dim * self.n_components, dim=1)

        mean = torch.stack(mean.split(self.out_dim, 1), 1)
        sd = torch.stack(sd.split(self.out_dim, 1), 1)
        
        sd = F.elu(sd) + 1 + 1e-7

        sd = torch.diag_embed(sd)

        multi_normal = MultivariateNormal(mean, sd)
        return multi_normal


class CategoricalNetwork(nn.Module):
    """
    Categorical distribution 
    """

    def __init__(self, in_dim, out_dim, hidden_dim=512):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # torch.nn.BatchNorm1d(hidden_dim),
            # torch.nn.Dropout(0.2),
            # nn.ELU(),
            torch.nn.LeakyReLU(),
            # torch.nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        params = self.network(x)
        return Categorical(logits=params)



if __name__ == "__main__":

    def train(x, y):

        model = MixtureDensityNetwork(x.shape[1], y.shape[1], n_components=4).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # x, y = torch.tensor(x_data), torch.tensor(y_data)
        for i in range(20_000):
            optimizer.zero_grad()
            loss = model.loss(x, y)
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print(loss)
        return model

    def gen_data(n=512):
        y = np.linspace(-1, 1, n)
        x = 7 * np.sin(5 * y) + 0.5 * y + 0.5 * np.random.randn(*y.shape)
        return x[:,np.newaxis], y[:,np.newaxis]

    # x = np.random.random((512, 15))
    # y = np.random.random((512, 4))
    x, y = gen_data()
    x = torch.Tensor(x).cuda()
    y = torch.Tensor(y).cuda()
    model = train(x, y)
    # torch.save(model, "mdn_model.pt")
    # torch.save(model.state_dict(), "mdn_model_dict.pt")
    plt.figure()
    plt.scatter(x.cpu().numpy(), y.cpu().numpy(), c="r")
    samples = model.sample(x)    
    samples_test = model.sample(x, test=True)

    plt.scatter(x.cpu().numpy(), samples.cpu().numpy())
    plt.scatter(x.cpu().numpy(), samples_test.cpu().numpy(), c="g")

    plt.savefig("mdn_sample.png")
