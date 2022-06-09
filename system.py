import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from oscillator import Oscillator, TorchOscillator
from limb import TorchOneLinkTorqueLimb, TorchOneLinkMuscleLimb

#TODO: muscle not initially tight?
#TODO: learning rate higher after oscillator? (so it tends to exploit whatever basis is available)
#TODO: remove second conv layer?
#TODO: save oscillator params - make oscillator a nn.Module and impl state_dict and load_state_dict
#TODO: provide steady-state input to oscillators?
#TODO: need curriculum training for more complex action spaces?
#TODO: gate other additive controls as well, e.g. feedback control, low-level reflexes
#TODO: how to control movement vigour - consider role of basal ganglia? more than that


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.n_oscillators = 50
        self.n_per_oscillator = 40

        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.n_oscillators*2)

        # note inputs to physical system start small without bias for stability

        self.fcdirect = nn.Linear(128, 2)
        self.fcdirect.weight = torch.nn.parameter.Parameter(self.fcdirect.weight / 100)
        self.fcdirect.bias = torch.nn.parameter.Parameter(self.fcdirect.bias * 0 + .1)
        # self.fcdirect.bias = torch.nn.parameter.Parameter(self.fcdirect.bias * 0 + torch.Tensor([0, .5]))

        self.oscillators = []
        for i in range(self.n_oscillators):
            freq = 4*np.pi*np.random.rand()
            tau = .5*np.random.rand()
            o = Oscillator(self.n_per_oscillator, freq=freq, tau=tau)
            torch_oscillator = TorchOscillator(o.encoders, o.biases, o.decoders)
            self.oscillators.append(torch_oscillator)

        nb = 32
        self.fc4 = nn.Linear(self.n_oscillators * self.n_per_oscillator, nb)

        # self.fc5a = nn.Linear(self.n_oscillators * self.n_per_oscillator, 2)
        # self.fc5a.weight = torch.nn.parameter.Parameter(self.fc5a.weight / 10000)
        # self.fc5a.bias = torch.nn.parameter.Parameter(self.fc5a.bias * 0)

        self.fc5b = nn.Linear(nb, 2)
        self.fc5b.weight = torch.nn.parameter.Parameter(self.fc5b.weight / 10000)
        self.fc5b.bias = torch.nn.parameter.Parameter(self.fc5b.bias * 0)

        # self.limb = TorchOneLinkTorqueLimb()
        self.limb = TorchOneLinkMuscleLimb()

        self.device = None

    def forward(self, x, return_oscillator_state=False):
        """
        :param x: (current angle, desired angle)
        """

        batch_size = x.shape[0]
        current_angle = x[:,0]

        x = self.fc1(x)
        x = F.relu(x)
        direct = self.fcdirect(x) # direct static input to muscles rather than oscillators
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        o_states = torch.zeros((self.n_oscillators, batch_size, 2))
        # l_state = torch.zeros((batch_size, 2))
        # l_state[:,0] = current_angle
        l_state = torch.zeros((batch_size, 4))
        l_state[:,0] = current_angle
        l_state[:,2] = self.limb.get_m1_length(current_angle)
        l_state[:,3] = self.limb.get_m2_length(current_angle)

        dt = 0.001
        steps = 700
        # l_states = torch.zeros(steps, batch_size, 2)
        # torques = torch.zeros(steps, batch_size)
        l_states = torch.zeros(steps, batch_size, 4)
        activations = torch.zeros(steps, batch_size, 2)

        if self.device is not None:
            o_states = o_states.to(self.device)
            l_state = l_state.to(self.device)
            l_states = l_states.to(self.device)
            activations = activations.to(self.device)

        o_state_history = []

        if torch.mean(x).isnan(): # alert to unstable run
           print(x)

        # input sets initial oscillator state
        for j in range(self.n_oscillators):
            # TODO: use torch.split instead of torch.narrow?
            xi = torch.narrow(x, 1, j * 2, 2)
            o_states[j,:,:] = xi

        for i in range(steps):
            activities = torch.zeros((batch_size, self.n_oscillators*self.n_per_oscillator))
            if self.device is not None:
                activities = activities.to(self.device)

            for j in range(self.n_oscillators):
                a = self.oscillators[j].get_activities(o_states[j,:,:])
                activities[:,j*self.n_per_oscillator:(j+1)*self.n_per_oscillator] = a
                derivative = torch.matmul(self.oscillators[j].decoders, a.T).T

                o_states[j,:,:] = o_states[j,:,:] + dt * derivative

            if return_oscillator_state:
                o_state_history.append(o_states.detach().cpu().numpy().copy())

            # a = self.fc5a(activities)
            b = self.fc4(activities)
            b = F.relu(b)
            b = self.fc5b(b)
            # activation = (a + b + direct)
            activation = (b + direct)

            activations[i,:,:] = activation

            derivative = self.limb.f(l_state, activation)
            l_state = l_state + dt*derivative

            # range of motion limits
            limited = torch.zeros_like(l_state)
            pos = l_state[:, 0]
            limited[:,0] = torch.clip(pos, -np.pi / 2, np.pi / 2)
            limited[:,1] = l_state[:,1] * torch.gt(pos, -np.pi/2).int() * torch.lt(pos, np.pi/2).int()
            limited[:,2:] = l_state[:,2:]
            l_state = limited

            l_states[i,:,:] = l_state

        if return_oscillator_state:
            return l_states, activations, np.array(o_state_history)
        else:
            return l_states, activations


def train(net, batch_size=3, batches=20000, device=None):
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(net.parameters(), lr=.00001, momentum=.9)
    optimizer = optim.Adam(net.parameters(), lr=.0005, weight_decay=1e-6)

    running_loss = 0.0
    best_loss = 1e6
    for i in range(batches):
        # if i == 400: #freeze pre-oscillator layers after a bit?
        #     net.fc1.requires_grad = False
        #     net.fc2.requires_grad = False
        #     net.fc3.requires_grad = False

        input = -np.pi/2 + np.pi*np.random.rand(batch_size, 2)
        # input[:,0] = 0

        input = torch.Tensor(input)
        if device is not None:
            input = input.to(device)

        optimizer.zero_grad()

        outputs, torques = net(input)

        print('.', end='')

        target_loss = torch.mean( (outputs[:,:,0] - input[:,1]).pow(2) )
        torque_loss = torch.mean(torques.pow(2))
        loss = target_loss + torque_loss/10 # using velocity loss impairs learning
        # loss = target_loss + velocity_loss + torque_loss

        # param_norm = sum(param.norm(2) for param in net.parameters())
        # loss = loss + 1e-6 * param_norm

        # print('start: {} target: {}'.format(start, target))
        # print('{} {}'.format(i, loss.item()))
        running_loss += loss.mean().detach()
        if i % 50 == 49:
            mean_loss = running_loss / 50
            print('[%5d] loss: %.3f' % (i + 1, mean_loss))

            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save({
                    'epoch': i,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': mean_loss
                    }, 'oscillator_limb_checkpoint.pt')

            running_loss = 0.0

            if mean_loss < .2:
                plot_example(net, device)

            torch.cuda.empty_cache()

        loss.mean().backward()
        optimizer.step()


def plot_example(net, device=None):
    start = -np.pi / 2 + np.pi * np.random.rand()
    target = -np.pi / 2 + np.pi * np.random.rand()
    print(target)

    input = torch.Tensor([[start, target]])
    if device is not None:
        input = input.to(device)
    outputs, torques, o_states = net.forward(input, return_oscillator_state=True)
    outputs = outputs.detach().cpu().numpy()
    outputs = outputs[:,0,:]
    torques = torques.detach().cpu().numpy()

    time = np.arange(outputs.shape[0]) * .01
    plt.figure(figsize=(10,4))
    plt.subplot(131)
    plt.plot(time, target*np.ones_like(time))
    plt.plot(time, outputs)
    plt.subplot(132)
    plt.plot(time, torques.squeeze())
    plt.subplot(133)
    print(o_states.shape)
    print(np.min(o_states.flatten()))
    for i in range(o_states.shape[1]):
        plt.plot(time, o_states[:,i,0], '.')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    net = Net()
    # print(net)

    current_angle = 0.0
    desired_angle = 1.0
    input = torch.tensor([[current_angle, desired_angle]])
    l_states, activations = net.forward(input)
    plt.plot(l_states.detach().numpy().squeeze())
    plt.legend(('angle', 'velocity', 'm1', 'm2'))
    plt.show()
    plt.plot(activations.detach().numpy().squeeze())
    plt.show()

    # train(net)

    # checkpoint = torch.load('oscillator_limb_checkpoint.pt')
    # print(checkpoint['loss'])
    # net.load_state_dict(checkpoint['model_state_dict'])
    # plot_example(net)

