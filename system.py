import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from oscillator import Oscillator, TorchOscillator
from limb import TorchOneLinkTorqueLimb

#DONE: enforce 2D oscillator input - seems worse
#DONE: support minibatches
#DONE: try always starting at zero (should be easier)
#DONE: add constant / bias - should relate to target (bypass oscillators?)
#DONE: regularization
#TODO: learning rate higher after oscillator? (so it tends to exploit whatever basis is available)
#DONE: higher cost for velocity? or cost for torque?
#DONE: try setting initial state instead of input (like Sahani's results)
#TODO: remove second conv layer?
#TODO: save oscillator params - make oscillator a nn.Module and impl state_dict and load_state_dict
#TODO: Hill-type muscle model
#DONE: reduce torque cost (model consistently undershoots a lot)
#TODO: reduce weight regularization
#TODO: lower cost on velocity (or introduce later in simulation?)
#TODO: provide steady-state input to oscillators?

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.n_oscillators = 50
        self.n_per_oscillator = 40

        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, self.n_oscillators*self.n_per_oscillator)
        self.fc3 = nn.Linear(128, self.n_oscillators*2)

        self.fcdirect = nn.Linear(128, 1)
        self.fcdirect.weight = torch.nn.parameter.Parameter(self.fcdirect.weight / 100)
        self.fcdirect.bias = torch.nn.parameter.Parameter(self.fcdirect.bias * 0)

        self.oscillators = []
        # x1_encoders = []
        for i in range(self.n_oscillators):
            freq = 4*np.pi*np.random.rand()
            tau = .5*np.random.rand()
            o = Oscillator(self.n_per_oscillator, freq=freq, tau=tau)
            torch_oscillator = TorchOscillator(o.encoders, o.biases, o.decoders)
            self.oscillators.append(torch_oscillator)
            # x1_encoders.append(torch_oscillator.encoders[0,:])

        # x1_encoders = torch.cat(x1_encoders)

        self.fc4 = nn.Linear(self.n_oscillators * self.n_per_oscillator, 1)
        self.fc4.weight = torch.nn.parameter.Parameter(self.fc4.weight / 10000)
        self.fc4.bias = torch.nn.parameter.Parameter(self.fc4.bias * 0)

        self.limb = TorchOneLinkTorqueLimb()

        self.device = None

        # self.fcfoo = nn.Linear(self.n_oscillators * self.n_per_oscillator, 2)

    def forward(self, x, return_oscillator_state=False):
        """
        :param x: (current state, desired state)
        """

        batch_size = x.shape[0]
        current_state = x[:,0]
        # current_state = x[0]

        x = self.fc1(x)
        x = F.relu(x)
        direct = self.fcdirect(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        #TODO: use torch.split instead of torch.narrow?

        o_states = torch.zeros((self.n_oscillators, batch_size, 2))
        l_state = torch.zeros((batch_size, 2))
        l_state[:,0] = current_state
        # o_states = torch.zeros((self.n_oscillators, 2))
        # l_state = torch.zeros(2)
        # l_state[0] = current_state

        dt = 0.01
        steps = 100
        l_states = torch.zeros(steps, batch_size, 2)
        torques = torch.zeros(steps, batch_size)

        if self.device is not None:
            o_states = o_states.to(self.device)
            l_state = l_state.to(self.device)
            l_states = l_states.to(self.device)
            torques = torques.to(self.device)

        o_state_history = []

        if torch.mean(x).isnan():
           print(x)

        # input sets initial oscillator state
        for j in range(self.n_oscillators):
            xi = torch.narrow(x, 1, j * 2, 2)
            o_states[j,:,:] = xi

        for i in range(steps):
            # activities = []
            activities = torch.zeros((batch_size, self.n_oscillators*self.n_per_oscillator))
            if self.device is not None:
                activities = activities.to(self.device)

            for j in range(self.n_oscillators):
                # xi = torch.narrow(x, 0, j * self.n_per_oscillator, self.n_per_oscillator)
                # a = self.oscillators[j].get_activities(o_states[j,:], direct=xi)
                # xi = torch.narrow(x, 0, j * 2, 2)

                a = self.oscillators[j].get_activities(o_states[j,:,:])
                # print(a.shape)
                # activities.append(a)
                activities[:,j*self.n_per_oscillator:(j+1)*self.n_per_oscillator] = a
                # derivative = torch.matmul(self.oscillators[j].decoders, a)
                # # derivative = torch.matmul(self.oscillators[j].decoders, a) + xi
                derivative = torch.matmul(self.oscillators[j].decoders, a.T).T
                # derivative = self.oscillators[j].get_output(o_states[j,:,:]).T

                o_states[j,:,:] = o_states[j,:,:] + dt * derivative

            if return_oscillator_state:
                o_state_history.append(o_states.detach().cpu().numpy().copy())

            # activities = torch.cat(activities)
            # print(activities.shape)
            # print(activities)
            torque = (self.fc4(activities) + direct)
            # torque = direct

            torques[i,:] = torque.T

            derivative = self.limb.f(l_state, torque)
            l_state = l_state + dt*derivative

            # range of motion limits
            out_of_bounds = (l_state[:,0] > np.pi/3) + (l_state[:,0] > np.pi/3)
            # print(out_of_bounds) ########################

            # l_state[:,0] = torch.minimum(l_state[:,0], np.pi/3)
            # l_state[:,0] = torch.maximum(l_state[:,0], -np.pi/3)
            # if l_state[0] > np.pi/3:
            #     l_state[0] = np.pi/3
            #     l_state[1] = 0
            # if l_state[0] < -np.pi/3:
            #     l_state[0] = -np.pi/3
            #     l_state[1] = 0

            l_states[i,:,:] = l_state

        # return torch.cat((self.fcfoo(activities), self.fcfoo(activities))).reshape((2,2))
        # return torch.cat((l_state, l_state)).reshape((2,2))

        if return_oscillator_state:
            return l_states, torques, np.array(o_state_history)
        else:
            return l_states, torques


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

        # start = -np.pi/2 + np.pi*np.random.rand()
        # start = 0
        # target = -np.pi/2 + np.pi*np.random.rand()

        input = -np.pi/2 + np.pi*np.random.rand(batch_size, 2)
        input[:,0] = 0

        # input = torch.Tensor([start, target])
        input = torch.Tensor(input)
        if device is not None:
            input = input.to(device)

        optimizer.zero_grad()

        outputs, torques = net(input)

        # labels = torch.ones(outputs.shape)
        # labels[:,0] = labels[:,0] * target
        # labels[:,1] = labels[:,1] * 0 # aim for zero velocity
        labels = torch.zeros(outputs.shape)
        labels[:,:,0] = input[:,1]
        if device is not None:
            labels = labels.to(device)

        print('.', end='')
        # print(input.is_cuda)
        # print(labels.is_cuda)
        # print(outputs.is_cuda)
        # print(torques.is_cuda)
        loss = criterion(outputs, labels) + 1*torch.mean(torques.pow(2))

        # param_norm = sum(param.norm(2) for param in net.parameters())
        # loss = loss + 1e-6 * param_norm

        # print('start: {} target: {}'.format(start, target))
        # print('{} {}'.format(i, loss.item()))
        running_loss += loss.mean()
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

        loss.mean().backward() #note mean will kind of scale learning rate with batch size
        optimizer.step()


def plot_example(net, device=None):
    # start = -np.pi / 2 + np.pi * np.random.rand()
    start = 0
    target = -np.pi / 2 + np.pi * np.random.rand()
    print(target)

    input = torch.Tensor([[start, target]])
    if device is not None:
        input = input.to(device)
    # outputs = net(input).detach().numpy()
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
    plt.plot(time, torques)
    plt.subplot(133)
    print(o_states.shape)
    print(np.min(o_states.flatten()))
    for i in range(o_states.shape[1]):
        # print(time, o_states[:,i,0])
        plt.plot(time, o_states[:,i,0], '.')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    net = Net()
    # print(net)

    # current_state = 0.0
    # desired_state = 1.0
    # input = torch.tensor([current_state, desired_state])
    # l_states = net.forward(input)
    # trajectory = l_states.detach().numpy()
    #
    # plt.plot(trajectory)
    # plt.legend(('angle', 'velocity'))
    # plt.show()

    # train(net)

    checkpoint = torch.load('oscillator_limb_checkpoint.pt')
    print(checkpoint['loss'])
    net.load_state_dict(checkpoint['model_state_dict'])
    plot_example(net)


    #TODO: test with other random targets & plot
    #TODO: train in batches, move to cluster
