import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Oscillator:
    def __init__(self, n_neurons, freq=2*np.pi, tau=.5):
        self.n_neurons = n_neurons
        self.encoders = np.random.randn(2, n_neurons)
        self.encoders = self.encoders / np.linalg.norm(self.encoders, axis=0)
        self.biases = -1 + 2*np.random.rand(n_neurons)

        self.decoders = self.get_decoders(freq=freq, tau=tau)

    def get_activities(self, x):
        drive = np.matmul(x, self.encoders) + self.biases
        return np.maximum(0, drive)

    def get_decoders(self, freq=2*np.pi, tau=.5):
        """
        :return: decoding weights for state derivatives
        """
        n = 100
        eval_points = np.random.randn(2,n)
        eval_points = eval_points / np.linalg.norm(eval_points, axis=0)
        lengths = np.random.rand(n)**.5
        eval_points = eval_points * lengths

        responses = []
        for x in eval_points.T:
            responses.append(self.get_activities(x))

        responses = np.array(responses)
        dx1_decoders = np.linalg.lstsq(responses, -(1/tau)*eval_points[0,:]-freq*eval_points[1,:], rcond=.001)[0]
        dx2_decoders = np.linalg.lstsq(responses, freq*eval_points[0,:]-(1/tau)*eval_points[1,:], rcond=.001)[0]

        return np.array((dx1_decoders, dx2_decoders))

    def get_output(self, x):
        activities = self.get_activities(x)
        return np.matmul(self.decoders, activities)


class TorchOscillator:
    def __init__(self, encoders, biases, decoders):
        self.encoders = torch.Tensor(encoders)
        self.biases = torch.Tensor(biases)
        self.decoders = torch.Tensor(decoders)

    def get_activities(self, x, direct=None):
        drive = torch.matmul(x, self.encoders) + self.biases
        if direct is not None:
            drive = drive + direct
        return F.relu(drive)

    def get_output(self, x, direct=None):
        activities = self.get_activities(x, direct=direct)
        return torch.matmul(self.decoders, activities.T)


def visualize_outputs(oscillator):
    states = []
    outputs = []
    for i in range(200):
        state = -1 + 2*np.random.rand(2)
        output = o.get_output(state)
        states.append(state)
        outputs.append(output)

    states = np.array(states)
    outputs = np.array(outputs)

    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(states[:,0], states[:,1], outputs[:,0], marker='.')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('dx1/dt')
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(states[:,0], states[:,1], outputs[:,1], marker='.')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('dx2/dt')
    plt.show()


def simulate(oscillator, dt, T):
    times = []
    states = []

    time = 0
    state = [0, 0]

    torch_oscillator = TorchOscillator(oscillator.encoders, oscillator.biases, oscillator.decoders)
    state = torch.Tensor(state)

    times.append(time)
    states.append(state.numpy())

    while time < T:
        # derivative = oscillator.get_output(state)
        derivative = torch_oscillator.get_output(state)

        # assume input of [1, 0]
        derivative[0] = derivative[0] + 1

        time = time + dt
        state = state + dt*derivative

        times.append(time)
        states.append(state.numpy())

    print(states)

    times = np.array(times)
    states = np.array(states)

    plt.plot(times, states)
    plt.show()


if __name__ == '__main__':
    # o = Oscillator(50)
    # # visualize_outputs(o)
    # simulate(o, .01, 2*np.pi)

    freq = 1 * np.pi
    tau = 0.25
    o = Oscillator(50, freq=freq, tau=tau)
    # simulate(o, .01, 1)

    torch_oscillator = TorchOscillator(o.encoders, o.biases, o.decoders)
    # output = torch_oscillator.get_output(torch.zeros(2))
    output = torch_oscillator.get_output(torch.zeros(10,2))
    print(output)

