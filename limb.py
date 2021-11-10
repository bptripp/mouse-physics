import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class TorchOneLinkTorqueLimb:
    def __init__(self):
        self.com = .025
        self.mass = .01
        self.I = .0002

        self.device = None

        # self.derivative = torch.zeros(2) # this was screwing up grad on second pass

    def f(self, x, u):
        """
        :param x: state (angle, angular velocity)
        :param u: torque input
        :return: temporal derivative of state
        """
        batch_size = len(u)
        derivative = torch.zeros(batch_size, 2)

        if self.device is not None:
            derivative = derivative.to(self.device)

        # note u[:,0] needed here to avoid adding 3x1 tensor to 3 tensor, resulting in 3x3
        derivative[:,0] = x[:,1]
        derivative[:,1] = (u[:,0] - self.mass * 9.81 * self.com * torch.sin(x[:,0])) / self.I
        # print('{} {} {} {}'.format(dx1, dx2, x, u))

        return derivative


class OneLinkTorqueLimb:
    def __init__(self):
        self.com = .025
        self.mass = .01
        self.I = .0002

        # self.x1 = 0 # angle from vertical
        # self.x2 = 0 # angular velocity

    def f(self, x, u):
        """
        :param u: input vector
        :return: temporal derivative of state
        """
        dx1 = x[1]
        dx2 = (u - self.mass * 9.81 * self.com * np.sin(x[0])) / self.I
        # print('{} {} {} {}'.format(dx1, dx2, x, u))

        return np.array([dx1, dx2])


def simulate(model, dt, T):
    times = []
    states = []

    time = 0
    state = [0, 0]
    state = torch.Tensor(state)

    times.append(time)
    states.append(state.numpy())

    while time < T:
        u = 0.0005 * (time > .1)
        derivative = model.f(state, u)

        time = time + dt
        state = state + dt*derivative

        # range of motion limits
        if state[0] > np.pi / 3:
            state[0] = np.pi / 3
            state[1] = 0
        if state[0] < -np.pi / 3:
            state[0] = -np.pi / 3
            state[1] = 0

        times.append(time)
        states.append(state.numpy())

    times = np.array(times)
    states = np.array(states)

    plt.plot(times, states)
    plt.xlabel('Time (s)')
    plt.ylabel('State')
    plt.legend(('x1', 'x2'))
    plt.show()


if __name__ == '__main__':
    model = OneLinkTorqueLimb()

    # def fun(t, y):
    #     u = 0.002 * (t>1)
    #     return model.f(y, u)
    #
    # solution = solve_ivp(fun, [0, 2], [0, 0], max_step=.01)
    #
    # plt.plot(solution.t, solution.y.T)
    # plt.xlabel('Time (s)')
    # plt.ylabel('State')
    # plt.legend(('x1', 'x2'))
    # plt.show()

    model = TorchOneLinkTorqueLimb()
    simulate(model, .01, 4)
