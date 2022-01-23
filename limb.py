import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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


# class TorchTwoLinkTorqueLimb:
#     def __init__(self):
#         self.com = [.025, .025]
#         self.mass = [.01, .0075]
#         self.I = [.0002, .00015]
#
#     def f(self, x, u):

class TwoLinkTorqueLimb:
    # https://www.youtube.com/watch?v=9ctGhk3cAas
    # https://ocw.mit.edu/courses/mechanical-engineering/2-12-introduction-to-robotics-fall-2005/lecture-notes/chapter7.pdf
    def __init__(self):
        self.m1 = 1
        self.m2 = 1
        self.lc1 = 1
        self.lc2 = 1
        self.l1 = 1
        self.I1 = 1
        self.I2 = 1

    def f(self, x, u):
        tau = u
        q1 = x[0]
        q2 = x[1]
        dq1 = x[2]
        dq2 = x[3]
        c1 = np.cos(q1)
        c2 = np.cos(q2)
        s1 = np.sin(q1)
        s2 = np.sin(q2)
        c12 = np.cos(q1+q2)
        g = 9.81

        # damping
        tau[0] = tau[0] - 3*dq1
        tau[1] = tau[1] - 3*dq2


        # soft range of motion limits
        def ligament_torque(stretch):
            return 2*(stretch/.1)**2

        limits1 = [-np.pi, 0]
        limits2 = [-np.pi/2, np.pi/2]
        if q1 < limits1[0]:
            tau[0] = tau[0] + ligament_torque(limits1[0]-q1)
        if q1 > limits1[1]:
            tau[0] = tau[0] - ligament_torque(q1-limits1[1])
        if q2 < limits2[0]:
            tau[1] = tau[1] + ligament_torque(limits2[0]-q2)
        if q2 > limits2[1]:
            tau[1] = tau[1] - ligament_torque(q2-limits2[1])

        H11 = self.m1*self.lc1**2 + self.I1 + self.m2*(self.l1**2 + self.lc2**2 + 2*self.l1*self.lc2*c2) + self.I2
        H22 = self.m2*self.lc2**2 + self.I2
        H12 = self.m2*(self.lc2**2 + self.l1*self.lc2*c2) + self.I2
        h = self.m2*self.l1*self.lc2*s2
        G1 = self.m1*self.lc1*g*(self.lc2*c12 + self.l1*c1)
        G2 = self.m2*g*self.lc2*c12

        H = np.array([[H11, H12], [0, H22]])
        # print(H)
        Hinv = np.linalg.inv(H)
        # print(Hinv)
        Hddq = tau + np.array([h*dq2**2 + 2*h*dq1*dq2 - G1, -h*dq1**2 - G2])
        # print(h)
        # print(G1)
        # print(dq2)
        # print(h*dq2**2 + 2*h*dq1*dq2 - G1)
        # print(Hddq)
        ddq = np.matmul(Hinv, Hddq)

        return np.array([dq1, dq2, ddq[0], ddq[1]])

        # T = u
        # q1 = x[0]
        # q2 = x[1]
        # dq1 = x[2]
        # dq2 = x[3]
        # c1 = np.cos(q1)
        # c2 = np.cos(q2)
        # s1 = np.sin(q1)
        # s2 = np.sin(q2)
        # c12 = np.cos(q1+q2)
        # s12 = np.sin(q1+q2)
        # g = 9.81
        # M = np.array([[3+2*c2, 1+c2], [1+c2, 1]])
        # C = np.array([2*s2*dq1*dq2 + s2*dq2*dq2, -s2*dq1*dq1])
        # G = g * np.array([2*s1+s12, s12])
        # ddQ = np.invert(M) * (T + C + G)

def simulate(model, dt, T):
    times = []
    states = []

    time = 0
    state = [0, 0, 0, 0]

    times.append(time)
    states.append(state)

    while time < T:
        u = np.zeros(2)
        # u = np.array([0.0005 * (time > .1), 0])
        derivative = model.f(state, u)

        time = time + dt
        state = state + dt*derivative

        times.append(time)
        states.append(state)

    figure, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    line1,  = ax.plot(0, 0)
    line2,  = ax.plot(0, 0)

    def animate_function(i):
        state = states[i*10]
        line1.set_xdata([0, 1*np.cos(state[0])])
        line1.set_ydata([0, 1*np.sin(state[0])])
        line2.set_xdata([1*np.cos(state[0]), 1*np.cos(state[0])+1*np.cos(state[0]+state[1])])
        line2.set_ydata([1*np.sin(state[0]), 1*np.sin(state[0])+1*np.sin(state[0]+state[1])])

    animation = FuncAnimation(figure,
                          func = animate_function,
                          frames = int(len(times)/10),
                          interval = dt)
    plt.show()

    times = np.array(times)
    states = np.array(states)

    plt.plot(times, states)
    plt.xlabel('Time (s)')
    plt.ylabel('State')
    plt.legend(('x1', 'x2', 'dx1', 'dx2'))
    plt.show()


# def simulate(model, dt, T):
#     times = []
#     states = []
#
#     time = 0
#     state = [0, 0]
#     state = torch.Tensor(state)
#
#     times.append(time)
#     states.append(state.numpy())
#
#     while time < T:
#         u = 0.0005 * (time > .1)
#         derivative = model.f(state, u)
#
#         time = time + dt
#         state = state + dt*derivative
#
#         # range of motion limits
#         if state[0] > np.pi / 3:
#             state[0] = np.pi / 3
#             state[1] = 0
#         if state[0] < -np.pi / 3:
#             state[0] = -np.pi / 3
#             state[1] = 0
#
#         times.append(time)
#         states.append(state.numpy())
#
#     times = np.array(times)
#     states = np.array(states)
#
#     plt.plot(times, states)
#     plt.xlabel('Time (s)')
#     plt.ylabel('State')
#     plt.legend(('x1', 'x2'))
#     plt.show()


if __name__ == '__main__':
    # model = OneLinkTorqueLimb()

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

    # model = TorchOneLinkTorqueLimb()
    # simulate(model, .01, 4)

    model = TwoLinkTorqueLimb()
    simulate(model, .005, 10)
