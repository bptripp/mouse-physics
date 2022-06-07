import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from muscle import Muscle

import time

array_type = torch.float32

class TorchOneLinkTorqueLimb:
    def __init__(self):
        self.com = .025
        self.mass = .01
        self.I = .0002

        self.device = None

    def f(self, x, u):
        """
        :param x: state (angle, angular velocity)
        :param u: torque input
        :return: temporal derivative of state
        """
        batch_size = len(u)
        derivative = torch.zeros(batch_size, 2, dtype=array_type)

        if self.device is not None:
            derivative = derivative.to(self.device)

        # note u[:,0] needed here to avoid adding 3x1 tensor to 3 tensor, resulting in 3x3
        derivative[:,0] = x[:,1]
        derivative[:,1] = (u[:,0] - self.mass * 9.81 * self.com * torch.sin(x[:,0])) / self.I

        return derivative


class TorchOneLinkMuscleLimb:
    def __init__(self):
        self.com = .025
        self.mass = .01
        self.I = .0002

        self.m1_origin = torch.tensor([.002, 0], dtype=array_type)  # global coords
        self.m1_insertion = torch.tensor([0, -.025], dtype=array_type)  # link coords
        self.m1_moment_arm = .002

        self.m2_origin = torch.tensor([-.002, 0], dtype=array_type)  # global coords
        self.m2_insertion = torch.tensor([0, -.025], dtype=array_type)  # link coords
        self.m2_moment_arm = -.002

        self.resting_angle = torch.zeros(1, dtype=array_type)

        max_iso_force = 8
        m1_rest_length = get_muscle_tendon_length(torch.zeros(1, dtype=array_type), self.m1_origin, self.m1_insertion)
        self.m1 = Muscle(max_iso_force, .9 * m1_rest_length, .1 * m1_rest_length, 3 * m1_rest_length)
        m2_rest_length = get_muscle_tendon_length(torch.zeros(1, dtype=array_type), self.m1_origin, self.m1_insertion)
        self.m2 = Muscle(max_iso_force, .9 * m2_rest_length, .1 * m2_rest_length, 3 * m2_rest_length)

        self.device = None

    def set_device(self, device):
        self.device = device
        self.m1_origin = self.m1_origin.to(device)
        self.m1_insertion = self.m1_insertion.to(device)
        self.m2_origin = self.m2_origin.to(device)
        self.m2_insertion = self.m2_insertion.to(device)
        self.resting_angle = self.resting_angle.to(device)

    def get_m1_length(self, angle):
        return get_muscle_tendon_length(angle, self.m1_origin, self.m1_insertion, self.device)

    def get_m2_length(self, angle):
        return get_muscle_tendon_length(angle, self.m2_origin, self.m2_insertion, self.device)

    def f(self, x, u):
        """
        :param x: state (angle, angular velocity, m1_contractile_length, m2_contractile_length)
        :param u: input (m1_activation, m2_activation)
        :return: temporal derivative of state
        """
        batch_size = len(u)
        derivative = torch.zeros(batch_size, 4, dtype=array_type)

        if self.device is not None:
            derivative = derivative.to(self.device)

        m1_contractile_length = x[:,2]
        m2_contractile_length = x[:,3]
        m1_length = self.get_m1_length(x[:,0])
        m2_length = self.get_m2_length(x[:,0])

        # print(m1_contractile_length.device)
        # print(m1_length.device)

        m1_torque = self.m1_moment_arm * self.m1.force(m1_contractile_length, m1_length)
        m2_torque = self.m2_moment_arm * self.m2.force(m2_contractile_length, m2_length)
        torque = m1_torque + m2_torque

        torque = torque - .0005*x[:,1]

        # note u[:,0] needed here to avoid adding 3x1 tensor to 3 tensor, resulting in 3x3
        derivative[:,0] = x[:,1]
        derivative[:,1] = (torque - self.mass * 9.81 * self.com * torch.sin(x[:,0])) / self.I
        derivative[:,2] = self.m1.derivative(m1_contractile_length, m1_length, u[:,0])
        derivative[:,3] = self.m2.derivative(m2_contractile_length, m2_length, u[:,1])

        return derivative


def get_muscle_tendon_length(angle, global_origin, local_insertion, device=None):
    rotation = torch.zeros((len(angle), 2, 2), dtype=array_type)
    if device is not None:
        rotation = rotation.to(device)

    rotation[:,0,0] = torch.cos(angle)
    rotation[:,0,1] = -torch.sin(angle)
    rotation[:,1,0] = torch.sin(angle)
    rotation[:,1,1] = torch.cos(angle)
    global_insertion = torch.matmul(rotation, local_insertion)
    difference = global_origin - global_insertion
    return torch.linalg.vector_norm(difference, dim=1)


class OneLinkTorqueLimb:
    def __init__(self):
        self.com = .025
        self.mass = .01
        self.I = .0002

    def f(self, x, u):
        """
        :param x: [angle from vertical (rad), angular velocity]
        :param u: input vector
        :return: temporal derivative of state
        """
        dx1 = x[1]
        dx2 = (u - self.mass * 9.81 * self.com * np.sin(x[0])) / self.I
        # print('{} {} {} {}'.format(dx1, dx2, x, u))

        return np.array([dx1, dx2])


class TorchTwoLinkTorqueLimb:
    def __init__(self):
        self.m1 = .01
        self.m2 = .01
        self.lc1 = .025
        self.lc2 = .025
        self.l1 = .05
        self.l2 = .05
        self.I1 = .0002
        self.I2 = .0002

        self.device = None

    def f(self, x, u):
        """
        :param x: state (angle 1st link from horizontal; angle 2nd link from 1st; ang velocity 1; ang velocity 2)
        :param u: input (torque on 1st joint; torque on 2nd joint)
        :return: state derivative
        """
        batch_size = len(u)
        derivative = torch.zeros(batch_size, 4)

        if self.device is not None:
            derivative = derivative.to(self.device)

        tau = u.clone()
        q1 = x[:,0]
        q2 = x[:,1]
        dq1 = x[:,2]
        dq2 = x[:,3]
        c1 = torch.cos(q1)
        c2 = torch.cos(q2)
        s2 = torch.sin(q2)
        c12 = torch.cos(q1+q2)
        g = 9.81

        # damping
        tau[:,0] = tau[:,0] - .0003*dq1
        tau[:,1] = tau[:,1] - .0003*dq2

        # soft range of motion limits
        def ligament_torque(stretch):
            return (stretch > 0) * .0001*(stretch/.05)**2

        limits1 = [-np.pi, 0]
        limits2 = [-np.pi/2, np.pi/2]
        tau[:,0] = tau[:,0] + ligament_torque(limits1[0]-q1)
        tau[:,0] = tau[:,0] - ligament_torque(q1-limits1[1])
        tau[:,1] = tau[:,1] + ligament_torque(limits2[0]-q2)
        tau[:,1] = tau[:,1] - ligament_torque(q2-limits2[1])

        H11 = self.m1*self.lc1**2 + self.I1 + self.m2*(self.l1**2 + self.lc2**2 + 2*self.l1*self.lc2*c2) + self.I2
        H22 = self.m2*self.lc2**2 + self.I2
        H12 = self.m2*(self.lc2**2 + self.l1*self.lc2*c2) + self.I2
        h = self.m2*self.l1*self.lc2*s2
        G1 = self.m1*self.lc1*g*c1 + self.m2*g*(self.lc2*c12 + self.l1*c1)
        G2 = self.m2*g*self.lc2*c12

        H = torch.zeros((batch_size, 2, 2))
        H[:,0,0] = H11
        H[:,0,1] = H12
        H[:,1,1] = H22
        Hinv = torch.linalg.inv(H)

        Hddq = torch.zeros((batch_size,2,1))
        Hddq[:,0,0] = h*dq2**2 + 2*h*dq1*dq2 - G1
        Hddq[:,1,0] = -h*dq1**2 - G2
        Hddq[:,:,0] = Hddq[:,:,0] + tau

        ddq = torch.matmul(Hinv, Hddq)

        result = torch.zeros((batch_size, 4))
        result[:,0] = dq1
        result[:,1] = dq2
        result[:,2:] = ddq[:,:,0]
        return result


class TwoLinkTorqueLimb:
    # https://ocw.mit.edu/courses/mechanical-engineering/2-12-introduction-to-robotics-fall-2005/lecture-notes/chapter7.pdf
    def __init__(self):
        self.m1 = .01
        self.m2 = .01
        self.lc1 = .025
        self.lc2 = .025
        self.l1 = .05
        self.l2 = .05
        self.I1 = .0002
        self.I2 = .0002

    def f(self, x, u):
        """
        :param x: state (angle 1st link from horizontal; angle 2nd link from 1st; ang velocity 1; ang velocity 2)
        :param u: input (torque on 1st joint; torque on 2nd joint)
        :return: state derivative
        """
        tau = np.zeros(2)
        tau[:] = u[:]
        q1 = x[0]
        q2 = x[1]
        dq1 = x[2]
        dq2 = x[3]
        c1 = np.cos(q1)
        c2 = np.cos(q2)
        s2 = np.sin(q2)
        c12 = np.cos(q1+q2)
        g = 9.81

        # damping
        tau[0] = tau[0] - .0003*dq1
        tau[1] = tau[1] - .0003*dq2

        # soft range of motion limits
        def ligament_torque(stretch):
            return (stretch > 0) * .0001*(stretch/.05)**2

        limits1 = [-np.pi, 0]
        limits2 = [-np.pi/2, np.pi/2]
        tau[0] = tau[0] + ligament_torque(limits1[0]-q1)
        tau[0] = tau[0] - ligament_torque(q1-limits1[1])
        tau[1] = tau[1] + ligament_torque(limits2[0]-q2)
        tau[1] = tau[1] - ligament_torque(q2-limits2[1])

        H11 = self.m1*self.lc1**2 + self.I1 + self.m2*(self.l1**2 + self.lc2**2 + 2*self.l1*self.lc2*c2) + self.I2
        H22 = self.m2*self.lc2**2 + self.I2
        H12 = self.m2*(self.lc2**2 + self.l1*self.lc2*c2) + self.I2
        h = self.m2*self.l1*self.lc2*s2
        G1 = self.m1*self.lc1*g*c1 + self.m2*g*(self.lc2*c12 + self.l1*c1)
        G2 = self.m2*g*self.lc2*c12

        H = np.array([[H11, H12], [0, H22]])
        Hinv = np.linalg.inv(H)
        Hddq = tau + np.array([h*dq2**2 + 2*h*dq1*dq2 - G1, -h*dq1**2 - G2])
        ddq = np.matmul(Hinv, Hddq)

        return np.array([dq1, dq2, ddq[0], ddq[1]])


def simulate_two_link(model, dt, T, batch_size=3):
    times = []
    states = []

    time = 0
    if type(model) == TorchTwoLinkTorqueLimb:
        state = torch.zeros((batch_size, 4))
        u = torch.zeros((batch_size, 2))
    else:
        state = np.zeros(4)
        u = np.zeros(2)

    def record(time, state):
        times.append(time)
        if type(model) == TorchTwoLinkTorqueLimb:
            states.append(state.numpy())
        else:
            states.append(state)

    record(time, state)

    while time < T:
        derivative = model.f(state, u)

        time = time + dt
        state = state + dt*derivative

        record(time, state)

    times = np.array(times)
    states = np.array(states)

    if type(model) == TorchTwoLinkTorqueLimb:
        states = states[:,0,:]

    figure, ax = plt.subplots()
    ax.set_xlim(-2*model.l1, 2*model.l1)
    ax.set_ylim(-2*model.l1, 2*model.l1)
    line1,  = ax.plot(0, 0)
    line2,  = ax.plot(0, 0)

    plot_interval = 25
    def animate_function(i):
        state = states[i*plot_interval]
        line1.set_xdata([0, model.l1*np.cos(state[0])])
        line1.set_ydata([0, model.l1*np.sin(state[0])])
        line2.set_xdata([model.l1*np.cos(state[0]), model.l1*np.cos(state[0])+model.l2*np.cos(state[0]+state[1])])
        line2.set_ydata([model.l1*np.sin(state[0]), model.l1*np.sin(state[0])+model.l2*np.sin(state[0]+state[1])])

    animation = FuncAnimation(figure,
                          func = animate_function,
                          frames = int(len(times)/plot_interval),
                          interval = dt)
    plt.show()

    plt.plot(times, states)
    plt.xlabel('Time (s)')
    plt.ylabel('State')
    plt.legend(('x1', 'x2', 'dx1', 'dx2'))
    plt.show()


def simulate_one_link(model, dt, T, batch_size=1, method='rk4'):
    """
    Runs an example simulation with constant input

    :param model: one-link limb model
    :param dt: time step size
    :param T: duration of simulation
    :param batch_size: number of simulations to run in parallel
    """
    times = []
    states = []

    time = 0
    # state = np.zeros((batch_size,2))
    state = np.zeros((batch_size,4), dtype=np.float64)
    state[:,2] = model.m1.muscle_rest_length
    state[:,3] = model.m2.muscle_rest_length
    state = torch.Tensor(state)

    times.append(time)
    states.append(state.numpy())

    if method=='rk4':
        from torchdiffeq import odeint

        u = torch.zeros((batch_size, 2), dtype=array_type)
        u[:,0] = 0
        u[:,1] = .2
        def partial_func(t,s,u0=u):
            return model.f(s,u0)

        print(state)
        times = torch.arange(0,T+2*dt,step=dt)
        # known good: bosh3, adaptive_heun
        states = odeint(partial_func, state, times, method='bosh3',atol=1e-4)

        return times.numpy(), states.numpy()
    ### end if

    while time < T:
        # :param x: state (angle, angular velocity, m1_contractile_length, m2_contractile_length)
        # :param u: input (m1_activation, m2_activation)

        # u = 0.0005 * (time > .1) * np.ones((batch_size,1))
        u = torch.zeros((batch_size, 2), dtype=array_type)
        u[:,0] = 0
        u[:,1] = .2

        # euler
        if method == 'euler':
            derivative = model.f(state, u)
        # explicit trapezoid
        elif method == 'explicit-trap':
            s1 = model.f(state, u)
            s2 = model.f(state + dt*s1, u)
            derivative = (s1 + s2) / 2
        # # RK4
        elif method == 'rk4':
            s1 = model.f(state, u)
            s2 = model.f(state + dt/2*s1, u)
            s3 = model.f(state + dt/2*s2, u)
            s4 = model.f(state + dt*s3, u)
            derivative = 1/6 * (2 * s1 + s2 + s3 + 2 * s4)
        else:
            raise RuntimeWarning(f'Unrecognized integration method {method}')

        time = time + dt
        state = state + dt*derivative

        # range of motion limits
        for i in range(batch_size):
            if state[i,0] > np.pi / 3:
                state[i,0] = np.pi / 3
                state[i,1] = 0
            if state[i,0] < -np.pi / 3:
                state[i,0] = -np.pi / 3
                state[i,1] = 0

        times.append(time)
        states.append(state.numpy())

    times = np.array(times)
    states = np.array(states)

    return times, states

def plot_one_link_results(times, states):
    plt.plot(times, states.squeeze())


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
    model = TorchOneLinkMuscleLimb()

    slow_dt=0.1
    eu_001_start = time.time_ns()
    times_eu_001, states_eu_001 = simulate_one_link(model, .001, 4, method='euler')
    eu_001_end = time.time_ns() - eu_001_start

    eu_01_start = time.time_ns()
    times_eu_01, states_eu_01 = simulate_one_link(model, slow_dt, 4, method='euler')
    eu_01_end = time.time_ns() - eu_01_start


    rk_001_start = time.time_ns()
    times_rk_001, states_rk_001 = simulate_one_link(model, .001, 4)
    rk_001_end = time.time_ns() - rk_001_start

    rk_01_start = time.time_ns()
    times_rk_01, states_rk_01 = simulate_one_link(model, slow_dt, 4)
    rk_01_end = time.time_ns() - rk_01_start

    print(eu_001_end * 1e-9, eu_01_end * 1e-9)
    print(rk_001_end * 1e-9, rk_01_end * 1e-9)
#     exit()

    def rmse(x,y):
        return np.sqrt(np.mean(np.power(x-y,2)))

#     print('RK 0.001 vs Euler 0.001: ', rmse(states_eu_001, states_rk_001))
#     print('RK 0.01 vs Euler 0.001: ', rmse((states_eu_001[::10]), states_rk_01[:-1]))
#     print('RK 0.01 vs RK 0.001: ', rmse((states_rk_001[::10]), states_rk_01[:-1]))


    times_et_001, states_et_001 = simulate_one_link(model, .001, 4, method='explicit-trap')
    times_et_01, states_et_01 = simulate_one_link(model, slow_dt, 4, method='explicit-trap')


    plt.figure()
    plt.subplot(2,1,1)
    plot_one_link_results(times_rk_001, states_rk_001)
    plt.subplot(2,1,2)
    plot_one_link_results(times_rk_01, states_rk_01)


    plt.figure()
    plt.subplot(2,3,1)
    plot_one_link_results(times_eu_001, states_eu_001)
#     plt.xlabel('Time (s)')
    plt.legend(('Angle', 'Ang Vel', 'M1 Length', 'M2 Length'))
    plt.ylabel('State, dt=0.001')
    plt.title('Euler')

    plt.subplot(2,3,2)
    plot_one_link_results(times_rk_001, states_rk_001)
    plt.title('Runge-Kutta 4')

    plt.subplot(2,3,3)
    plot_one_link_results(times_et_001, states_et_001)
    plt.title('Explicit Trap')

    plt.subplot(2,3,4)
    plot_one_link_results(times_eu_01, states_eu_01)
    plt.xlabel('Time (s)')
    plt.ylabel('State, dt=0.1')

    plt.subplot(2,3,5)
    plot_one_link_results(times_rk_01, states_rk_01)
    plt.xlabel('Time (s)')

    plt.subplot(2,3,6)
    plot_one_link_results(times_et_01, states_et_01)
    plt.xlabel('Time (s)')

    plt.show()

    # angles = torch.Tensor(np.linspace(-np.pi, np.pi, 100))
    # m1 = model.get_m1_length(angles)
    # m2 = model.get_m2_length(angles)
    # plt.plot(angles, m1)
    # plt.plot(angles, m2)
    # plt.tight_layout()
    # plt.show()

    # model = TwoLinkTorqueLimb()
    # model = TorchTwoLinkTorqueLimb()
    # simulate_two_link(model, .001, 4)


