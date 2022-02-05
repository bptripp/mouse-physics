import numpy as np
import torch
import matplotlib.pyplot as plt

#TODO: note Medler 2002 re. muscle diversity across species
#TODO: clean up, rewrite in torch

"""
Winters, J. M. (1990). Hill-based muscle models: a systems engineering perspective. In Multiple muscle systems (pp. 69-93). 
Springer, New York, NY.

Chow, J. W., & Darling, W. G. (1999). The maximum shortening velocity of muscle should be scaled with activation. 
Journal of Applied Physiology, 86(3), 1025-1031.

Camilleri, M. J., & Hull, M. L. (2005). Are the maximum shortening velocity and the shape parameter in a Hill-type model of whole muscle 
related to activation?. Journal of biomechanics, 38(11), 2172-2180.

Zahalak, G. I., Duffy, J., Stewart, P. A., Litchman, H. M., Hawley, R. H., & Paslay, P. R. (1976). Partially activated human skeletal muscle: 
an experimental investigation of force, velocity, and EMG.
"""

class Muscle:
    def __init__(self, max_iso_force, muscle_rest_length, tendon_rest_length, max_velocity):
        self.max_iso_force = max_iso_force
        self.muscle_rest_length = muscle_rest_length
        self.tendon_rest_length = tendon_rest_length
        self.max_velocity = max_velocity

    def derivative(self, muscle_length, total_length, activation):
        tendon_length = total_length - muscle_length
        norm_tendon_length = tendon_length / self.tendon_rest_length
        norm_muscle_length = muscle_length / self.muscle_rest_length
        norm_tendon_force = force_length_series(norm_tendon_length)
        norm_parallel_force = force_length_parallel(muscle_length)
        norm_contractile_force = norm_tendon_force - norm_parallel_force
        force_velocity_factor = norm_contractile_force / activation / force_length_contractile(norm_muscle_length)
        norm_derivative = force_velocity_contractile_inverse(force_velocity_factor)
        activation_velocity_factor = .6 + .4 * activation
        return self.max_velocity * activation_velocity_factor * norm_derivative

    def force(self, muscle_length, total_length):
        tendon_length = total_length - muscle_length
        norm_tendon_length = tendon_length / self.tendon_rest_length
        return self.max_iso_force * force_length_series(norm_tendon_length)


class Activation:
    """
    Modified from Millard, M., Uchida, T., Seth, A., & Delp, S. L. (2013). Flexing computational muscle: modeling and
    simulation of musculotendon dynamics. Journal of biomechanical engineering, 135(2).
    """
    def __init__(self):
        self.a_min = .05
        self.tau_a = .01
        self.tau_d = .04

    def derivative(self, activation, excitation):
        excitation = torch.clip(excitation, self.a_min, 1)

        increasing = excitation > activation
        tau = increasing * self.tau_a * (.5 + 1.5*activation) + ~increasing * self.tau_d / (.5 + 1.5*activation)

        return (excitation - activation) / tau


def isometric_simulation():
    m = Muscle(100, .1, .05, .3)
    muscle_length = torch.tensor([.1, .1, .1])
    total_length = torch.tensor([.15, .15, .15])
    activation = torch.tensor([.33, .66, 1])
    batch_size = len(muscle_length)

    steps = 50
    times = []
    lengths = torch.zeros((batch_size, steps))
    forces = torch.zeros((batch_size, steps))
    t = 0
    dt = .005
    for i in range(steps):
        d = m.derivative(muscle_length, total_length, activation)
        muscle_length = muscle_length + dt*d
        t = t + dt
        times.append(t)
        lengths[:,i] = muscle_length
        forces[:,i] = m.force(muscle_length, total_length)

    print(total_length - muscle_length)

    plt.figure(figsize=(7,3))
    plt.subplot(121)
    plt.plot(times, lengths[2,:])
    plt.xlabel('Time (s)')
    plt.ylabel('CE Length')
    plt.subplot(122)
    plt.plot(times, forces[2,:])
    plt.xlabel('Time (s)')
    plt.ylabel('Force')
    plt.tight_layout()
    plt.show()


def activation_simulation():
    activation = Activation()
    a = torch.tensor([activation.a_min, activation.a_min]) # batch size 2

    times = []
    activations = torch.zeros((2,600))
    t = 0
    dt = .001

    for i in range(600):
        e = 1 if i < 300 else 0
        e = torch.tensor([.5 * e, e])
        d = activation.derivative(a, e)
        a = a + dt*d
        t = t + dt
        times.append(t)
        activations[:,i] = a

    plt.plot(times, activations[1,:])
    plt.xlabel('Time (s)')
    plt.ylabel('Activation')
    plt.ylim([0,1.1])
    plt.show()


def pendulum_simulation():
    link_com = .1
    link_mass = 1
    link_I = link_mass * link_com**2

    muscle_origin = torch.tensor([0, .01]) # global coords
    muscle_insertion = torch.tensor([0, -.2]) # link coords
    muscle_moment_arm = .01

    def get_total_length(angle):
        rotation = torch.zeros((len(angle), 2, 2))
        rotation[:,0,0] = torch.cos(angle)
        rotation[:,0,1] = -torch.sin(angle)
        rotation[:,1,0] = torch.sin(angle)
        rotation[:,1,1] = torch.cos(angle)
        # rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]])
        global_insertion = torch.matmul(rotation, muscle_insertion)
        difference = muscle_origin - global_insertion
        return torch.linalg.vector_norm(difference, dim=1)

    total_rest_length = get_total_length(torch.zeros(1))
    m = Muscle(100, .9*total_rest_length, .1*total_rest_length, 3*total_rest_length)
    a = Activation()

    batch_size = 3
    steps = 2000

    times = []
    lengths = torch.zeros(batch_size, steps)
    forces = torch.zeros(batch_size, steps)
    angles = torch.zeros(batch_size, steps)

    t = 0
    dt = .001

    angle = torch.zeros(batch_size)
    angular_velocity = torch.zeros(batch_size)
    muscle_length = m.muscle_rest_length
    activation = torch.tensor(0.1*np.ones(batch_size))
    excitation = torch.tensor([.25, .5, .75])
    # activation = .5

    for i in range(steps):
        t = t + dt
        total_length = get_total_length(angle)
        torque = muscle_moment_arm * m.force(muscle_length, total_length)
        # torque = .5

        torque = torque - .05*angular_velocity

        dactdt = a.derivative(activation, excitation)
        dmdt = m.derivative(muscle_length, total_length, activation)
        dvdt = (torque - link_mass * 9.81 * link_com * torch.sin(angle)) / link_I
        dadt = angular_velocity

        activation = activation + dt*dactdt
        muscle_length = muscle_length + dt*dmdt
        angular_velocity = angular_velocity + dt*dvdt
        angle = angle + dt*dadt

        times.append(t)
        lengths[:,i] = muscle_length
        forces[:,i] = torque/muscle_moment_arm
        angles[:,i] = angle

    print(lengths.size())

    plt.figure(figsize=(9,3))
    plt.subplot(131)
    plt.plot(times, torch.transpose(lengths, 0, 1))
    plt.xlabel('Time (s)')
    plt.ylabel('CE Length')
    plt.subplot(132)
    plt.plot(times, torch.transpose(forces, 0, 1))
    plt.xlabel('Time (s)')
    plt.ylabel('Force')
    plt.subplot(133)
    plt.plot(times, torch.transpose(angles, 0, 1))
    plt.xlabel('Time (s)')
    plt.ylabel('Angle')
    plt.tight_layout()
    plt.show()


def force_velocity_contractile(v):
    p0 = 1
    # b*p0/a = 1
    # (p+a)v=b(p-p0)
    # pv+av=bp-bp0
    # pv-pb=-av-bp0
    # p (v-b)  = -av - bp0
    # p = -(av+bp0) / (v-b)

    v = torch.clip(v, -1, .99)
    a = 1
    b = 1
    p = -(a*v+b*p0) / (v-b)

    #TODO: this may be too shallow for large lengthening velocities - could reduce 4 to 2
    # dpdv = (-a*(v-b) + (a*v+b*p0)) / (v-b)**2
    dpdv0 = (-a*(-b) + (b*p0)) / (-b)**2
    x = 4*dpdv0*v
    p2 = .5 + 1 / (1+torch.exp(-x))
    return torch.minimum(p, p2)
    # return p, p2


def force_velocity_contractile_inverse(p):
    p0 = 1
    a = 1
    b = 1
    p = torch.clip(p, 0, 1.4999)
    v1 = b*(p-p0)/(p+a)

    p2 = torch.clip(p - .5, .0001, .9999)
    x = torch.log(p2/(1-p2))
    dpdv0 = (-a * (-b) + (b * p0)) / (-b) ** 2
    v2 = x/4/dpdv0

    # return v1, v2
    return torch.maximum(v1, v2)


def force_length_contractile(l):
    # force = np.maximum(0, np.cos(np.pi*(l-1)))
    return torch.exp( - (l-1)**2 / (2*.3**2) )


def force_length_series(l):
    stretch = torch.relu(l-1)
    return 10*stretch + 200*stretch**2


def force_length_parallel(l):
    stretch = torch.relu(l - 1)
    return 3*stretch**2 / (.6 + stretch)

# logistic function: f(x) = 1/(1+e^-x)
# derivative: df(x)/dx = f(x) * (1-f(x))
# want derivative to match at x=0, where df(x)/dx = 1/4
# so if desired derivative is dp/dv, we want x=4(dp/dv)v
# dp/dv = (g'h -gh') / h^2 where g=-(av+bp0) and h=v-b
# g'=-a h'=1
# dp/dv = ( -a(v-b) + (av+bp0) ) / (v-b)^2


def plot_curves():
    plt.figure(figsize=(11,4))

    plt.subplot(131)
    length = torch.linspace(0, 2, 200)
    force = force_length_contractile(length)
    plt.plot(length, force)
    force = force_length_series(length)
    plt.plot(length, force)
    force = force_length_parallel(length)
    plt.plot(length, force)
    plt.legend(('contractile', 'series', 'parallel'))
    plt.xlabel('Normalized length')
    plt.ylabel('Normalized force')
    plt.ylim([0, 1.6])

    plt.subplot(132)
    velocity = torch.linspace(-1.2, 1, 100)
    force = force_velocity_contractile(velocity)
    plt.plot(velocity, force, 'k')
    plt.xlabel('Normalized velocity')
    plt.ylabel('Normalized force')
    plt.ylim([0, 1.6])

    plt.subplot(133)
    plt.plot(force, velocity, 'k.')
    force = torch.linspace(0, 1.5, 100)
    velocity = force_velocity_contractile_inverse(force)
    plt.plot(force, velocity, 'k')
    plt.ylim([-1,1])
    plt.xlabel('Normalized force')
    plt.ylabel('Normalized velocity')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_curves()
    # isometric_simulation()
    # activation_simulation()
    pendulum_simulation()
