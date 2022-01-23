import numpy as np
import torch
import matplotlib.pyplot as plt

#TODO: note Chow & Darling 1999 re scaling of max velocity with activation (see also Winters, 1990)
#TODO: note Medler 2002 re. muscle diversity across species
#TODO: clean up, rewrite in torch
#TODO: run system with variable starting points
#TODO: incorporate muscles into system

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
        return self.max_velocity * norm_derivative

    def force(self, muscle_length, total_length):
        tendon_length = total_length - muscle_length
        norm_tendon_length = tendon_length / self.tendon_rest_length
        return self.max_iso_force * force_length_series(norm_tendon_length)


def isometric_simulation():
    m = Muscle(100, .1, .05, .3)
    muscle_length = .1
    total_length = .15

    times = []
    lengths = []
    forces = []
    t = 0
    dt = .005
    for i in range(50):
        d = m.derivative(muscle_length, total_length, 1)
        muscle_length = muscle_length + dt*d
        t = t + dt
        times.append(t)
        lengths.append(muscle_length)
        forces.append(m.force(muscle_length, total_length))

    print(total_length - muscle_length)

    plt.figure(figsize=(7,3))
    plt.subplot(121)
    plt.plot(times, lengths)
    plt.xlabel('Time (s)')
    plt.ylabel('CE Length')
    plt.subplot(122)
    plt.plot(times, forces)
    plt.xlabel('Time (s)')
    plt.ylabel('Force')
    plt.tight_layout()
    plt.show()


def pendulum_simulation():
    link_com = .1
    link_mass = 1
    link_I = link_mass * link_com**2

    muscle_origin = np.array([0, .01]) # global coords
    muscle_insertion = np.array([0, -.2]) # link coords
    muscle_moment_arm = .01 # TODO: let moment arm vary?

    def get_total_length(angle):
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        global_insertion = np.matmul(rotation, muscle_insertion)
        return np.linalg.norm(muscle_origin - global_insertion)

    total_rest_length = get_total_length(0)
    m = Muscle(100, .95*total_rest_length, .05*total_rest_length, 3*total_rest_length)

    times = []
    lengths = []
    forces = []
    angles = []

    t = 0
    dt = .0005

    angle = 0
    angular_velocity = 0
    muscle_length = m.muscle_rest_length
    activation = .5

    for i in range(20000):
        t = t + dt
        total_length = get_total_length(angle)
        torque = muscle_moment_arm * m.force(muscle_length, total_length)
        # torque = .5

        dmdt = m.derivative(muscle_length, total_length, activation)
        dvdt = (torque - link_mass * 9.81 * link_com * np.sin(angle)) / link_I
        dadt = angular_velocity

        muscle_length = muscle_length + dt*dmdt
        angular_velocity = angular_velocity + dt*dvdt
        angle = angle + dt*dadt

        times.append(t)
        lengths.append(muscle_length)
        forces.append(torque/muscle_moment_arm)
        angles.append(angle)

    plt.figure(figsize=(9,3))
    plt.subplot(131)
    plt.plot(times, lengths)
    plt.xlabel('Time (s)')
    plt.ylabel('CE Length')
    plt.subplot(132)
    plt.plot(times, forces)
    plt.xlabel('Time (s)')
    plt.ylabel('Force')
    plt.subplot(133)
    plt.plot(times, angles)
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

    v = np.clip(v, -1, .99)
    a = 1
    b = 1
    p = -(a*v+b*p0) / (v-b)

    #TODO: this may be too shallow for large lengthening velocities - could reduce 4 to 2
    # dpdv = (-a*(v-b) + (a*v+b*p0)) / (v-b)**2
    dpdv0 = (-a*(-b) + (b*p0)) / (-b)**2
    # x = 4*dpdv*v
    x = 4*dpdv0*v
    # p2 = 0.5 + 1 / (1+np.exp(-x))
    p2 = .5 + 1 / (1+np.exp(-x))
    return np.minimum(p, p2)
    # return p, p2


def simulate_isometric():
    pass

def force_velocity_contractile_inverse(p):
    p0 = 1
    a = 1
    b = 1
    p = np.clip(p, 0, 1.4999)
    v1 = b*(p-p0)/(p+a)

    p2 = np.clip(p - .5, .0001, .9999)
    x = np.log(p2/(1-p2))
    dpdv0 = (-a * (-b) + (b * p0)) / (-b) ** 2
    v2 = x/4/dpdv0

    # return v1, v2
    return np.maximum(v1, v2)


def force_length_contractile(l):
    # force = np.maximum(0, np.cos(np.pi*(l-1)))
    return np.exp( - (l-1)**2 / (2*.3**2) )


def force_length_series(l):
    stretch = np.maximum(0, l-1)
    return 10*stretch + 200*stretch**2


def force_length_parallel(l):
    stretch = np.maximum(0, l - 1)
    return 3*stretch**2 / (.6 + stretch)

# logistic function: f(x) = 1/(1+e^-x)
# derivative: df(x)/dx = f(x) * (1-f(x))
# want derivative to match at x=0, where df(x)/dx = 1/4
# so if desired derivative is dp/dv, we want x=4(dp/dv)v
# dp/dv = (g'h -gh') / h^2 where g=-(av+bp0) and h=v-b
# g'=-a h'=1
# dp/dv = ( -a(v-b) + (av+bp0) ) / (v-b)^2


def plot_curves():
    # velocity = np.linspace(-1, 1, 50)
    # force = force_velocity_contractile(velocity)
    # plt.plot(velocity, force)
    # # plt.plot(velocity, foo)
    # plt.ylim(0, 2)
    # plt.show()

    plt.figure(figsize=(11,4))

    plt.subplot(131)
    length = np.linspace(0, 2, 200)
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
    velocity = np.linspace(-1.2, 1, 100)
    force = force_velocity_contractile(velocity)
    plt.plot(velocity, force, 'k')
    plt.xlabel('Normalized velocity')
    plt.ylabel('Normalized force')
    plt.ylim([0, 1.6])

    plt.subplot(133)
    plt.plot(force, velocity, 'k.')
    force = np.linspace(0, 1.5, 100)
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
    pendulum_simulation()
