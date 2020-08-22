# explict fem implement
# taichi version 0.6.27
import taichi as ti
import math

bg_color = 0x112f41
line_color = 0xffb169
boundary_color = 0xebaca2

screen_res = (400, 400)
screen_to_world_ratio = 40.0
boundary = [[0, screen_res[0] / screen_to_world_ratio],
            [0, screen_res[1] / screen_to_world_ratio]]

@ti.data_oriented
class Object:
    def __init__(self, filename):

        self.dim = 2
        self.inf = 1e10
        self.epsilon = 1e-5

        self.v = []
        self.f = []
        self.e = []
        # read nodes from *.node file
        with open(filename+".node", "r") as file:
            vn = int(file.readline().split()[0])
            for i in range(vn):
                self.v += [float(x) for x in file.readline().split()[1:self.dim+1]]  # [x,y] or [x,y,z]

        # read elements from *.ele file
        with open(filename+".ele", "r") as file:
            en = int(file.readline().split()[0])
            for i in range(en):  # index is 0-based
                self.e += [int(ind) for ind in file.readline().split()[1:self.dim+2]]  # triangle or tetrahedron

        # print(self.v)
        # print(self.e)

        self.vn = int(len(self.v)/self.dim)
        self.en = int(len(self.e)/(self.dim+1))

        self.node = ti.Vector(self.dim, dt=ti.f32, shape=self.vn, needs_grad=True)
        self.element = ti.Vector(self.dim+1, dt=ti.i32, shape=self.en)
        self.center = ti.Vector(self.dim, ti.f32, shape=())

        ## for simulation
        self.E = 1000  # Young modulus
        self.nu = 0.3  # Poisson's ratio: nu \in [0, 0.5)
        self.mu = self.E / (2 * (1 + self.nu))
        self.la = self.E * self.nu / ((1 + self.nu) * (1 -2 * self.nu))
        self.dt = 5e-4
        self.velocity = ti.Vector(self.dim, dt=ti.f32, shape=self.vn)
        self.neighbor_element_count = ti.var(dt=ti.i32, shape=self.vn)
        self.node_mass = ti.var(dt=ti.f32, shape=self.vn)
        self.element_mass = ti.var(dt=ti.f32, shape=self.en)
        self.element_volume = ti.var(dt=ti.f32, shape=self.en)
        self.B = ti.Matrix(self.dim, self.dim, dt=ti.f32, shape=self.en)  # a square matrix
        self.energy = ti.var(dt=ti.f32, shape=(), needs_grad=True)

        print("vertices: ", self.vn, "    elements: ", self.en)

        ## for rendering
        self.begin_point = ti.Vector(self.dim, ti.f32, shape=(self.en*3))
        self.end_point = ti.Vector(self.dim, ti.f32, shape=(self.en*3))

        for i in range(self.vn):
            self.node[i] = [self.v[2*i], self.v[2*i+1]]
            self.velocity[i] = [0, -5, 0]

        for i in range(self.en):
            self.element[i] = [self.e[3*i], self.e[3*i+1], self.e[3*i+2]]

    @ti.kernel
    def initialize(self):

        # calculate the center of the object (the target of the camera)
        for i in range(self.vn):
            for c in ti.static(range(self.dim)):
                self.center[None][c] += self.node[i][c]

        for c in ti.static(range(self.dim)):
            self.center[None][c] /= max(self.vn, 1)

        for i in range(self.en):
            D = self.D(i)
            self.B[i] = D.inverse()
            a, b, c = self.element[i][0], self.element[i][1], self.element[i][2]
            self.element_volume[i] = abs(D.determinant()) / 2  # space in 2d
            self.element_mass[i] = self.element_volume[i]
            self.node_mass[a] += self.element_mass[i]
            self.node_mass[b] += self.element_mass[i]
            self.node_mass[c] += self.element_mass[i]
            self.neighbor_element_count[a] += 1
            self.neighbor_element_count[b] += 1
            self.neighbor_element_count[c] += 1
            # print(i, "element_mass", self.element_mass[i], "element_volume", self.element_volume[i])

        for i in range(self.vn):
            self.node_mass[i] /= max(self.neighbor_element_count[i], 1)
            # print(i, "node_mass", self.node_mass[i])

    @ti.func
    def D(self, idx):
        a = self.element[idx][0]
        b = self.element[idx][1]
        c = self.element[idx][2]

        return ti.Matrix.cols([self.node[b] - self.node[a], self.node[c] - self.node[a]])

    @ti.func
    def F(self, i):  # deformation gradient
        return self.D(i) @ self.B[i]

    @ti.func
    def Psi(self, i):  # (strain) energy density
        F = self.F(i)
        J = max(F.determinant(), 0.01)
        return self.mu / 2 * ((F @ F.transpose()).trace() - self.dim) - self.mu * ti.log(J) + self.la / 2 * ti.log(
            J) ** 2

    @ti.func
    def U0(self, i):  # elastic potential energy
        return self.element_volume[i] * self.Psi(i)

    @ti.func
    def U1(self, i):  # gravitational potential energy E = mgh
        a = self.element[i][0]
        b = self.element[i][1]
        c = self.element[i][2]
        return self.element_mass[i] * 10 * 4 * (self.node[a].y + self.node[b].y + self.node[c].y) / 4

    @ti.kernel
    def energy_integrate(self):
        for i in range(self.en):
            self.energy[None] += self.U0(i) + self.U1(i)

    @ti.kernel
    def time_integrate(self):
        for i in range(self.vn):
            self.velocity[i] += (- self.node.grad[i] / self.node_mass[i]) * self.dt
            self.velocity[i] *= math.exp(self.dt * -6)
            self.node[i] += self.velocity[i] * self.dt

            # confine to boundary
            for c in ti.static(range(self.dim)):
                if self.node[i][c] < boundary[c][0]:
                    self.node[i][c] = boundary[c][0] + self.epsilon * ti.random()
                    self.velocity[i][c] = 0.0
                elif self.node[i][c] > boundary[c][1]:
                    self.node[i][c] = boundary[c][1] - self.epsilon * ti.random()
                    self.velocity[i][c] = 0.0

    @ti.kernel
    def create_lines(self):
        count = 0
        for i in range(self.en):
            p1 = self.node[self.element[i][0]] * screen_to_world_ratio / screen_res
            p2 = self.node[self.element[i][1]] * screen_to_world_ratio / screen_res
            p3 = self.node[self.element[i][2]] * screen_to_world_ratio / screen_res

            self.begin_point[count] = p1
            self.end_point[count] = p2
            self.begin_point[count+1] = p2
            self.end_point[count+1] = p3
            self.begin_point[count+2] = p3
            self.end_point[count+2] = p1

            count += 3


ti.init(arch=ti.cpu)

def render(gui, obj):
    canvas = gui.canvas
    canvas.clear(bg_color)
    obj.create_lines()

    gui.lines(obj.begin_point.to_numpy(), obj.end_point.to_numpy(), color=line_color, radius=1.5)
    gui.show()

def main():
    obj = Object('obj')
    obj.initialize()

    gui = ti.GUI('FEM', screen_res)
    while gui.running:

        for i in range(15):
            with ti.Tape(obj.energy):
                obj.energy_integrate()
            obj.time_integrate()
        render(gui, obj)


if __name__ == '__main__':
    main()