# explict fem implement
import math

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

bg_color = 0x112f41
line_color = 0xffb169
boundary_color = 0xebaca2

screen_res = (400, 400)
screen_to_world_ratio = 40.0
boundary = [[0, screen_res[0] / screen_to_world_ratio],
            [0, screen_res[1] / screen_to_world_ratio]]


class NewObject:
    def __init__(self, filename):

        self.dim = 2
        self.v = []
        self.e = []
        # read nodes from *.node file
        with open(filename + ".node", "r") as file:
            vn = int(file.readline().split()[0])
            for i in range(vn):
                self.v += [float(x) for x in file.readline().split()[1:self.dim + 1]]  # [x,y] or [x,y,z]

        # read elements from *.ele file
        with open(filename + ".ele", "r") as file:
            en = int(file.readline().split()[0])
            for i in range(en):  # index is 0-based
                self.e += [int(ind) for ind in file.readline().split()[1:self.dim + 2]]  # triangle or tetrahedron

        self.vn = int(len(self.v) / self.dim)
        self.en = int(len(self.e) / (self.dim + 1))

        self.node = np.zeros([self.vn, self.dim])

        self.element = np.zeros([self.en, self.dim + 1])

        # ini
        for i in range(self.vn):
            self.node[i] = [self.v[2 * i], self.v[2 * i + 1]]

        for i in range(self.en):
            self.element[i] = [self.e[3 * i], self.e[3 * i + 1], self.e[3 * i + 2]]


@ti.data_oriented
class System:
    def __init__(self):

        self.dim = 2
        self.inf = 1e10
        self.epsilon = 1e-5

        self.obj_list = []
        self.on = 100
        self.vn = 1000
        self.en = 1000
        self.node = ti.Vector(self.dim, dt=ti.f32, shape=self.vn, needs_grad=True)
        self.prev_node = ti.Vector(self.dim, dt=ti.f32, shape=self.vn)
        self.prev_t_node = ti.Vector(self.dim, dt=ti.f32, shape=self.vn)
        self.bar_node = ti.Vector(self.dim, dt=ti.f32, shape=self.vn)
        self.p = ti.Vector(self.dim, dt=ti.f32, shape=self.vn)
        self.element = ti.Vector(self.dim + 1, dt=ti.i32, shape=self.en)
        self.center = ti.Vector(self.dim, ti.f32, shape=self.on)

        #  the end index of i's object
        self.vn_object_index = ti.var(dt=ti.i32, shape=self.on)
        self.en_object_index = ti.var(dt=ti.i32, shape=self.on)
        self.count = ti.var(dt=ti.i32, shape=())

        ## for simulation
        self.E = 5000  # Young modulus
        self.nu = 0.4  # Poisson's ratio: nu \in [0, 0.5)
        self.mu = self.E / (2 * (1 + self.nu))
        self.la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.dt = 5e-2

        self.velocity = ti.Vector(self.dim, dt=ti.f32, shape=self.vn)
        self.node_mass = ti.var(dt=ti.f32, shape=self.vn)
        self.element_mass = ti.var(dt=ti.f32, shape=self.en)
        self.element_volume = ti.var(dt=ti.f32, shape=self.en)
        self.energy = ti.var(dt=ti.f32, shape=(), needs_grad=True)
        self.prev_energy = ti.var(dt=ti.f32, shape=())
        self.B = ti.Matrix(self.dim, self.dim, dt=ti.f32, shape=self.en)
        self.neighbor_element_count = ti.var(dt=ti.i32, shape=self.vn)
        #
        # print("vertices: ", self.vn, "    elements: ", self.en)
        #
        ## for rendering
        self.begin_point = ti.Vector(self.dim, ti.f32, shape=(self.en * 3))
        self.end_point = ti.Vector(self.dim, ti.f32, shape=(self.en * 3))
        #
        # for i in range(self.vn):
        #     self.node[i] = [self.v[2*i], self.v[2*i+1]]
        #     self.velocity[i] = [0, -5, 0]
        #
        # for i in range(self.en):
        #     self.element[i] = [self.e[3*i], self.e[3*i+1], self.e[3*i+2]]

    @ti.kernel
    def add_obj(self,
                vn: ti.i32,
                en: ti.i32,
                node: ti.ext_arr(),
                element: ti.ext_arr()
                ):
        # print("Add obj before vn:", vn)
        # print("Add obj before v:", self.vn_object_index[self.count])
        # print("Add obj before e:", self.en_object_index[self.count])

        for i in range(vn):
            self.node[self.vn_object_index[self.count] + i] = [node[i, 0], node[i, 1]]
            self.prev_node[self.vn_object_index[self.count] + i] = [node[i, 0], node[i, 1]]
            self.prev_t_node[self.vn_object_index[self.count] + i] = [node[i, 0], node[i, 1]]
            # self.velocity[self.vn_object_index[self.count] + i] = [0, -5]
        for i in range(en):
            # Mapping single object element id to system-wide
            self.element[self.en_object_index[self.count] + i] = \
                [self.vn_object_index[self.count] + element[i, 0],
                 self.vn_object_index[self.count] + element[i, 1],
                 self.vn_object_index[self.count] + element[i, 2]]

        # update vn_object_index and en_object_index
        self.vn_object_index[self.count + 1] = self.vn_object_index[self.count] + vn
        self.en_object_index[self.count + 1] = self.en_object_index[self.count] + en
        self.count += 1

        for i in range(self.en_object_index[max(self.count - 1, 0)], self.en_object_index[self.count]):
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

        for i in range(self.vn_object_index[max(self.count - 1, 0)], self.vn_object_index[self.count]):
            self.node_mass[i] /= max(self.neighbor_element_count[i], 1)
        # print("Add obj after v:", self.vn_object_index[self.count])
        # print("Add obj after e:", self.en_object_index[self.count])
        # print("Add obj after mass:", self.node_mass[0])
        print("Add obj after node:", self.node[0])

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
    def create_lines(self):
        count = 0
        for i in range(self.en_object_index[self.count]):
            p1 = self.node[self.element[i][0]] * screen_to_world_ratio / screen_res
            p2 = self.node[self.element[i][1]] * screen_to_world_ratio / screen_res
            p3 = self.node[self.element[i][2]] * screen_to_world_ratio / screen_res

            self.begin_point[count] = p1
            self.end_point[count] = p2
            self.begin_point[count + 1] = p2
            self.end_point[count + 1] = p3
            self.begin_point[count + 2] = p3
            self.end_point[count + 2] = p1

            count += 3
        # print("e:",self.element[0][0])
        # print("v:",self.node[self.element[0][0]])
        # print("p:",p1,p2,p3)

    @ti.kernel
    def energy_integrate(self):
        for i in range(self.en_object_index[self.count]):
            self.energy[None] += self.U0(i) + self.U1(i)

    @ti.kernel
    def time_integrate(self):
        for i in range(self.vn_object_index[self.count]):
            self.velocity[i] += (- self.node.grad[i] / self.node_mass[i]) * self.dt
            self.velocity[i] *= math.exp(self.dt * -6)
            self.node[i] += self.velocity[i] * self.dt

            # confine to boundary
            for c in ti.static(range(self.dim)):
                if self.node[i][c] < boundary[c][0]:
                    self.node[i][c] = boundary[c][0] + self.epsilon * ti.random()
                    self.velocity[i][c] *= -1
                elif self.node[i][c] > boundary[c][1]:
                    self.node[i][c] = boundary[c][1] - self.epsilon * ti.random()
                    self.velocity[i][c] *= -1

    @ti.func
    def implicit_U4(self, i):
        return 0.5*((self.bar_node[i] - self.node[i]).norm_sqr())

    @ti.func
    def implicit_U5(self, i):
        # bounding contact potential
        t=0.0
        if self.node[i].x>=1:
            t+=0
        else:
            t+=-1 * ((self.node[i].x-1)**2)*ti.log(ti.abs(self.node[i]).x/1)
        if self.node[i].y>=1:
            t+=0
        else:
            t+=-1 * ((self.node[i].y-1)**2)*ti.log(ti.abs(self.node[i]).y/1)
        t=1*t
        # print("U5 i",t)
        return  t

    @ti.kernel
    def implicit_energy_integrate(self):
        # TODO: Fix U0 here
        # print("U0",self.U0(0))
        # print("U1",self.U1(0))
        # # print("U2",self.U2(0))
        # # print("U3",self.U3(0))
        # print("U4",self.implicit_U4(0))
        # print("U5",self.implicit_U5(0))
        # self.energy[None]=0
        for i in range(self.en_object_index[self.count]):
            self.energy[None] += self.U0(i) * self.dt * self.dt
            self.energy[None] += self.U1(i) * self.dt * self.dt
        for i in range(self.vn_object_index[self.count]):
            # self.energy[None] += self.implicit_U4(i) + self.implicit_U5(i)
            self.energy[None] += self.implicit_U4(i)
            self.energy[None] += self.implicit_U5(i)

    @ti.kernel
    def implicit_cal_x_bar(self):
        for i in range(self.vn_object_index[self.count]):
            self.bar_node[i] = 2 * self.node[i] - self.prev_t_node[i] + [0, -10 * self.dt * self.dt / self.node_mass[i]]

    @ti.kernel
    def implicit_x_bar_to_x(self):
        for i in range(self.vn_object_index[self.count]):
            self.node[i] = self.bar_node[i]

    @ti.kernel
    def implicit_x_to_x_prev(self):
        for i in range(self.vn_object_index[self.count]):
            self.prev_node[i] = self.node[i]

    @ti.kernel
    def implicit_cal_p(self):
        for i in range(self.vn_object_index[self.count]):
            self.p[i] = self.node.grad[i]
        print(self.p[0])
    @ti.kernel
    def implicit_cal_p_max_norm(self)->float:
        m=0.0
        for i in range(self.vn_object_index[self.count]):
            m =max(self.p[i][0],m) 
            m =max(self.p[i][1],m) 
        return m
    
    @ti.kernel
    def implicit_x_to_prev_t_x(self):
        for i in range(self.vn_object_index[self.count]):
            self.prev_t_node[i] = self.node[i]
    @ti.kernel
    def implicit_try_update_node(self,alpha:ti.f32):

        for i in range(self.vn_object_index[self.count]):
            self.node[i] = self.prev_node[i] -self.p[i] * alpha
        # print("E:", self.energy[None])
        # print("p:", self.p[0])
        # print("bar:", self.bar_node[0], )
        # print("prev_t_node:", self.prev_t_node[0], )
        # print("prev_node:", self.prev_node[0], )
        # print("node:", self.node[0], )


    @ti.kernel
    def implicit_clip(self)->bool:
        flag=False
        for i in range(self.vn_object_index[self.count]):
            for c in ti.static(range(self.dim)):
                if self.node[i][c] < boundary[c][0]:
                    self.node[i][c] = boundary[c][0] + self.epsilon * ti.random()
                    # self.velocity[i][c] *= -1
                    flag=True

                elif self.node[i][c] > boundary[c][1]:
                    self.node[i][c] = boundary[c][1] - self.epsilon * ti.random()
                    # self.velocity[i][c] *= -1
                    flag = True
        return flag
    @ti.kernel
    def save_previous_energy(self):
        self.prev_energy[None]=self.energy[None]
        self.energy[None] = 0

    @ti.kernel
    def get_delta(self)->ti.f32:
        return self.energy[None]-self.prev_energy[None]

    @ti.kernel
    def print_node(self):
        print(self.node[0])

def CCD()->float:

    pass
def render(gui, system):
    canvas = gui.canvas
    canvas.clear(bg_color)
    system.create_lines()

    gui.lines(system.begin_point.to_numpy(), system.end_point.to_numpy(), color=line_color, radius=1.5)
    gui.show()


def explicit():
    s = System()

    gui = ti.GUI('FEM', screen_res)
    while gui.running:

        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == 'a':
                tmp_obj = NewObject("obj")
                s.add_obj(tmp_obj.vn, tmp_obj.en, tmp_obj.node, tmp_obj.element)
        for i in range(15):
            with ti.Tape(s.energy):
                s.energy_integrate()
            s.time_integrate()

        render(gui, s)


@ti.kernel
def tprint(t: ti.template()):
    print(t.node[0])


def inplicit():
    s = System()
    tmp_obj = NewObject("obj")
    s.add_obj(tmp_obj.vn, tmp_obj.en, tmp_obj.node, tmp_obj.element)
    gui = ti.GUI('FEM', screen_res)
    while gui.running:

        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == 'a':
                tmp_obj = NewObject("obj")
                s.add_obj(tmp_obj.vn, tmp_obj.en, tmp_obj.node, tmp_obj.element)
                tprint(s)

        max_iter_num=10
        tol=1e-3
        p_max_norm=1e3
        i=0
        # cal x_bar
        # print("1")
        # s.print_node()
        s.implicit_cal_x_bar()
        # print("2")
        # s.print_node()
        s.implicit_x_bar_to_x()
        # print("3")
        # s.print_node()
        s.implicit_energy_integrate()
        # print("4")
        # s.print_node()
        s.save_previous_energy()
        # print("5")
        # s.print_node()
        s.implicit_x_to_x_prev()
        # print("6")
        # s.print_node()

        while p_max_norm/s.dt>tol and i < max_iter_num:
            i+=1

            with ti.Tape(s.energy):
                s.implicit_energy_integrate()
            s.implicit_cal_p()

            s.save_previous_energy()

            # alpha=CCD()
            alpha=0.01
            delta=1

            while delta >= 0:

                s.implicit_try_update_node(alpha)
                s.implicit_energy_integrate()
                delta=s.get_delta()
                alpha *= 0.5

            # print(alpha)
            # s.implicit_update_node()
            s.implicit_x_to_x_prev()

            s.implicit_energy_integrate()
            s.save_previous_energy()
            p_max_norm=s.implicit_cal_p_max_norm()
        print(p_max_norm)
        s.implicit_x_to_prev_t_x()
        render(gui, s)


if __name__ == '__main__':
    # explicit()
    inplicit()