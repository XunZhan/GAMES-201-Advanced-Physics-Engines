import math

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu, kernel_profiler=True)
# ti.init(arch=ti.cpu)

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

        self.on = 100
        self.vn = 1000
        self.en = 1000
        self.node = ti.Vector(self.dim, dt=ti.f32, shape=self.vn, needs_grad=True)
        self.prev_node = ti.Vector(self.dim, dt=ti.f32, shape=self.vn)
        self.prev_t_node = ti.Vector(self.dim, dt=ti.f32, shape=self.vn)
        self.bar_node = ti.Vector(self.dim, dt=ti.f32, shape=self.vn)
        self.p = ti.Vector(self.dim, dt=ti.f32, shape=self.vn)
        self.element = ti.Vector(self.dim + 1, dt=ti.i32, shape=self.en)

        #  the end index of i's object
        self.vn_object_index = ti.var(dt=ti.i32, shape=self.on)
        self.en_object_index = ti.var(dt=ti.i32, shape=self.on)
        self.count = ti.var(dt=ti.i32, shape=())

        #  the inverse obj id of each node and ele
        self.node_obj_idx = ti.var(dt=ti.i32, shape=self.vn)
        self.element_obj_idx = ti.var(dt=ti.i32, shape=self.en)

        ## for simulation
        self.E = 4000  # Young modulus
        self.nu = 0.3  # Poisson's ratio: nu \in [0, 0.5)
        self.mu = self.E / (2 * (1 + self.nu))
        self.la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.dt = 5e-3
        self.bar_d = 0.1
        self.k = 1  # contact stiffness

        # self.velocity = ti.Vector(self.dim, dt=ti.f32, shape=self.vn)
        self.node_mass = ti.var(dt=ti.f32, shape=self.vn)
        self.element_mass = ti.var(dt=ti.f32, shape=self.en)
        self.element_volume = ti.var(dt=ti.f32, shape=self.en)
        self.energy = ti.var(dt=ti.f32, shape=(), needs_grad=True)
        self.prev_energy = ti.var(dt=ti.f32, shape=())
        self.B = ti.Matrix(self.dim, self.dim, dt=ti.f32, shape=self.en)
        self.neighbor_element_count = ti.var(dt=ti.i32, shape=self.vn)

        ## for rendering
        self.begin_point = ti.Vector(self.dim, ti.f32, shape=(self.en * 3))
        self.end_point = ti.Vector(self.dim, ti.f32, shape=(self.en * 3))

    @ti.kernel
    def add_obj(self,
                vn: ti.i32,
                en: ti.i32,
                node: ti.ext_arr(),
                element: ti.ext_arr()
                ):

        print("Add obj before vn:", vn)
        print("Add obj before v:", self.vn_object_index[self.count])
        print("Add obj before e:", self.en_object_index[self.count])

        for i in range(vn):
            self.node[self.vn_object_index[self.count] + i] = [node[i, 0], node[i, 1]]
            self.prev_node[self.vn_object_index[self.count] + i] = [node[i, 0], node[i, 1]]
            self.prev_t_node[self.vn_object_index[self.count] + i] = [node[i, 0], node[i, 1]]
            self.bar_node[self.vn_object_index[self.count] + i] = [node[i, 0], node[i, 1]]
            self.node_obj_idx[self.vn_object_index[self.count] + i] = self.count + 1

        for i in range(en):
            # Mapping single object element id to system-wide
            self.element[self.en_object_index[self.count] + i] = \
                [self.vn_object_index[self.count] + element[i, 0],
                 self.vn_object_index[self.count] + element[i, 1],
                 self.vn_object_index[self.count] + element[i, 2]]
            self.element_obj_idx[self.en_object_index[self.count] + i] = self.count + 1

        # update vn_object_index and en_object_index
        self.vn_object_index[self.count + 1] = self.vn_object_index[self.count] + vn
        self.en_object_index[self.count + 1] = self.en_object_index[self.count] + en
        self.count += 1

        for i in range(self.en_object_index[self.count - 1], self.en_object_index[self.count]):
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

        for i in range(self.vn_object_index[self.count - 1], self.vn_object_index[self.count]):
            self.node_mass[i] /= max(self.neighbor_element_count[i], 1)

        print("Add obj after v:", self.vn_object_index[self.count])
        print("Add obj after e:", self.en_object_index[self.count])
        print("Add obj after mass:", self.node_mass[0])
        print("Printing nodes:")
        for i in range(self.vn_object_index[self.count]):
            print(self.node[i], " ", )
        print("\nPrinting element:")
        for i in range(self.en_object_index[self.count]):
            print(self.element[i], " ", )
            print(self.element_volume[i], " ", )
            print(self.element_mass[i], " ", )

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
        # print("coutn:",coun)

    @ti.func
    def U0(self, i):  # elastic potential energy
        return self.element_volume[i] * self.Psi(i)

    @ti.func
    def U1(self, i):  # gravitational potential energy E = mgh
        a = self.element[i][0]
        b = self.element[i][1]
        c = self.element[i][2]
        return self.element_mass[i] * 10 * (self.node[a].y + self.node[b].y + self.node[c].y)

    @ti.func
    def U2(self, i):
        return 0.5 * ((self.bar_node[i] - self.node[i]).norm_sqr())

    @ti.func
    def U3(self, i):
        # bounding contact potential
        t = 0.0
        dist_x_l = self.node[i].x - boundary[0][0]
        dist_x_u = boundary[0][1] - self.node[i].x
        dist_y_l = self.node[i].y - boundary[1][0]
        dist_y_u = boundary[1][1] - self.node[i].y

        # if dist_x_l<0 or dist_x_u<0 or dist_y_l<0 or dist_y_u<0:
        #     print("negtive dist: ",dist_x_l)
        #     print("negtive dist: ",dist_x_u)
        #     print("negtive dist: ",dist_y_l)
        #     print("negtive dist: ",dist_y_u)

        if dist_x_l <= self.bar_d:
            t += - self.k * ((dist_x_l - self.bar_d) ** 2) * ti.log(dist_x_l / self.bar_d)
        if dist_x_u <= self.bar_d:
            t += - self.k * ((dist_x_u - self.bar_d) ** 2) * ti.log(dist_x_u / self.bar_d)
        if dist_y_l <= self.bar_d:
            t += - self.k * ((dist_y_l - self.bar_d) ** 2) * ti.log(dist_y_l / self.bar_d)
        if dist_y_u <= self.bar_d:
            t += - self.k * ((dist_y_u - self.bar_d) ** 2) * ti.log(dist_y_u / self.bar_d)
        # if dist_x_l-self.bar_d<0 or dist_x_u-self.bar_d<0 or dist_y_l-self.bar_d<0 or dist_y_u-self.bar_d<0:
        #     # print(" dist: ",dist_x_l)
        #     # print(" dist: ",dist_x_u)
        #     print(" dist: ",dist_y_l)
        #     # print(" dist: ",dist_y_u)
        #     # print("T:", t)
        return t

    @ti.func
    def barrier_energy(self, p, q):
        # tmp = 0.0
        rt = 0.0
        d1, d2, d3 = 0.0, 0.0, 0.0
        # points: p
        pt = self.node[p]
        # triangle
        t1 = self.node[self.element[q][0]]
        t2 = self.node[self.element[q][1]]
        t3 = self.node[self.element[q][2]]

        # point - triangle pair, find point - triangle shortest dist
        # 2d problem: point - line segment shortest dist

        ab1 = t2 - t1
        ac1 = pt - t1
        bc1 = pt - t2
        a1 = ab1.dot(ac1)
        abd1 = ab1.norm_sqr()
        if a1 < 0:
            d1 = ac1.norm()
        elif a1 > abd1:
            d1 = bc1.norm()
        else:
            d1 = ti.abs((t2.y - t1.y) * pt.x - (t2.x - t1.x) * pt.y + t2.x * t1.y - t2.y * t1.x) / (t2 - t1).norm()

        ab2 = t3 - t2
        ac2 = pt - t2
        bc2 = pt - t3
        a2 = ab2.dot(ac2)
        abd2 = ab2.norm_sqr()
        if a2 < 0:
            d2 = ac2.norm()
        elif a2 > abd2:
            d2 = bc2.norm()
        else:
            d2 = ti.abs((t3.y - t2.y) * pt.x - (t3.x - t2.x) * pt.y + t3.x * t2.y - t3.y * t2.x) / (t3 - t2).norm()

        ab3 = t1 - t3
        ac3 = pt - t3
        bc3 = pt - t1
        a3 = ab3.dot(ac2)
        abd3 = ab3.norm_sqr()
        if a3 < 0:
            d3 = ac3.norm()
        elif a3 > abd3:
            d3 = bc3.norm()
        else:
            d3 = ti.abs((t1.y - t3.y) * pt.x - (t1.x - t3.x) * pt.y + t1.x * t3.y - t1.y * t3.x) / (t1 - t3).norm()

        # t2=t2
        # t3=t3
        # d1 = ti.abs(ac.cross(ab)) / ab.norm()
        # t3=t3
        # t1=t1
        # d1 = ti.abs(ac.cross(ab)) / ab.norm()

        # if d1 <= d2 and d1 <= d3:
        #     dist = d1
        # elif d2 <= d1 and d2 <= d3:
        #     dist = d2
        # else:
        #     dist = d3
        dist = min(d1, d2, d3)
        # print("contact dist: ",dist)
        # b_C2 function
        if dist < self.bar_d:
            rt = -self.k * ((dist - self.bar_d) ** 2) * ti.log(dist / self.bar_d)
        else:
            rt = 0

        # print(dist, tmp, -1 * tmp * tmp * ti.log(dist / self.bar_d))
        return rt

    @ti.kernel
    def reset_energy(self):
        self.energy[None] = 0

    @ti.kernel
    def update_barrier_energy(self):

        for i in range(self.en_object_index[self.count]):
            self.energy[None] += self.U0(i) * self.dt * self.dt
            self.energy[None] += self.U1(i) * self.dt * self.dt

        for i in range(self.vn_object_index[self.count]):
            self.energy[None] += self.U2(i)
            self.energy[None] += self.U3(i)

        # i, j: object
        # p: node, q:face
        for i in range(1, self.count + 1):
            for j in range(i + 1, self.count + 1):
                for p in range(self.vn_object_index[i - 1], self.vn_object_index[i]):
                    for q in range(self.en_object_index[j - 1], self.en_object_index[j]):
                        # print(p , self.element[q][0], self.element[q][1], self.element[q][2])
                        # if i!=j:

                        # print("i,j,p,q,t", i, j, p, q, t)
                        self.energy[None] += self.barrier_energy(p, q)
        for i in range(1, self.count + 1):
            for j in range(i + 1, self.count + 1):
                for p in range(self.vn_object_index[j - 1], self.vn_object_index[j]):
                    for q in range(self.en_object_index[i - 1], self.en_object_index[i]):
                        # print(p , self.element[q][0], self.element[q][1], self.element[q][2])
                        # if i!=j:

                        # print("i,j,p,q,t", i, j, p, q, t)
                        self.energy[None] += self.barrier_energy(p, q)

        # for i in range(1, self.count+1):
        #     for j in range(i+1, self.count+1):
        #         for p in range(self.vn_object_index[j-1], self.vn_object_index[j]):
        #             for q in range(self.en_object_index[i-1], self.en_object_index[i]):
        #                 # print(p , self.element[q][0], self.element[q][1], self.element[q][2])
        #                 self.energy[None] += self.barrier_energy(i, j, p, q)

    @ti.func
    def implicit_prob_node_in_element(self, i, j):
        # 这样的假设是有问题的,假设相对距离随时间单调
        # node i, element j
        rt = 0
        a, b, c = self.node[self.element[j][0]], self.node[self.element[j][1]], self.node[self.element[j][2]]
        p = self.node[i]
        Sabc = abs((b - a).cross(c - a))
        Spbc = abs((b - p).cross(c - p))
        Sapc = abs((p - a).cross(c - a))
        Sabp = abs((b - a).cross(p - a))
        # print("ti.abs(Sabc-Spbc-Sapc-Sabp)",ti.abs(Sabc-Spbc-Sapc-Sabp))

        if ti.abs(Sabc - Spbc - Sapc - Sabp) > self.epsilon:
            rt = 0
        else:
            rt = 1
        return rt

    @ti.kernel
    def implicit_prob_all(self) -> ti.i32:
        # i,j object
        # p,q node, face
        # flag 0: not contact, 1: contact
        flag = 0
        for i in range(1, self.count + 1):
            for j in range(1, self.count + 1):
                for p in range(self.vn_object_index[i - 1], self.vn_object_index[i]):
                    for q in range(self.en_object_index[j - 1], self.en_object_index[j]):
                        if i != j:
                            flag += self.implicit_prob_node_in_element(p, q)

        # boundary
        for t in range(self.vn_object_index[self.count]):
            if self.node[t][0] <= boundary[0][0] + self.epsilon:
                flag += 1
            if self.node[t][0] > boundary[0][1] - self.epsilon:
                flag += 1
            if self.node[t][1] <= boundary[1][0] + self.epsilon:
                flag += 1
            if self.node[t][1] > boundary[1][1] - self.epsilon:
                flag += 1

        return flag

    def ccd(self, l=0, r=1, max_iter=10):
        iter = 0
        while True:
            iter += 1
            mid = (l + r) / 2

            if iter > max_iter or r - l < self.epsilon:
                break

            self.update_node(mid)
            flag = self.implicit_prob_all()
            if flag == 0:
                l = mid
            else:
                r = mid
        # if l < self.epsilon:
        #     print("warning: small alpha")
        # self.update_node(l)
        # if self.implicit_prob_all() > 0.0:
        #     print("error: l prob", l)
        #     exit(-1)
        return l

    @ti.kernel
    def calc_x_bar(self):
        for i in range(self.vn_object_index[self.count]):
            self.bar_node[i] = 2 * self.node[i] - self.prev_t_node[i] + [0, -10 * self.dt * self.dt / self.node_mass[i]]
        # print("Printing nodes:")
        # for i in range(self.vn_object_index[self.count]):
        #     print(self.bar_node[i])

    @ti.kernel
    def x_bar_to_x(self):
        for i in range(self.vn_object_index[self.count]):
            self.node[i] = self.bar_node[i]

    @ti.kernel
    def x_to_prev_x(self):
        for i in range(self.vn_object_index[self.count]):
            self.prev_node[i] = self.node[i]

    @ti.kernel
    def x_to_prev_t_x(self):
        for i in range(self.vn_object_index[self.count]):
            self.prev_t_node[i] = self.node[i]

    @ti.kernel
    def calc_p(self):
        # print("P: ")
        for i in range(self.vn_object_index[self.count]):
            self.p[i] = self.node.grad[i]
            # print(self.p[i])

    @ti.kernel
    def save_energy(self):
        self.prev_energy[None] = self.energy[None]

    @ti.kernel
    def calc_p_inf_norm(self) -> float:
        m = 0.0
        for i in range(self.vn_object_index[self.count]):
            m = max(self.p[i][0], m)
            m = max(self.p[i][1], m)
        return m

    @ti.kernel
    def update_node(self, alpha: ti.f32):
        for i in range(self.vn_object_index[self.count]):
            self.node[i] = self.prev_node[i] - self.p[i] * alpha

    @ti.kernel
    def check_node(self):
        for i in range(self.vn_object_index[self.count]):
            if self.node[i].y < self.epsilon:
                print("small y")


def render(gui, system):
    canvas = gui.canvas
    canvas.clear(bg_color)
    system.create_lines()

    gui.lines(system.begin_point.to_numpy(), system.end_point.to_numpy(), color=line_color, radius=1.5)
    gui.show()


def implicit():
    s = System()
    gui = ti.GUI('IPC', screen_res)
    it = 0
    while gui.running:
        it += 1
        if it % 30 == 0:
            # print(s.prev_energy[None])
            print("iter = ", it)

            # it = 0
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == 'a':
                tmp_obj = NewObject("obj")
                s.add_obj(tmp_obj.vn, tmp_obj.en, tmp_obj.node.astype(np.float32), tmp_obj.element)
                # tprint(s)

        max_iter = 50
        cur_iter = 0
        tol = (1e-2) / s.dt

        s.calc_x_bar()

        s.x_bar_to_x()
        s.reset_energy()
        s.update_barrier_energy()
        s.save_energy()
        s.x_to_prev_x()

        while True:
            cur_iter += 1
            s.reset_energy()
            with ti.Tape(s.energy):
                s.update_barrier_energy()
            s.save_energy()

            s.calc_p()
            alpha = s.ccd(r=1)
            # alpha = 0.01
            if alpha < 0.01:
                print("ccd: ", alpha)
            while True:
                s.update_node(alpha)
                alpha *= 0.5
                s.reset_energy()
                s.update_barrier_energy()
                # print("energy: ",s.energy[None], s.prev_energy[None])
                # print("alpha: ",alpha)

                if np.isnan(s.energy[None]):
                    exit(-1)

                if s.energy[None] - s.prev_energy[None] < 0 or alpha < s.epsilon:
                    break

            s.x_to_prev_x()
            s.reset_energy()
            s.update_barrier_energy()
            s.save_energy()

            p_inf_norm = s.calc_p_inf_norm()
            # print("p_inf_norm", p_inf_norm)
            if p_inf_norm / s.dt <= tol or cur_iter > max_iter:
                break
        s.x_to_prev_t_x()
        if it % 12 == 0:
            render(gui, s)


if __name__ == '__main__':
    implicit()
    ti.kernel_profiler_print()
