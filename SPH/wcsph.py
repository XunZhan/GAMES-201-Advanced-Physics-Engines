# Becker, Markus, and Matthias T. Weakly compressible SPH for free surface flows. Proceedings of the 2007 ACM SIGGRAPH

import taichi as ti
import numpy as np
import math

ti.init(arch=ti.cpu, debug=False)

# Render Parameters
stop = ti.var(ti.i32, shape=())
bg_color = 0x112f41
particle_color = 0x068587
boundary_color = 0xebaca2
added_color = 0xffb169
max_add = 3  # max add particle time
cur_add = 0
add_interval = 90  # wait at least count between two adds
board_states = ti.Vector(2, dt=ti.f32)  # 0: x-pos, 1: timestep in sin()
screen_res = (400, 400)
screen_to_world_ratio = 35.0
boundary = (screen_res[0] / screen_to_world_ratio,
            screen_res[1] / screen_to_world_ratio)

# SPH Parameters
dim = 2
epsilon = 1e-5
particle_radius = 0.1

# Particals
mass = ti.var(ti.f32, shape=())
particles_x0 = 20
init_num_particles = particles_x0 * 15
cur_num_particles = ti.var(ti.i32, shape=())
total_num_particles = ti.var(ti.i32, shape=())
max_num_particles = 1000
max_num_particles_per_cell = 500
max_num_neighbors = 500
old_positions = ti.Vector(dim, dt=ti.f32)
cur_positions = ti.Vector(dim, dt=ti.f32)

density = ti.var(dt=ti.f32)
velocity = ti.Vector(dim, dt=ti.f32)
pressure = ti.var(dt=ti.f32)

delta_density = ti.var(dt=ti.f32)
delta_velocity = ti.Vector(dim, dt=ti.f32)
delta_viscosity = ti.Vector(dim, dt=ti.f32)  # vector, not scalar
delta_pressure = ti.Vector(dim, dt=ti.f32)   # vector, not scalar

# Function Parameters
rho0 = 1000.0  # reference density
gamma = 7.0
h = 1.3 * particle_radius
time_delta = ti.var(ti.f32, shape=())
gravity = -9.8 * 30
alpha = 0.3
C0 = 200
CFL_v = 0.20
CFL_a = 0.20
poly6_factor = 315.0 / 64.0 / np.pi
spiky_grad_factor = -45.0 / np.pi

# Neighbors
cell_size = 2.5
cell_recpr = 1.0 / cell_size
cube_size = 5  # add particle cube_size x cube_size to fluid


def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s


grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1))
grid_num_particles = ti.var(ti.i32)
grid2particles = ti.var(ti.i32)
neighbor_radius = h * 2.0
particle_num_neighbors = ti.var(ti.i32)
particle_neighbors = ti.var(ti.i32)

# Init Space
ti.root.dense(ti.i, max_num_particles).place(old_positions, cur_positions)
ti.root.dense(ti.i, max_num_particles).place(velocity)
ti.root.dense(ti.i, max_num_particles).place(delta_velocity, delta_viscosity, delta_pressure)

grid_snode = ti.root.dense(ti.ij, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.k, max_num_particles_per_cell).place(grid2particles)

nb_node = ti.root.dense(ti.i, max_num_particles)
nb_node.place(particle_num_neighbors)
nb_node.place(density, delta_density, pressure)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)

ti.root.place(board_states)

@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius
    bmax = ti.Vector([board_states[None][0], boundary[1]
                      ]) - particle_radius
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p


@ti.kernel
def confine_to_boundary():
    for i in range(total_num_particles[None]):
        pos = cur_positions[i]
        # change position
        cur_positions[i] = confine_position_to_boundary(pos)
        # change velocity
        velocity[i] = (cur_positions[i] - old_positions[i]) / time_delta[None]


@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[1] < grid_size[1]


@ti.func
def get_cell(pos):
    return (pos * cell_recpr).cast(int)


@ti.kernel
def find_particle_neighbors():
    for p_i in range(total_num_particles[None]):
        pos_i = old_positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (
                            pos_i - old_positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i


@ti.kernel
def update_grid():
    for p_i in range(total_num_particles[None]):
        cell = get_cell(old_positions[p_i])

        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = grid_num_particles[cell].atomic_add(1)
        grid2particles[cell, offs] = p_i


@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result


@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result


@ti.func
def cubic_kernel(r, hh):
    # B-cubic spline smoothing kernel
    k = 10. / (7. * np.pi * hh ** dim)
    q = r / hh
    res = ti.cast(0.0, ti.f32)
    if q < 1.0:
        res = (k / hh) * (-3 * q + 2.25 * q**2)
    elif q < 2.0:
        res = -0.75 * (k / hh) * (2 - q)**2
    return res


@ti.func
def p_update(rho):
    # Weakly compressible, tait function
    b = rho0 * C0**2 / gamma
    return b * ((rho / rho0)**gamma - 1.0)


@ti.kernel
def calc_density():
    # Eq (4)
    for p_i in range(total_num_particles[None]):
        delta_density[p_i] = 0.0
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            p_ij = old_positions[p_i] - old_positions[p_j]
            r_mod = ti.max(p_ij.norm(), epsilon)
            delta_density[p_i] += mass[None] * cubic_kernel(r_mod, h) * \
                            (velocity[p_i] - velocity[p_j]).dot(p_ij / r_mod)


@ti.kernel
def calc_pressure():
    for p_i in range(total_num_particles[None]):
        delta_pressure[p_i] = ti.Vector([0.0 for _ in range(dim)])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            p_ij = old_positions[p_i] - old_positions[p_j]
            r_mod = ti.max(p_ij.norm(), epsilon)
            delta_pressure[p_i] -= mass[None] * (pressure[p_i] / density[p_j] ** 2 + \
                                    pressure[p_j] / density[p_j] ** 2) * \
                                    cubic_kernel(r_mod, h) * p_ij / r_mod


@ti.kernel
def calc_viscosity():
    for p_i in range(total_num_particles[None]):
        delta_viscosity[p_i] = ti.Vector([0.0 for _ in range(dim)])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            p_ij = old_positions[p_i] - old_positions[p_j]
            r_mod = ti.max(p_ij.norm(), epsilon)

            v_ij = (velocity[p_i] - velocity[p_j]).dot(p_ij)
            # Artifical viscosity
            if v_ij < 0:
                v = -2.0 * alpha * particle_radius * C0 / (density[p_i] + density[p_j])
                delta_viscosity[p_i] -= mass[None] * v_ij * v / (r_mod ** 2 + 0.01 * particle_radius ** 2) * \
                                    cubic_kernel(r_mod, h) * p_ij / r_mod


@ti.kernel
def calc_velocity():
    for p_i in range(total_num_particles[None]):
        val = [0.0 for _ in range(dim - 1)]
        val.extend([gravity])
        delta_velocity[p_i] = delta_pressure[p_i] + delta_viscosity[p_i] + ti.Vector(val, dt=ti.f32)
        velocity[p_i] += time_delta[None] * delta_velocity[p_i]


@ti.kernel
def calc_position():
    for p_i in range(total_num_particles[None]):
        cur_positions[p_i] = old_positions[p_i] + time_delta[None] * velocity[p_i]


@ti.kernel
def update():
    for i in range(total_num_particles[None]):
        # velocity
        velocity[i] = (cur_positions[i] - old_positions[i]) / time_delta[None]
        # density
        density[i] += time_delta[None] * delta_density[i]
        # pressure
        pressure[i] = p_update(density[i])
        # positions
        old_positions[i] = cur_positions[i]


def adaptive_step():
    total_num = total_num_particles[None]
    max_v = np.max(np.linalg.norm(velocity.to_numpy()[:total_num], 2, axis=1))
    # CFL analysis, constrained by v_max
    dt_cfl = CFL_v * h / max_v

    max_a = np.max(np.linalg.norm((delta_velocity.to_numpy())[:total_num], 2, axis=1))
    # Constrained by a_max
    dt_f = CFL_a * np.sqrt(h / max_a)

    max_rho = np.max(density.to_numpy()[:total_num])
    dt_a = 0.20 * h / (C0 * np.sqrt((max_rho / rho0) ** gamma))

    time_delta[None] = np.min([dt_cfl, dt_f, dt_a])

def run_sph():
    #p = old_positions.to_numpy()[init_num_particles:total_num_particles]
    grid_num_particles.fill(0)
    particle_neighbors.fill(-1)
    update_grid()
    #a = grid_num_particles.to_numpy()
    find_particle_neighbors()
    #a = particle_neighbors.to_numpy()[init_num_particles:total_num_particles]

    calc_density()
    calc_pressure()
    calc_viscosity()
    calc_velocity()
    calc_position()

    confine_to_boundary()
    update()
    adaptive_step()


def render(gui):
    canvas = gui.canvas
    canvas.clear(bg_color)
    pos_np = old_positions.to_numpy()
    for i in range(total_num_particles[None]):
        for j in range(dim):
            pos_np[i,j] *= screen_to_world_ratio / screen_res[j]

    gui.circles(pos_np[:init_num_particles], radius=2.5, color=particle_color)
    gui.circles(pos_np[init_num_particles:total_num_particles[None]], radius=2.5, color=added_color)
    gui.rect((0, 0), (board_states[None][0] / boundary[0], 1.0),
             radius=1.5,
             color=boundary_color)

    gui.text(content=f'Press A: Add Particles to Fluid', pos=(0, 0.95), color=0xffffff)
    gui.text(content=f'Current Add: {cur_add} (Max: {max_add})', pos=(0, 0.9), color=0xffffff)
    gui.show()


def print_stats():
    print('SPH stats:')
    num = grid_num_particles.to_numpy()
    avg, max = np.mean(num), np.max(num)
    print(f'  #particles per cell: avg={avg:.2f} max={max}')
    num = particle_num_neighbors.to_numpy()
    avg, max = np.mean(num), np.max(num)
    print(f'  #neighbors per particle: avg={avg:.2f} max={max}')


@ti.kernel
def move_board():
    # probably more accurate to exert force on particles according to hooke's law.
    b = board_states[None]
    b[1] += 1.0
    period = 90
    vel_strength = 100.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta[None]
    board_states[None] = b

def init_env():
    # init time delta and mass and total num particles
    time_delta.from_numpy(np.array(0.1 * h / C0, dtype=np.float32))
    mass[None] = particle_radius ** dim * rho0
    total_num_particles[None] = init_num_particles
    cur_num_particles[None] = init_num_particles

    np_positions = np.zeros((cur_num_particles[None], dim), dtype=np.float32)
    delta = particle_radius * 2.1
    num_x = particles_x0
    num_y = cur_num_particles[None] // num_x
    assert num_x * num_y == cur_num_particles[None]
    offs = np.array([(boundary[0] - delta * num_x) * 0.5,
                     (boundary[1] * 0.05)],
                    dtype=np.float32)

    for i in range(cur_num_particles[None]):
        np_positions[i] = np.array([i % num_x, i // num_x]) * delta + offs
        np_velocities = (np.random.rand(cur_num_particles[None], dim).astype(np.float32) - 0.5) * 0.5

    @ti.kernel
    def init_particles(p: ti.ext_arr(), v: ti.ext_arr()):
        for i in range(cur_num_particles[None]):
            velocity[i] = ti.Vector([v[i, 0], -5.0])
            for c in ti.static(range(dim)):
                old_positions[i][c] = p[i, c]
                # velocity[i][c] = v[i, c]
                density[i] = rho0

    @ti.kernel
    def init_board():
        board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])

    init_particles(np_positions, np_velocities)
    init_board()

def add_particle():
    add_num = cube_size ** 2
    if cur_num_particles[None] + add_num >= max_num_particles:
        print('Error: Add too much particles.')
        exit()

    total_num_particles[None] = cur_num_particles[None] + add_num
    np_positions = np.zeros((add_num, dim), dtype=np.float32)
    delta = particle_radius * 2.1
    offs = np.array([(boundary[0] - delta * cube_size) * 0.5,
                     (boundary[1] * 0.5)],
                    dtype=np.float32)

    for i in range(add_num):
        np_positions[i] = np.array([i % cube_size, i // cube_size]) * delta + offs
        np_velocities = (np.random.rand(add_num, dim).astype(np.float32) - 0.5) * 0.25

    @ti.kernel
    def update_particles(p: ti.ext_arr(), v: ti.ext_arr()):
        for i in range(cur_num_particles[None], total_num_particles[None]):
            velocity[i] = ti.Vector([v[i - cur_num_particles[None], 0], -3.0])
            for c in ti.static(range(dim)):
                old_positions[i][c] = p[i - cur_num_particles[None], c]
                density[i] = rho0

    update_particles(np_positions, np_velocities)
    cur_num_particles[None] = total_num_particles[None]

def main():
    init_env()
    print(f'boundary={boundary} grid={grid_size} cell_size={cell_size}')
    gui = ti.GUI('WCSPH', screen_res)
    stop[None] = True
    print_counter = 0
    move_board()
    global cur_add
    last_call = add_interval

    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == gui.SPACE:
                stop[None] = not stop[None]
            elif e.key == 'a':
                if last_call >= add_interval:
                    last_call = 0

                    if cur_add < max_add:
                        add_particle()
                        cur_add += 1

        if not stop[None]:
            for _ in range(30):
                run_sph()

        last_call += 1
        if last_call > add_interval:
            last_call = add_interval

        # print_counter += 1
        # if print_counter == 30:
        #     print_stats()
        #     print_counter = 0

        render(gui)


if __name__ == '__main__':
    main()