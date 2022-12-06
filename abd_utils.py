import taichi as ti
import numpy as np
from arp import Cube, skew
from scipy.linalg import lu
ti.init(ti.x64, default_fp=ti.f32)

n_cubes = 1
hinge = False
gravity = -np.array([0., -9.8e1, 0.0])
m = (n_cubes - 1) * 3 if not hinge else (n_cubes - 1) * 6
n_dof = 12 * n_cubes
delta = 0.08
centered = False
kappa = 1e4
max_iters = 10
dim_grad = 4
grad_field = ti.Vector.field(3, float, shape=(dim_grad))
hess_field = ti.Matrix.field(3, 3, float, shape=(dim_grad, dim_grad))
dt = 1e-2
ZERO = 1e-9
a_cols = n_cubes * 4
a_rows = m // 3
alpha = 1.0
a = ti.field(float, shape=(a_rows, a_cols)) if a_rows and a_cols else None
d = ti.field(float, shape=(a_rows, a_cols)) if a_rows and a_cols else None
# dual matrix to get the inverse


def fill_C(k, pk, r_kl, r_pkl):
    line = np.zeros((3, n_dof))
    line_pk = np.zeros((3, n_dof))
    fill_Jck(line, k, r_kl)
    fill_Jck(line_pk, pk, r_pkl)
    return line_pk - line


def fill_Jck(line, k, r_kl):
    k0 = k * 12
    line[:, k0: k0 + 3] = np.identity(3, np.float32)
    line[:, k0 + 3: k0 + 6] = r_kl[0] * np.identity(3, np.float32)
    line[:, k0 + 6: k0 + 9] = r_kl[1] * np.identity(3, np.float32)
    line[:, k0 + 9: k0 + 12] = r_kl[2] * np.identity(3, np.float32)


@ti.func
def argmax(i):
    m = 0.0
    id = i
    for j in range(i, a_cols):
        if abs(a[i, j]) > m:
            m = abs(a[i, j])
            id = j
    return id


@ti.func
def swap_col(i, j):
    for k in ti.static(range(a_rows)):
        t, s = a[k, i], d[k, i]
        a[k, i] = a[k, j]
        d[k, i] = d[k, j]

        a[k, j] = t
        d[k, j] = s


@ti.kernel
def gaussian_elimination_row_pivot(C: ti.types.ndarray(), V_inv: ti.types.ndarray()):
    for i, j in ti.static(ti.ndrange(a_rows, a_cols)):
        a[i, j] = C[i * 3, j * 3]

    for i in ti.static(range(a_rows)):
        d[i, i] = 1.0
    for i in range(a_rows):
        # forward
        u = argmax(i)
        swap_col(u, i)
        for k in ti.static(range(a_cols)):
            a[i, k] /= a[i, i]
            d[i, k] /= a[i, i]
        for j in range(i+1, a_rows):
            v = a[j, i]  # / a[i, i]
            for k in ti.static(range(a_cols)):
                a[j, k] -= v * a[i, k]
                d[j, k] -= v * d[i, k]
    for _i in range(a_rows):
        i = a_rows - 1 - _i
        for j in range(i+1, a_cols):
            if abs(a[i, j]) > ZERO:
                v = a[i, j]
                a[i, j] = 0

                # dik -= aij djk, forall k < cols
                if j < a_rows:
                    for k in ti.static(range(a_cols)):
                        d[i, k] -= v * d[j, k]
                else:
                    d[i, j] -= v

    for i, j in ti.static(ti.ndrange(a_rows, a_cols)):
        for k in ti.static(range(3)):
            V_inv[i * 3 + k, j * 3 + k] = d[i, j]


def U(C):
    m, n = C.shape[0], C.shape[1]
    # for i in range(m):
    #     i = np.argmax()

    '''
    pivoting not needed for 2 cubes
    identity already up front
    '''
    V = np.zeros((n, n), np.float32)
    if hinge:
        C[3:6, :] -= C[:3, :]
        C[3:6, :] *= -1.
        # no need to reorder for this selected hinge
        print(C[:, : 12])
    V[:m, :] = C
    V[m:, m:] = np.identity(n-m, np.float32)

    _V_inv = np.zeros_like(V)
    if d is not None:
        d.fill(0.0)
        a.fill(0.0)
        gaussian_elimination_row_pivot(C, _V_inv)
        print(a.to_numpy())
        print(d.to_numpy())
    _V_inv[m:, m:] = np.identity(n - m, np.float32)

    # V_inv = -V
    # V_inv[m:, m:] = np.identity(n - m, np.float32)
    # V_inv[:3, :3] = np.identity(3, np.float32)
    # if hinge:
    #     V_inv[3:6, 3:6] = np.identity(3, np.float32)
    #     V_inv[:3, 15:18] = np.zeros((3,3), np.float32)
    # print((V_inv - _V_inv)[:6])
    return V, _V_inv


# @ti.kernel
# def tiled_C(C: ti.types.ndarray(), m: int, n: int):
#     '''
#     gaussian elimination
#     row pivot
#     C_m*n: linear constraint
#     return: U_n*n
#     '''
#     for i in range(m):

#         max_front()
#         for j in range(i + 1, m):
#             range

#     pass

class Global:
    def __init__(self):
        self.g = np.zeros((n_dof), np.float32)
        self.H = np.zeros((n_dof, n_dof), np.float32)
        self.Eo = np.zeros((n_cubes), np.float32)


globals = Global()


@ti.func
def kronecker(i, j):
    return 1 if i == j else 0


@ti.kernel
def grad_Eo(q: ti.template()):
    for i in range(1, 4):
        g = ti.Vector.zero(float, 3)
        for j in range(1, 4):
            g += (q[i].dot(q[j]) - kronecker(i, j)) * q[j]
        grad_field[i] = 4 * kappa * g


@ti.kernel
def hess_Eo(q: ti.template()):
    for i, j in ti.ndrange((1, 4), (1, 4)):
        h = (q[j].dot(q[i]) - kronecker(i, j)) * ti.Matrix.identity(float, 3)
        for k in range(1, 4):
            h += (q[i] * kronecker(k, j) + q[k] * kronecker(i, j)) @ q[k].transpose()
            
        hess_field[i, j] = 4 * kappa * h

@ti.kernel
def Eo(q: ti.template()) -> float:
    A = ti.Matrix.cols([q[1], q[2], q[3]])
    return kappa * (A.transpose() @ A - ti.Matrix.identity(float, 3)).norm_sqr()


@ti.data_oriented
class AffineCube(Cube):
    def __init__(self, id, scale=[1.0, 1.0, 1.0], omega=[0., 0., 0.], pos=[0., 0., 0.], parent=None, Newton_Euler=False, mass = 1.0):
        super().__init__(id, scale=scale, omega=omega, pos=pos,
                         parent=parent, Newton_Euler=Newton_Euler, mass= mass)
        self.q = ti.Vector.field(3, float, shape=(4))
        self.q_dot = ti.Vector.field(3, float, shape=(4))
        self.q0 = ti.Vector.field(3, float, shape = (4))
        self._reset()
        for c in self.children:
            c._reset()
        # self.init_q_q_dot()

    @ti.kernel
    def init_q_q_dot(self):
        R = skew(self.omega[None])
        I = ti.Matrix.identity(float, 3)
        self.q[0] = self.p[None]
        for i, j in ti.static(ti.ndrange(3, 3)):
            self.q_dot[i + 1][j] = R[i, j]
            self.q[i + 1][j] = I[i, j]

    def _reset(self):
        self.p[None] = ti.Vector(self.initial_state[0])
        self.omega[None] = ti.Vector(self.initial_state[1])
        self.init_q_q_dot()
        self.q0.copy_from(self.q)
        for c in self.children:
            c._reset()

    def grad_Eo_top_down(self):
        global globals
        grad_Eo(self.q)
        i0 = self.id * 12
        gf_np = grad_field.to_numpy().reshape((1, -1))
        globals.g[i0: i0 + 12] = gf_np
        for c in self.children:
            c.grad_Eo_top_down()

    def hess_Eo_top_down(self):
        global globals
        hess_Eo(self.q)
        i0 = self.id * 12
        hf_np = hess_field.to_numpy().reshape((12, 12))
        # FIXME: probably wrong shape
        globals.H[i0: i0 + 12, i0: i0 + 12] = hf_np

        for c in self.children:
            c.hess_Eo_top_down()

    def Eo_top_down(self):
        global globals
        E = Eo(self.q)
        i0 = self.id
        globals.Eo[i0] = E
        for c in self.children:
            c.Eo_top_down()
        # if self.parent is None:
        #     print(f'Eo = {globals.Eo}')

    @ti.kernel
    def project_vertices(self):
        for i in ti.static(range(8)):
            A = ti.Matrix.rows([self.q[1], self.q[2], self.q[3]])
            self.v_transformed[i] = A @ self.vertices[i] + self.q[0]

    def traverse(self, q, update_q = True, update_q_dot = True, project = True):
        i0 = self.id * 12
        q_next = q[0, i0: i0 + 12].reshape((4, 3))
        q_current = self.q0.to_numpy()
        q_dot_next = (q_next - q_current) / dt
        q_dot_current = self.q_dot.to_numpy()

        # q_next = alpha * q_next + (1 - alpha) * q_current
        # q_dot_next = q_dot_next * alpha + (1 - alpha) * q_dot_current
        if update_q:
            self.q.from_numpy(q_next)
        if update_q_dot:
            self.q_dot.from_numpy(q_dot_next)
        if project:
            self.q0.copy_from(self.q)
            self.project_vertices()
        for c in self.children:
            c.traverse(q, update_q, update_q_dot, project)

    def fill_M(self, M):
        i0 = self.id * 12
        M[i0: i0 + 3] = self.m
        M[i0 + 3: i0 + 12] = self.Ic
        for c in self.children:
            c.fill_M(M)

    def assemble_q_q_dot(self):
        _q = self.q.to_numpy().reshape((1, -1))
        _q_dot = self.q_dot.to_numpy().reshape((1, -1))
        for c in self.children:
            q, q_dot = c.assemble_q_q_dot()
            _q = np.hstack([_q, q])
            _q_dot = np.hstack([_q_dot, q_dot])
        return _q, _q_dot

# @ti.func


def tiled_q(dt, q_dot_t, q_t, f_tp1):
    return q_t + dt * q_dot_t + dt ** 2 * f_tp1


def main():
    window = ti.ui.Window(
        "Articulated Multibody Simualtion", (800, 800), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0, 0, 0))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera_pos = np.array([0.0, 0.0, 3.0])
    camera_dir = np.array([0.0, 0.0, -1.0])

    root = AffineCube(0, omega=[10., 0., 0.], mass = 1e3)
    pos = [0., -1., 1.] if hinge else [-1., -1., -
                                       1.] if not centered else [-0.5, -0.5, -0.5]
    link = None if n_cubes < 2 else AffineCube(
        1, omega=[-10., 0., 0.], pos=pos, parent=root, mass = 1e3)

    C = np.zeros((m, n_dof), np.float32) if link is None else fill_C(
        1, 0, -link.r_lk_hat.to_numpy(), link.r_pkl_hat.to_numpy())
    if hinge and link is not None:
        v = link.vertices.to_numpy()
        # print("7, 6, 1, 0", v[7], v[6], v[1], v[0])
        C_p1 = fill_C(1, 0, v[6], v[5])
        C_p2 = fill_C(1, 0, v[2], v[1])
        C = np.vstack([C_p1, C_p2])
        # q, q_dot = root.assemble_q_q_dot()
        # print(C[:, : 12])
        # print(C @ q.T)

    V, V_inv = U(C)
    # V_inv = np.identity(n_dof, np.float32)
    print(np.max(V @ V_inv - np.identity(n_dof, np.float32)))
    
    # copied code ---------------------------------------
    mouse_staled = np.zeros(2, dtype=np.float32)
    diag_M = np.zeros((n_dof), np.float32)
    root.fill_M(diag_M)
    M = np.diag(diag_M)
    print(diag_M)
    while window.running:
        mouse = np.array([*window.get_cursor_pos(), 0.0])
        if window.is_pressed('a'):
            camera_pos[0] -= delta
        if window.is_pressed('d'):
            camera_pos[0] += delta
        if window.is_pressed('q'):
            camera_pos[1] -= delta
        if window.is_pressed('e'):
            camera_pos[1] += delta
        if window.is_pressed('w'):
            camera_pos[2] -= delta
        if window.is_pressed('s'):
            camera_pos[2] += delta
        if window.is_pressed('r'):
            root._reset()
        if window.is_pressed(ti.GUI.ESCAPE):
            quit()
        if window.is_pressed(ti.GUI.LMB):

            if (mouse_staled == 0.0).all():
                mouse_staled = mouse
            dmouse = mouse - mouse_staled
            camera_dir += dmouse * 1.0
            mouse_staled = mouse
        else :
            mouse_staled = np.zeros(2, dtype=np.float32)
        camera.position(*camera_pos)
        camera.lookat(*(camera_pos + camera_dir))
        scene.set_camera(camera)

        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        # copied code ---------------------------------------

        f = np.zeros((n_dof))
        # f[:3] = root.m * gravity
        # f[12: 15] = link.m * gravity
        q, q_dot = root.assemble_q_q_dot()
        q_tiled = tiled_q(dt, q_dot, q, f)
        for iter in range(max_iters):
            q, q_dot = root.assemble_q_q_dot()
            # shape = 1 * 24, transpose before use
            print(f'iter = {iter}')
            root.grad_Eo_top_down()
            root.hess_Eo_top_down()
            root.Eo_top_down()
            print('E = ', globals.Eo[0] * dt ** 2 + 0.5 * (q - q_tiled) @ M @ (q - q_tiled).T)
            print("Eo = ", globals.Eo[0])
            # gradient for transformed space
            # partial E(Uz)/partial z

            # print(f'shape V_inv = {V_inv.shape}, g = {globals.g.shape}, M = {M.shape}, q - q_tiled = {(q - q_tiled).T.shape}')
            grad = V_inv.T @ (globals.g.reshape((-1, 1)) *
                              dt ** 2 + M @ (q - q_tiled).T)

            hess = V_inv.T @ (globals.H * dt ** 2 + M) @ V_inv

            # set rows and columns to zero
            # grad[0: 3] = np.zeros((3), np.float32)
            # hess[0:3, :] = np.zeros((3, n_dof), np.float32)
            # hess[:, 0: 3] = np.zeros((n_dof, 3), np.float32)
            dz = -np.linalg.solve(hess[m:, m:], grad[m:])
            # set z_i = s_i if s_i != 0
            dz = np.hstack([np.zeros((1, m), np.float32), dz.reshape(1, -1)])
            dq = dz @ V_inv.T
            # print("norm(C dq) = ", np.max(C @ dq.T))
            q += dq
            print(dq, q - q_tiled)
            # root.traverse(q)
            root.traverse(q, update_q = True, update_q_dot = False, project = False)

        root.traverse(q)
        root.mesh(scene)
        root.particles(scene)
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
