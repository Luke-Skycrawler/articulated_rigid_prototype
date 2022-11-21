import taichi as ti
import numpy as np
from arp import Cube, skew
ti.init(ti.x64, default_fp=ti.f32)

n_cubes = 2
m = (n_cubes - 1) * 3
n_dof = 12 * n_cubes
delta = 0.08
centered = False
kappa = 1e5
max_iters = 10
dim_grad = 4
grad_field = ti.Vector.field(3, float, shape = (dim_grad))
hess_field = ti.Matrix.field(3,3, float, shape = (dim_grad, dim_grad))
dt = 3e-4

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


def U(C):
    m, n = C.shape[0], C.shape[1]
    # for i in range(m):
    #     i = np.argmax()

    '''
    pivoting not needed for 2 cubes
    identity already up front
    '''
    V = np.zeros((n, n), np.float32)
    V[:m, :] = C
    V[m:, m:] = np.identity(n-m, np.float32)
    V_inv = -V
    V_inv[m:, m:] = np.identity(n - m, np.float32)
    V_inv[:m, :m] = np.identity(m, np.float32)
    return V, V_inv


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
        # partial i, partial j
        # h = ti.Matrix.zero(float, 3, 3)
        k = j
        h = (q[k] @ q[i].transpose() + ti.Matrix.identity(float, 3) * (q[i].dot(q[k]) - kronecker(i, k)))
        hess_field[i, j] = 4 * kappa * h

@ti.data_oriented
class AffineCube(Cube):
    def __init__(self, id, scale=[1.0, 1.0, 1.0], omega=[0., 0., 0.], pos=[0., 0., 0.], parent=None, Newton_Euler=False):
        super().__init__(id, scale=scale, omega=omega, pos=pos, parent=parent, Newton_Euler=Newton_Euler)
        self.q = ti.Vector.field(3, float, shape = (4))
        self.q_dot = ti.Vector.field(3, float, shape = (4))
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
    
    @ti.kernel
    def project_vertices(self):
        for i in ti.static(range(8)):
            A = ti.Matrix.cols([self.q[1], self.q[2], self.q[3]])
            self.v_transformed[i] = A @ self.vertices[i] + self.q[0]

    def traverse(self, q):
        i0 = self.id * 12
        q_next = q[0, i0: i0 + 12].reshape((4, 3))
        q_current = self.q.to_numpy()
        q_dot_next = (q_next - q_current) / dt
        self.q.from_numpy(q_next)
        self.q_dot.from_numpy(q_dot_next)
        self.project_vertices()
        for c in self.children:
            c.traverse(q)
    
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

    root = AffineCube(0, omega = [10., 0., 0.])
    link = None if n_cubes < 2 else AffineCube(1, omega=[-10., 0., 0.], pos = [-1., -1., -1.] if not centered else [-0.5, -0.5, -0.5], parent = root) 
    
    C = np.zeros((m, n_dof), np.float32) if link is None else fill_C(1, 0, -link.r_lk_hat.to_numpy(), link.r_pkl_hat.to_numpy())
    V, V_inv = U(C)
    # V_inv = np.identity(n_dof, np.float32)
    # print(np.max(V @ V_inv - np.identity(n_dof, np.float32)))

    # copied code ---------------------------------------
    mouse_staled = np.zeros(2, dtype=np.float32)
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

        if (mouse_staled == 0.0).all():
            mouse_staled = mouse
        dmouse = mouse - mouse_staled
        camera_dir += dmouse * 1.0
        mouse_staled = mouse

        camera.position(*camera_pos)
        camera.lookat(*(camera_pos + camera_dir))
        scene.set_camera(camera)

        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        # copied code ---------------------------------------

        f = np.zeros((n_dof))
        q, q_dot = root.assemble_q_q_dot()
        # shape = 1 * 24, transpose before use
        q_tiled = tiled_q(dt, q_dot, q, f)
        for iter in range(max_iters):
            root.grad_Eo_top_down()
            root.hess_Eo_top_down()
            diag_M = np.zeros((n_dof), np.float32)
            root.fill_M(diag_M)
            M = np.diag(diag_M)
            # gradient for transformed space
            # partial E(Uz)/partial z

            # print(f'shape V_inv = {V_inv.shape}, g = {globals.g.shape}, M = {M.shape}, q - q_tiled = {(q - q_tiled).T.shape}')
            grad = V_inv.T @ (globals.g.reshape((-1, 1)) * dt ** 2 + M @ (q - q_tiled).T)
            
            hess = V_inv.T @ (globals.H * dt ** 2 + M) @ V_inv 

            # set rows and columns to zero
            # grad[0: 3] = np.zeros((3), np.float32)
            # hess[0:3, :] = np.zeros((3, n_dof), np.float32)
            # hess[:, 0: 3] = np.zeros((n_dof, 3), np.float32)
            dz = -np.linalg.solve(hess[m: , m: ], grad[m: ])
            # set z_i = s_i if s_i != 0
            dz = np.hstack([np.zeros((1, m), np.float32), dz.reshape(1, -1)])
            dq = dz @ V_inv.T
            # print("norm(C dq) = ", np.max(C @ dq.T))
            q += dq
            
        root.traverse(q)
        root.mesh(scene)
        root.particles(scene)
        canvas.scene(scene)
        window.show()    

if __name__ == "__main__":
    main()
