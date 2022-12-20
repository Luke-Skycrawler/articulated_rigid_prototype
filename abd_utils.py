import taichi as ti
import numpy as np
from arp import Cube, skew
from scipy.linalg import lu, ldl, solve
import ipctk

ti.init(ti.x64, default_fp=ti.f64)

n_cubes = 2
hinge = True
gravity = -np.array([0., -9.8e1, 0.0])
m = (n_cubes - 1) * 3 if not hinge else (n_cubes - 1) * 6
n_dof = 12 * n_cubes
delta = 0.08
centered = False
kappa = 1e7
max_iters = 10
dim_grad = 4
grad_field = ti.Vector.field(3, float, shape=(dim_grad))
hess_field = ti.Matrix.field(3, 3, float, shape=(dim_grad, dim_grad))
dt = 1e-3
ZERO = 1e-9
a_cols = n_cubes * 4
a_rows = m // 3
alpha = 1.0
trace = False
articulated = False
dhat = 1e-2
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
    line[:, k0: k0 + 3] = np.identity(3, np.float64)
    line[:, k0 + 3: k0 + 6] = r_kl[0] * np.identity(3, np.float64)
    line[:, k0 + 6: k0 + 9] = r_kl[1] * np.identity(3, np.float64)
    line[:, k0 + 9: k0 + 12] = r_kl[2] * np.identity(3, np.float64)


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
    V = np.zeros((n, n), np.float64)
    if hinge:
        C[3:6, :] -= C[:3, :]
        C[3:6, :] *= -1.
        # no need to reorder for this selected hinge
        print(C[:, : 12])
    V[:m, :] = C
    V[m:, m:] = np.identity(n-m, np.float64)

    _V_inv = np.zeros_like(V)
    if d is not None:
        d.fill(0.0)
        a.fill(0.0)
        gaussian_elimination_row_pivot(C, _V_inv)
        print(a.to_numpy())
        print(d.to_numpy())
    _V_inv[m:, m:] = np.identity(n - m, np.float64)

    # V_inv = -V
    # V_inv[m:, m:] = np.identity(n - m, np.float64)
    # V_inv[:3, :3] = np.identity(3, np.float64)
    # if hinge:
    #     V_inv[3:6, 3:6] = np.identity(3, np.float64)
    #     V_inv[:3, 15:18] = np.zeros((3,3), np.float64)
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
        self.g = np.zeros((n_dof), np.float64)
        self.H = np.zeros((n_dof, n_dof), np.float64)
        self.Eo = np.zeros((n_cubes), np.float64)


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
    A = ti.Matrix.rows([q[1], q[2], q[3]])
    return kappa * (A @ A.transpose() - ti.Matrix.identity(float, 3)).norm_sqr()


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
        self.p_t0 = None
        self.p_t1 = None
        self.t_t0 = None
        self.t_t1 = None

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
        _hf_np = hess_field.to_numpy()
        hf_np = np.zeros((12, 12), np.float64)
        # print(_hf_np)
        # hf_np = hf_np.reshape((12, 12))
        for i in range(4):
            for j in range(4):
                hf_np[i * 3: i * 3 + 3, j * 3 : j * 3  +3] = _hf_np[i, j] 

        # print(hf_np)
        # FIXME: probably wrong shape, fixed
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

    @ti.kernel
    def project_vertices_t2(self, dq: ti.types.ndarray()):
        for i in ti.static(range(8)):
            dq0 = ti.Vector([dq[0], dq[1], dq[2]])
            dq1 = ti.Vector([dq[3], dq[4], dq[5]])
            dq2 = ti.Vector([dq[6], dq[7], dq[8]])
            dq3 = ti.Vector([dq[9], dq[10], dq[11]])

            A = ti.Matrix.rows([self.q[1] + dq1, self.q[2] + dq2, self.q[3] + dq3])
            self.v_transformed[i] = A @ self.vertices[i] + (self.q[0] + dq0)

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

    @ti.kernel
    def gen_triangles(self, t: ti.types.ndarray()):
        for i, j in ti.ndrange(12, 3):
            
            I = self.indices[i * 3 + j]
            v = self.v_transformed[I]
            for k in ti.static(range(3)):
                t[i, j, k] = v[k]
            
# @ti.func
c1 = 1e-4
def line_search(dq, root, q0, grad0, q_tiled, M):
    wolfe = False
    alpha = 1
    E0 = np.sum(globals.Eo) * dt ** 2 + 0.5 * (q0 - q_tiled) @ M @ (q0 - q_tiled).T
    while not wolfe and np.linalg.norm(grad0) > 1e-3:
        q1 = q0 + dq * alpha
        if not isinstance(root, list):       
            root.traverse(q1, True, False, False)
            root.Eo_top_down()
        else :
            traverse(root, q1, True, False, False) 
            for c in root:
                E = Eo(c.q)
                i0 = c.id
                globals.Eo[i0] = E

        E1 = np.sum(globals.Eo) * dt ** 2 + 0.5 * (q1 - q_tiled) @ M @ (q1 - q_tiled).T
        # print(dq.shape, grad0.shape)
        wolfe = E1 <= E0 + c1 * alpha * (dq @ grad0) 
        alpha /= 2
        if alpha < 1e-8:
            print(f'dq_norm = {np.linalg.norm(dq)}, grad norm = {np.linalg.norm(grad0)}, dq @ grad = {dq @ grad0}')
            print(f'E1 = {E1[0]}, alpha = {alpha}, wolfe rhs = {E0[0] + c1 * alpha * (dq @ grad0)}')
            print(f"error, alpha = {alpha}, grad norm = {np.linalg.norm( grad0)}, dq = {np.linalg} ")
            quit() 
    return alpha * 2

def tiled_q(dt, q_dot_t, q_t, f_tp1):
    return q_t + dt * q_dot_t + dt ** 2 * f_tp1


def assemble_q_q_dot(roots):
    q = np.zeros((1, n_dof))
    q_dot = np.zeros_like(q)
    for c in roots:
        _q = c.q.to_numpy().reshape((1, -1))
        _q_dot = c.q_dot.to_numpy().reshape((1, -1))

        q[0, c.id * 12: (c.id + 1) * 12] = _q
        q_dot[0, c.id * 12: (c.id + 1) * 12] = _q_dot
    return q, q_dot


def Eo_differentials(cubes):
    global globals
    for c in cubes:
        grad_Eo(c.q)
        i0 = c.id * 12
        gf_np = grad_field.to_numpy().reshape((1, -1))
        globals.g[i0: i0 + 12] = gf_np

        hess_Eo(c.q)
        _hf_np = hess_field.to_numpy()
        hf_np = np.zeros((12, 12), np.float64)
        for i in range(4):
            for j in range(4):
                hf_np[i * 3: i * 3 + 3, j * 3: j * 3 + 3] = _hf_np[i, j]
        globals.H[i0: i0 + 12, i0: i0 + 12] = hf_np

        E = Eo(c.q)
        globals.Eo[c.id] = E


def step_size_upper_bound(dq, cubes, pts, idx):
    t = 1.0
    
    for pt, ij in zip(pts, idx):
        p, t0, t1, t2 = pt
        i, v, j, f  = ij
        def t2_np(i):
            
            i0 = cubes[i].id * 12
            dq_slice = dq[i0: i0 + 12].reshape((12))
            cubes[i].project_vertices_t2(dq_slice)
            return cubes[i].v_transformed.to_numpy()

        p_t1 = t2_np(i)[v]

        t_x = t2_np(j)
        t_idx = cubes[j].indices.to_numpy()[3 * f: 3 * f + 3]
        t0_t1 = t_x[t_idx[0]]
        t1_t1 = t_x[t_idx[1]]
        t2_t1 = t_x[t_idx[2]]

        _, _t = ipctk.point_triangle_ccd(
            p, t0, t1, t2, p_t1, t0_t1, t1_t1, t2_t1)
        t = min(t, _t)
    return t


def compute_constraint_set(cubes):
    for c in cubes:
        t_t0 = np.zeros((12, 3, 3))
        c.project_vertices()
        c.p_t0 = c.v_transformed.to_numpy()
        # shape 8x3
        c.gen_triangles(t_t0)

        t_t1 = np.zeros_like(t_t0)
        i0 = c.id * 12
        # dq_slice = dq[i0: i0 + 12]
        # c.project_vertices_t2(dq_slice)
        c.p_t1 = c.v_transformed.to_numpy()
        c.gen_triangles(t_t1)

        c.t_t0 = t_t0
        c.t_t1 = t_t1

        p = np.array([c.p_t0, c.p_t1])
        c.lp = np.min(p, axis = 0)
        c.up = np.max(p, axis = 0)

        l_t0 = np.min(t_t0, axis = 1)
        u_t0 = np.max(t_t0, axis = 1)
        l_t1 = np.min(t_t1, axis = 1)
        u_t1 = np.max(t_t1, axis = 1)

        l = np.array([l_t0, l_t1])
        u = np.array([u_t0, u_t1])

        c.lt = np.min(l, axis = 0)
        c.ut = np.max(u, axis = 0)

        # print(c.lt.shape, c.ut.shape)
        # assert 12x3

    pt_set = []
    idx_set = []
    for i, ci in  enumerate(cubes):
        for j, cj in enumerate(cubes):
            if i == j :
                continue 
            pts, idx = pt_intersect(ci, cj, i, j, dhat)
            pt_set += pts
            idx_set += idx
    return np.array(pts), np.array(idx_set)
                
            
@ti.kernel
def candidacy(up:ti.types.ndarray(), lp:ti.types.ndarray(), ut:ti.types.ndarray(), lt:ti.types.ndarray(), ret:ti.types.ndarray()):
    for p, t in ti.ndrange(8, 12):
        upx = ti.Vector([up[p, 0], up[p, 1], up[p, 2]])
        lpx = ti.Vector([lp[p, 0], lp[p, 1], lp[p, 2]])

        utx = ti.Vector([ut[t, 0], ut[t, 1], ut[t, 2]])
        ltx = ti.Vector([lt[t, 0], lt[t, 1], lt[t, 2]])
        
        board_phase_candidacy = not ((upx < ltx).all() or (lpx > utx).all())

        ret[p, t] = board_phase_candidacy



    
def pt_intersect(ci, cj, _i, _j, dhat = 0.0):
    '''
    test body ci's point against body cj's triangles  
    '''
    pts = []
    idx = []
    # cand = np.zeros((8, 12))
    # candidacy(ci.up, ci.lp, cj.ut, cj.lt, cand)
    cand = np.ones((8, 12))
    for i in range(8):
        for j in range(12):
            if cand[i, j]:
                p = ci.p_t0[i]
                t0 = cj.t_t0[j, 0]
                t1 = cj.t_t0[j, 1]
                t2 = cj.t_t0[j, 2]

                d2 = ipctk.point_triangle_distance(p, t0, t1, t2)
                d = np.sqrt(d2)

                if d < dhat:
                    # p_t1 = ci.p_t1[i]
                    # t0_t1 = cj.t_t1[j, 0]
                    # t1_t1 = cj.t_t1[j, 1]
                    # t2_t1 = cj.t_t1[j, 2]
                    # ret.append([p, t0, t1, t2, p_t1, t0_t1, t1_t1, t2_t1])
                    pts.append([p, t0, t1, t2 ])
                    idx.append([_i, i, _j, j])
    return pts, idx
                
def fill_M(cubes, M):
    for c in cubes:
        i0 = c.id * 12
        M[i0: i0 + 3] = c.m
        M[i0 + 3: i0 + 12] = c.Ic


def traverse(cubes, q, update_q=True, update_q_dot=True, project=True):
    for c in cubes:
        i0 = c.id * 12
        q_next = q[0, i0: i0 + 12].reshape((4, 3))
        q_current = c.q0.to_numpy()
        q_dot_next = (q_next - q_current) / dt
        q_dot_current = c.q_dot.to_numpy()

        # q_next = alpha * q_next + (1 - alpha) * q_current
        # q_dot_next = q_dot_next * alpha + (1 - alpha) * q_dot_current
        if update_q:
            c.q.from_numpy(q_next)
        if update_q_dot:
            c.q_dot.from_numpy(q_dot_next)
        if project:
            c.q0.copy_from(c.q)
            c.project_vertices()


def step_disjoint(cubes, M, V_inv):
    f = np.zeros((n_dof))
    q, q_dot = assemble_q_q_dot(cubes)
    q_tiled = tiled_q(dt, q_dot, q, f)
    do_iter = True
    iter = 0
    pts, idx = compute_constraint_set(cubes)
    # print(constraints)
    # quit()
    while do_iter:
        q, q_dot = assemble_q_q_dot(cubes)
        Eo_differentials(cubes)
        hess = globals.H * dt ** 2 + M
        grad = globals.g.reshape((-1, 1)) * dt ** 2 + M @ (q - q_tiled).T

        dq = -solve(hess, grad, assume_a="pos")
        if trace:
            print(f'dq shape = {dq.shape}')
            print(f'hess shape = {hess.shape}')
            print(f'grad shape = {grad.shape}')
        t = step_size_upper_bound(dq, cubes, pts, idx)
        dq *= t
        dq = dq.reshape((1 , -1)) 
        alpha = line_search(dq, cubes, q, grad, q_tiled, M)
        # FIXME: line search arguments, fixed
        q += dq * alpha
        traverse(cubes, q, update_q=True, update_q_dot=False, project=False)

        do_iter = np.linalg.norm(dq * alpha) > 1e-4
        iter += 1
        if not do_iter:
            print(f'converge after {iter} iters')
    traverse(cubes, q)


def step_link(root, M, V_inv):
    f = np.zeros((n_dof))
    # f[:3] = root.m * gravity
    # f[12: 15] = link.m * gravity
    q, q_dot = root.assemble_q_q_dot()
    q_tiled = tiled_q(dt, q_dot, q, f)
    do_iter = True
    iter = 0
    while do_iter:
    # for iter in range(max_iters):
        q, q_dot = root.assemble_q_q_dot()
        # shape = 1 * 24, transpose before use
        root.grad_Eo_top_down()
        root.hess_Eo_top_down()
        root.Eo_top_down()
        if trace :
            print(f'iter = {iter}')
            print('E = ', globals.Eo[0] * dt ** 2 + 0.5 * (q - q_tiled) @ M @ (q - q_tiled).T)
            print("Eo = ", globals.Eo[0])
            
        # gradient for transformed space
        # partial E(Uz)/partial z

        # print(f'shape V_inv = {V_inv.shape}, g = {globals.g.shape}, M = {M.shape}, q - q_tiled = {(q - q_tiled).T.shape}')
        grad = V_inv.T @ (globals.g.reshape((-1, 1)) *
                            dt ** 2 + M @ (q - q_tiled).T)

        hess = V_inv.T @ (globals.H * dt ** 2 + M) @ V_inv

        # set rows and columns to zero
        # grad[0: 3] = np.zeros((3), np.float64)
        # hess[0:3, :] = np.zeros((3, n_dof), np.float64)
        # hess[:, 0: 3] = np.zeros((n_dof, 3), np.float64)


        # dz = -np.linalg.solve(hess[m:, m:], grad[m:])
        # l, d, perm = ldl(hess[m:, m:], lower= 0)
        # print(hess[m:, m:] - hess[m:, m:].T)
        # print(hess[m:, m:])
        dz = -solve(hess[m:, m:], grad[m:], assume_a = "pos")
        # set z_i = s_i if s_i != 0
        dz = np.hstack([np.zeros((1, m), np.float64), dz.reshape(1, -1)])
        dq = dz @ V_inv.T
        # print("norm(C dq) = ", np.max(C @ dq.T))
        alpha = line_search(dq, root, q, grad, q_tiled, M)
        if trace:
            print(f'alpha = {alpha}')
        # alpha = 1.0
        q += dq * alpha
        # print(dq, q - q_tiled)
        root.traverse(q, update_q = True, update_q_dot = False, project = False)
        do_iter = np.linalg.norm(dq * alpha) > 1e-4
        iter += 1
        if not do_iter: 
            print(f'converge after {iter} iters')
            
    root.traverse(q)


step = step_link if articulated else step_disjoint
def main():
    formatter = '{:.2e}'.format
    np.set_printoptions(formatter={'float_kind': formatter})
    window = ti.ui.Window(
        "Articulated Multibody Simualtion", (800, 800), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0, 0, 0))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera_pos = np.array([0.0, 0.0, 3.0])
    camera_dir = np.array([0.0, 0.0, -1.0])

    _root = AffineCube(0, omega=[10., 0., 0.], mass = 1e3)
    pos = [0., -1., 1.] if hinge else [-1., -1., -
                                       1.] if not centered else [-0.5, -0.5, -0.5]

    if not articulated:
        pos[2] += dhat
    link = None if n_cubes < 2 else AffineCube(
        1, omega=[-10., 0., 0.], pos=pos, parent=_root, mass = 1e3)

    link2 = None if n_cubes < 3 else AffineCube(
        1, omega=[10., 0., 0.], pos=[0. , -2., 2.], parent=link, mass = 1e3)

    C = np.zeros((m, n_dof), np.float64) if link is None else fill_C(
        1, 0, -link.r_lk_hat.to_numpy(), link.r_pkl_hat.to_numpy())
    
    root = _root if articulated else [_root, link]
    if hinge and link is not None:
        v = link.vertices.to_numpy()
        # print("7, 6, 1, 0", v[7], v[6], v[1], v[0])
        C_p1 = fill_C(1, 0, v[6], v[5])
        C_p2 = fill_C(1, 0, v[2], v[1])
        C = np.vstack([C_p1, C_p2])
        if n_cubes == 3: 
            C_p1 = fill_C(2, 1, v[6], v[5])
            C_p2 = fill_C(1, 0, v[2], v[1])
            C = np.vstack([C, C_p1, C_p2])
        # q, q_dot = root.assemble_q_q_dot()
        # print(C[:, : 12])
        # print(C @ q.T)

    V, V_inv = U(C)
    # V_inv = np.identity(n_dof, np.float64)
    print(np.max(V @ V_inv - np.identity(n_dof, np.float64)))
    
    # copied code ---------------------------------------
    mouse_staled = np.zeros(2, dtype=np.float64)
    diag_M = np.zeros((n_dof), np.float64)
    if isinstance(root, list):
        fill_M(root, diag_M)
    else :
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
            mouse_staled = np.zeros(2, dtype=np.float64)
        camera.position(*camera_pos)
        camera.lookat(*(camera_pos + camera_dir))
        scene.set_camera(camera)

        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        # copied code ---------------------------------------
        step(root, M, V_inv)
        if isinstance(root, list):
            for c  in root:
                c.mesh(scene)
            for c in root:
                c.particles(scene)
        else :
            root.mesh(scene)
            root.particles(scene)
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
