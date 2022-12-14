import taichi as ti
import numpy as np
from arp import Cube, skew
from scipy.linalg import lu, ldl, solve
import ipctk
from PSD_projection import project_PSD
from scipy.linalg.lapack import dsysv
ti.init(ti.x64, default_fp=ti.f64)

n_cubes = 3
hinge = True
gravity = -np.array([0., -9.8e1, 0.0])
articulated = False
m = 0 if not articulated else (n_cubes - 1) * 3 if not hinge else (n_cubes - 1) * 6
n_dof = 12 * n_cubes
delta = 0.08
centered = False
kappa = 1e9
kappa_ipc = 1.0e9
# kappa_ipc = 0.0
local = True
max_iters = 10
dim_grad = 4
grad_field = ti.Vector.field(3, float, shape=(dim_grad))
hess_field = ti.Matrix.field(3, 3, float, shape=(dim_grad, dim_grad))
dt = 5e-3
mass = 1e3
ZERO = 1e-9
a_cols = n_cubes * 4
a_rows = m // 3
alpha = 1.0
trace = False
dhat = 1e-4
a = ti.field(float, shape=(a_rows, a_cols)) if a_rows and a_cols else None
d = ti.field(float, shape=(a_rows, a_cols)) if a_rows and a_cols else None
display_particle = ti.Vector.field(3, ti.f32, shape = (10))
display_triangle = ti.Vector.field(3, ti.f32, shape = (30))
display_line_green = ti.Vector.field(3, ti.f32, shape = (n_cubes * 2))
display_line_yellow = ti.Vector.field(3, ti.f32, shape = (n_cubes * 2))
display_line_blue = ti.Vector.field(3, ti.f32, shape = (n_cubes * 2))
visual_dq = True
# dual matrix to get the inverse
disable_line_search = False

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
    if hinge and articulated:
        C[3:6, :] -= C[:3, :]
        C[3:6, :] *= -1.
        # no need to reorder for this selected hinge
        print(f'C[:, :12] = {C[:, : 12]} ')
    V[:m, :] = C
    V[m:, m:] = np.identity(n-m, np.float64)

    _V_inv = np.zeros_like(V)
    if d is not None:
        d.fill(0.0)
        a.fill(0.0)
        gaussian_elimination_row_pivot(C, _V_inv)
        print(f'a = {a.to_numpy()}')
        print(f'd = {d.to_numpy()}')
    _V_inv[m:, m:] = np.identity(n - m, np.float64)

    # V_inv = -V
    # V_inv[m:, m:] = np.identity(n - m, np.float64)
    # V_inv[:3, :3] = np.identity(3, np.float64)
    # if hinge:
    #     V_inv[3:6, 3:6] = np.identity(3, np.float64)
    #     V_inv[:3, 15:18] = np.zeros((3,3), np.float64)
    # print((V_inv - _V_inv)[:6])
    return V, _V_inv


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
        g += q[i] * (q[i].dot(q[i]) - 1)
        for j in range(1, 4):
            if j != i:
                g += q[j].dot(q[i]) * q[j]
        grad_field[i] = 4 * kappa * g

# @ti.kernel
# def grad_Eo(q: ti.template()):
#     for i in range(1, 4):
#         g = ti.Vector.zero(float, 3)
#         for j in range(1, 4):
#             g += (q[i].dot(q[j]) - kronecker(i, j)) * q[j]
#         grad_field[i] = 4 * kappa * g


# @ti.kernel
# def hess_Eo(q: ti.template()):
#     for i, j in ti.ndrange((1, 4), (1, 4)):
#         h = (q[j].dot(q[i]) - kronecker(i, j)) * ti.Matrix.identity(float, 3)
#         for k in range(1, 4):
#             h += (q[i] * kronecker(k, j) + q[k] *
#                   kronecker(i, j)) @ q[k].transpose()

#         hess_field[i, j] = 4 * kappa * h

@ti.kernel
def hess_Eo(q: ti.template()):
    for i, j in ti.ndrange((1, 4), (1, 4)):
        h = ti.Matrix.zero(float, 3, 3)
        if i == j: 
            h += 2 * (q[i] @ q[i].transpose() + (q[i].dot(q[i]) - 1) * ti.Matrix.identity(float, 3))
            for k in range(1, 4):
                if k != i:
                    h += q[k] @ q[k].transpose()
        else :
            h += ti.Matrix.identity(float, 3) * q[j].dot(q[i]) + q[j] @ q[i].transpose()
        hess_field[i, j] = 4 * kappa * h

@ti.kernel
def Eo(q: ti.template()) -> float:
    A = ti.Matrix.cols([q[1], q[2], q[3]])
    return kappa * (A.transpose() @ A - ti.Matrix.identity(float, 3)).norm_sqr()


@ti.data_oriented
class AffineCube(Cube):
    def __init__(self, id, scale=[1.0, 1.0, 1.0], omega=[0., 0., 0.], pos=[0., 0., 0.], vc=[0.0, 0.0, 0.0], parent=None, Newton_Euler=False, mass=1.0):
        super().__init__(id, scale=scale, omega=omega, pos=pos, vc=vc,
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
        self.q_dot[0] = self.v[None]

    def _reset(self):
        self.p[None] = ti.Vector(self.initial_state[0])
        self.omega[None] = ti.Vector(self.initial_state[1])
        self.init_q_q_dot()
        self.q0.copy_from(self.q)
        for c in self.children:
            c._reset()


    @ti.kernel
    def project_vertices(self):
        for i in ti.static(range(8)):
            A = ti.Matrix.cols([self.q[1], self.q[2], self.q[3]])
            self.v_transformed[i] = A @ self.vertices[i] + self.q[0]

    @ti.kernel
    def project_vertices_t2(self, dq: ti.types.ndarray()):
        for i in ti.static(range(8)):
            dq0 = ti.Vector([dq[0], dq[1], dq[2]])
            dq1 = ti.Vector([dq[3], dq[4], dq[5]])
            dq2 = ti.Vector([dq[6], dq[7], dq[8]])
            dq3 = ti.Vector([dq[9], dq[10], dq[11]])

            A = ti.Matrix.cols([self.q[1] + dq1, self.q[2] + dq2, self.q[3] + dq3])
            self.v_transformed[i] = A @ self.vertices[i] + (self.q[0] + dq0)

    def traverse(self, q, update_q = True, update_q_dot = True, project = True):
        traverse([self], q, update_q, update_q_dot, project)

    def fill_M(self, M):
        fill_M([self], M)

    def assemble_q_q_dot(self):
        return assemble_q_q_dot([self])

    @ti.kernel
    def gen_triangles(self, t: ti.types.ndarray()):
        for i, j in ti.ndrange(12, 3):
            
            I = self.indices[i * 3 + j]
            v = self.v_transformed[I]
            for k in ti.static(range(3)):
                t[i, j, k] = v[k]
            
# @ti.func
c1 = 1e-4
def line_search(dq, root, q0, grad0, q_tiled, M, idx = [], pts = [], cubes = []):

    global globals

    def v_barrier(pt):
        d2 = ipctk.point_triangle_distance(*pt)
        # d = np.sqrt(d2)
        return ipctk.barrier(d2, dhat)

    def Vb(q0, q_tiled, M, cubes, idx, dq = None, pts = []):
        # assert dq is not None or pts is not None, "specify at least one argument from dq and pts"
        global globals
        e = 0.0
        if dq is not None:
            pt_dq(dq, cubes)
            for ij in idx:
                pt = pt_t1_array(cubes, ij)
                e += v_barrier(pt)
        else :
            for pt in pts:
                e += v_barrier(pt)
        print(f'ipc energy at {"line search start" if dq is None else "iter"}, e = {e * kappa_ipc}')
        return np.sum(globals.Eo) * dt ** 2 + 0.5 * ((q0 - q_tiled) @ M @ (q0 - q_tiled).T)[0, 0] + e * kappa_ipc
        
    wolfe = False
    alpha = 1
    E0 = Vb(q0, q_tiled, M, cubes, idx, pts = pts)
    dq_norm_inf = np.max(np.abs(dq))
    dq_norm_2 = np.linalg.norm(dq)
    grad_norm_2 = np.linalg.norm(grad0)
    while not wolfe and grad_norm_2 > 1e-3:
        q1 = q0 + dq * alpha

        traverse(root, q1, True, False, False) 
        for c in root:
            E = Eo(c.q)
            i0 = c.id
            globals.Eo[i0] = E

        E1 = Vb(q1, q_tiled, M, cubes, idx, dq = np.zeros_like(dq))
        # print(dq.shape, grad0.shape)
        wolfe = E1 <= E0 + c1 * alpha * (dq @ grad0)[0, 0]
        alpha /= 2
        # assert alpha * dq_norm_inf >= 1e-10, f'''dq_norm_2 = {dq_norm_2}, grad norm = {grad_norm_2}, dq @ grad = {dq @ grad0}
        #     E1 = {E1:.2e}, alpha = {alpha:.2e}, wolfe rhs = {E0 + c1 * alpha * (dq @ grad0)}
        #     E0 - E1 = {E0 - E1:.2e}, requested descend = {c1 * alpha * (dq @ grad0)}
        #     wolfe = {wolfe}'''
        if alpha < 1e-2 or wolfe:
            print(f"line search: descend = {E1 - E0}, E0 = {E0:.2e}, E1 = {E1: .2e}")
            break
    return alpha * 2

def tiled_q(dt, q_dot_t, q_t, f_tp1):
    return q_t + dt * q_dot_t + dt ** 2 * f_tp1


def assemble_q_q_dot(cubes):
    q = np.zeros((1, n_dof))
    q_dot = np.zeros_like(q)
    for c in cubes:
        _q = c.q.to_numpy().reshape((1, -1))
        _q_dot = c.q_dot.to_numpy().reshape((1, -1))
        q[0, c.id * 12: (c.id + 1) * 12] = _q
        q_dot[0, c.id * 12: (c.id + 1) * 12] = _q_dot

        __q, __q_dot = assemble_q_q_dot(c.children)
        q+= __q
        q_dot += __q_dot
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

        Eo_differentials(c.children)


def pt_dq(dq, cubes):
    for c in cubes:
        i0 = c.id * 12
        dq_slice = dq[0, i0: i0 + 12].reshape((12))
        c.project_vertices_t2(dq_slice)
        c.p_t1 = c.v_transformed.to_numpy()
        c.t_t1 = np.zeros((12, 3 ,3))
        c.gen_triangles(c.t_t1)
        
        pt_dq(dq, c.children)

def pt_t1_array(cubes, ij):
    i, v, j, f  = ij

    p_t1 = cubes[i].p_t1[v]
    t0_t1 = cubes[j].t_t1[f, 0]
    t1_t1 = cubes[j].t_t1[f, 1]
    t2_t1 = cubes[j].t_t1[f, 2]
    
    pt_t1 = np.array([p_t1, t0_t1, t1_t1, t2_t1])
    return pt_t1

def step_size_upper_bound(dq, cubes, pts, idx):
    t = 1.0
    pt_dq(dq, cubes)
    for pt, ij in zip(pts, idx):
        # p, t0, t1, t2 = pt
        
        pt_t1 = pt_t1_array(cubes, ij)

        # assert (pt_t1 - pt < dhat * 3).all(), f"inf norm pt_t1 - pt = {np.max(np.abs(pt_t1 - pt))}"

        _t = 1.0
        _, _t = ipctk.point_triangle_ccd(
            *pt, *pt_t1)
        if _t < 1.0 :
            print(f'''
                |--------------------------------------------------------------------------------------------|
                |                                                                                            |
                |                  collision detected, toi, points = {_t}, {pt}, {pt_t1}                     |
                |                                                                                            |
                |--------------------------------------------------------------------------------------------|
                ''')
        t = min(t, _t)
    print(f'step size upper bound = {t}')
    return t


def compute_constraint_set(cubes):
    def pt_intersect(ci, cj, I, J):
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
                    if d2 < dhat * 9:
                        pts.append([p, t0, t1, t2 ])
                        idx.append([I, i, J, j])
        return pts, idx

    for c in cubes:
        c.t_t0 = np.zeros((12, 3, 3))
        c.project_vertices()
        c.p_t0 = c.v_transformed.to_numpy()
        # shape 8x3
        c.gen_triangles(c.t_t0)


    pt_set = []
    idx_set = []
    for i, ci in  enumerate(cubes):
        for j, cj in enumerate(cubes):
            if i == j :
                continue 
            pts, idx = pt_intersect(ci, cj, i, j)
            pt_set += pts
            idx_set += idx

    return np.array(pt_set), np.array(idx_set)
                
            
@ti.kernel
def candidacy(up:ti.types.ndarray(), lp:ti.types.ndarray(), ut:ti.types.ndarray(), lt:ti.types.ndarray(), ret:ti.types.ndarray()):
    for p, t in ti.ndrange(8, 12):
        upx = ti.Vector([up[p, 0], up[p, 1], up[p, 2]])
        lpx = ti.Vector([lp[p, 0], lp[p, 1], lp[p, 2]])

        utx = ti.Vector([ut[t, 0], ut[t, 1], ut[t, 2]])
        ltx = ti.Vector([lt[t, 0], lt[t, 1], lt[t, 2]])
        
        board_phase_candidacy = not ((upx < ltx).all() or (lpx > utx).all())

        ret[p, t] = board_phase_candidacy
    
                
def fill_M(cubes, M):
    for c in cubes:
        i0 = c.id * 12
        M[i0: i0 + 3] = c.m
        M[i0 + 3: i0 + 12] = c.Ic

        fill_M(c.children, M)


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
        
        traverse(c.children, q, update_q, update_q_dot, project)


def ipc_term(H, g, pts, idx, cubes):
    assert (H == H.T).all(), f'input hessian not symetric'
    vnp = cubes[0].vertices.to_numpy()
    inp = cubes[0].indices.to_numpy()
    print(f'before adding ipc term, norm = {np.linalg.norm(H)}')
    for pt, ij in zip(pts, idx):
        p, t0, t1, t2 = pt
        _i, v, _j, f = ij

        grad_d2 = ipctk.point_triangle_distance_gradient(p, t0, t1, t2)
        hess_d2 = ipctk.point_triangle_distance_hessian(p, t0, t1, t2)
        d = ipctk.point_triangle_distance(p, t0, t1, t2)
        # d = np.sqrt(d)
        grad = grad_d2# / (2 * d)
        grad = grad.reshape((12, 1))
        hess = hess_d2 #(hess_d2 / 2 - grad @ grad.T) / d
        assert (np.abs(hess - hess.T) < 1e-5).all(), 'laplacian d not symetric'
        B_ = ipctk.barrier_gradient(d, dhat) * kappa_ipc
        B__ = ipctk.barrier_hessian(d, dhat) * kappa_ipc
        # print(B_, B__)
        def jacobian(x):
            I = np.identity(3)
            return np.hstack([I, I * x[0], I * x[1], I * x[2]])
        t0_tile, t1_tile, t2_tile = vnp[inp[3 * f]], vnp[inp[3 * f + 1]], vnp[inp[3 * f + 2]]
        p_tile = vnp[v]
        Jt = np.vstack([jacobian(t0_tile), jacobian(t1_tile), jacobian(t2_tile)])
        Jp = jacobian(p_tile)

        grad_t = grad[3:]
        grad_p = grad[: 3]
        
        # ipc_hess = B_ * hess + B__ * grad @ grad.T
        # ipc_hess = project_PSD(ipc_hess)
        ipc_hess = ipctk.project_to_psd(B_ * hess) + B__ * grad @ grad.T

        hess_t = Jt.T @ (ipc_hess[3: , 3:]) @ Jt
        hess_p = Jp.T @ (ipc_hess[:3 , : 3]) @ Jp
        # off_diag = np.zeros((12, 12))
        off_diag = Jp.T @ (ipc_hess[:3, 3: ]) @ Jt
        
        # hess_t = Jt.T @ (B_ * hess[3:, 3:] + B__ * grad_t @ grad_t.T) @ Jt
        # hess_p = Jp.T @ (B_ * hess[:3, : 3] + B__ * grad_p @ grad_p.T) @ Jp

        # off_diag = Jp.T @ (B_ * hess[: 3, 3:] + B__ * grad_p @ grad_t.T) @ Jt

        # print(hess_t)
        # print(hess_p)
        # print(off_diag)

        # assert (hess_p == hess_p.T).all()

        # hess_t = project_PSD(hess_t)
        # hess_p = project_PSD(hess_p)
        # off_diag = project_PSD(off_diag)

        # FIXME: i should be cubes[i].id
        i, j = cubes[_i].id, cubes[_j].id
        assert i == _i and j == _j
        # assert (hess_t == hess_t.T).all()
        # assert (hess_p == hess_p.T).all()

        H[12 * i: 12 * (i + 1), 12 * i: 12 * (i + 1)] += hess_p
        H[12 * j: 12 * (j + 1), 12 * j: 12 * (j + 1)] += hess_t

        # print(np.max(np.abs(H - H.T)))
        # assert (np.abs(H - H.T) < 1e-5).all()

        H[12 * i: 12 * (i + 1), 12 * j: 12 * (j + 1)] += off_diag
        H[12 * j: 12 * (j + 1), 12 * i: 12 * (i + 1)] += off_diag.T
        # assert (np.abs(H - H.T) < 1e-5).all()

        g[i * 12: 12 * (i + 1)] += Jp.T @ grad_p  * B_
        g[j * 12: 12 * (j + 1)] += Jt.T @ grad_t  * B_

    print(f'after adding ipc term, norm = {np.linalg.norm(H)}')
    # assert (np.abs(H - H.T) < 1e-5).all(), "output hessian not symetric"


@ti.kernel
def vis(dq: ti.types.ndarray(), q0: ti.types.ndarray(), display_line: ti.template()):
    for i in range(n_cubes):
        for k in ti.static(range(3)):
            display_line[i * 2][k] = q0[0, i * 12 + k]
            display_line[i * 2 + 1][k] = q0[0, i * 12 + k] + dq[0, i * 12 + k]

def step(cubes, M, V_inv):
    global globals
    
    f = np.zeros((n_dof))
    q, q_dot = assemble_q_q_dot(cubes)
    q0 = np.copy(q)
    q_tiled = tiled_q(dt, q_dot, q, f)
    do_iter = True
    iter = 0
    pts, idx = compute_constraint_set(cubes)
    print(f'constraint set size = {len(pts)}')
    while do_iter:
        q, q_dot = assemble_q_q_dot(cubes)
        Eo_differentials(cubes)
        # grad = V_inv.T @ (globals.g.reshape((-1, 1)) *
        #                     dt ** 2 + M @ (q - q_tiled).T)

        # hess = V_inv.T @ (globals.H * dt ** 2 + M) @ V_inv
        grad = globals.g.reshape((-1, 1)) * dt ** 2 + M @ (q - q_tiled).T
        hess = globals.H * dt ** 2 + M

        grad0 = np.copy(grad)
        ipc_term(hess, grad, pts, idx, cubes)
        # print(hess, grad)
        # dq = -solve(hess, grad, assume_a="pos")
        _, _, dz, _ = dsysv(hess[m: , m:], grad[m :])
        dz = -dz
        dz = np.hstack([np.zeros((1, m), np.float64), dz.reshape(1, -1)])
        dq = dz #@ V_inv.T
        if trace:
            print(f'dq shape = {dq.shape}')
            print(f'hess shape = {hess.shape}')
            print(f'grad shape = {grad.shape}')
        
        print(f"norm(dq) = {np.max(np.abs(dq))}, grad = {np.linalg.norm(grad)}")
        
        
        dq = dq.reshape((1 , -1)) 
        if visual_dq and iter == 0:
            # vis(dq, q0)
            vis(-grad , q0, display_line_green)
            vis(-grad0 , q0, display_line_yellow)
            vis(dq * 1000, q0, display_line_blue)
        t = step_size_upper_bound(dq, cubes, pts, idx)
        dq *= t

        alpha = 1.0
        if disable_line_search or np.linalg.norm(dq) < 1e-4:
            q += dq 
            traverse(cubes, q, update_q=True, update_q_dot= False, project=False)
        else:
            alpha = line_search(dq, cubes, q, grad, q_tiled, M, idx, pts, cubes)
            print(f'line search: alpha = {alpha}, grad = {np.linalg.norm(grad)}')

            q += dq * alpha

        pt_dq(np.zeros_like(dq), cubes)
        for i, ij in enumerate(idx):
            pts[i] = pt_t1_array(cubes, ij)

        do_iter = np.linalg.norm(dq) > 1e-4 and iter < max_iters
        iter += 1

    print(f'\nconverge after {iter} iters')
    traverse(cubes, q)
    return pts

def step_link(root, M, V_inv):
    '''
    deprecated
    '''
    f = np.zeros((n_dof))
    # f[:3] = root.m * gravity
    # f[12: 15] = link.m * gravity
    q, q_dot = assemble_q_q_dot(root)
    q_tiled = tiled_q(dt, q_dot, q, f)
    do_iter = True
    iter = 0
    while do_iter:
    # for iter in range(max_iters):
        q, q_dot = assemble_q_q_dot(root)
        # shape = 1 * 24, transpose before use
        # root.grad_Eo_top_down()
        # root.hess_Eo_top_down()
        # root.Eo_top_down()
        Eo_differentials(root)
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
        root[0].traverse(q, update_q = True, update_q_dot = False, project = False)
        do_iter = np.linalg.norm(dq * alpha) > 1e-4 and iter < max_iters
        iter += 1
        if not do_iter: 
            print(f'converge after {iter} iters')
            
    root[0].traverse(q)


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

    _root = AffineCube(0, omega=[10., 0., 0.], mass = mass)
    pos = [0., -1., 1.] if hinge else [-1., -1., -1.] if not centered else [-0.5, -0.5, -0.5]
    pos2 = [1.1, 0.8, 0.8] if not articulated else [0., -2., 2.]
    vc2 = [-2.0, 0.0, 0.0]

    if not articulated:
        pos[2] += 0.01 * 0.9
        # pos = [0.0, 100.0, 0.0]
    link = None if n_cubes < 2 else AffineCube(
        1, omega=[-10., 0., 0.], pos=pos, parent=_root if articulated else None, mass = mass)

    link2 = None if n_cubes < 3 else AffineCube(
        2, omega=[0., 0., 0.], pos=pos2, vc=vc2, parent=link if articulated else None, mass=mass)

    C = np.zeros((m, n_dof), np.float64) if link is None else fill_C(
        1, 0, -link.r_lk_hat.to_numpy(), link.r_pkl_hat.to_numpy())
    
    root = [_root] if articulated or link is None else [_root, link]
    if link2 is not None and not articulated:
        root.append(link2)
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
    V = V_inv = np.identity(n_dof, np.float64)
    if articulated:
        V, V_inv = U(C)

    # print(np.max(V @ V_inv - np.identity(n_dof, np.float64)))
    
    # copied code ---------------------------------------
    mouse_staled = np.zeros(2, dtype=np.float64)
    diag_M = np.zeros((n_dof), np.float64)
    if isinstance(root, list):
        fill_M(root, diag_M)
    else :
        root.fill_M(diag_M)
    M = np.diag(diag_M)
    # print(diag_M)
    ts = 0
    pause = False
    green , yellow, blue = True, True, True
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
            camera_pos += delta * camera_dir
        if window.is_pressed('s'):
            camera_pos -= delta * camera_dir
        if window.is_pressed('r'):
            if articulated: 
                root._reset()
            else: 
                for c in root:
                    c._reset()
        if window.is_pressed('p'):
            pause = not pause
        
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
        if window.is_pressed('z'):
            green = True
        if window.is_pressed('x'):
            yellow = True
        if window.is_pressed('c'):
            blue = True 
        # if window.is_pressed('1'):
        #     green = not green
        
        camera.position(*camera_pos)
        camera.lookat(*(camera_pos + camera_dir))
        scene.set_camera(camera)

        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        # copied code ---------------------------------------
        if not pause:
            pts = step(root, M, V_inv)
            # step_link(root, M, V_inv)
            ts += 1
            print(f'timestep = {ts}\n')

        for c  in root:
            c.mesh(scene)
        
        dp0 = np.zeros((10, 3)) - 300
        dt0 = np.zeros((30, 3)) - 300
        def add_dp(pts, dp0, dt0):
            pts = np.array(pts)[:10]
            dp = pts[:10, 0]
            dt = pts[:10, 1:]
            
            l = dp.shape[0]
            dp0[:l] = dp
            dt0[:3 * l] = dt.reshape((3 * l, 3))
            return dp0, dt0
        if len(pts) :
            add_dp(pts, dp0, dt0)
            pause = True
        display_particle.from_numpy(dp0)
        display_triangle.from_numpy(dt0)
        if green:
            scene.lines(display_line_green, width = 20, color = (0.0, 1.0, 0.0))
        if yellow:
            scene.lines(display_line_yellow, width = 20, color = (1.0, 1.0, 0.0))
        if blue:
            scene.lines(display_line_blue, width = 20, color = (0.0, 0.0, 1.0))
        scene.particles(display_particle, radius = 0.05, color = (1.0, 0.0, 0.))
        scene.mesh(display_triangle, color = (0.7, 0.3, 0.3), show_wireframe= True)
            
        # for c in root:
        #     c.particles(scene)
        canvas.scene(scene)
        if local:
            window.show()

        elif not local and ts % 10 == 0:
            window.save_image(f'{ts // 10}.png')

        green, yellow, blue = False, False, False

if __name__ == "__main__":
    main()
