from re import L
import taichi as ti
from taichi import cos, sin
import numpy as np

# FIXME: drifting of the R matrix

ti.init(arch=ti.x64, default_fp=ti.f64)

lagrange = False
delta = 0.08
per_trace = 10
trajectory = ti.Vector.field(3, float, shape=(80))
gravity = np.array([0.0, -9.8, 0.0], dtype = np.float64)
n_cubes = 2
m = 3 * (n_cubes - 1) 
# n_constraints
n_dofs = 3 * n_cubes + 3 if lagrange else 6 * n_cubes
# n_3x3blocks = 5 * n_cubes - 3
centered = False


class Globals:
    def __init__(self):

        self.Jw_k = np.zeros((3, n_dofs), dtype=np.float64) if lagrange else None
        self.Jw_pk = np.zeros((3, n_dofs), dtype=np.float64) if lagrange else None
        self.Jv_k = np.zeros((3, n_dofs), dtype=np.float64) if lagrange else None

        self.Jw_pk_dot = np.zeros((3, n_dofs), dtype=np.float64) if lagrange else None
        self.Jw_k_dot = np.zeros((3, n_dofs), dtype=np.float64) if lagrange else None
        self.Jv_k_dot = np.zeros((3, n_dofs), dtype=np.float64) if lagrange else None

        self.M = np.zeros((n_dofs, n_dofs), np.float64) if lagrange else None
        self.C = np.zeros_like(self.M) if lagrange else None
        self.f = np.zeros((n_dofs), np.float64) if lagrange else None
        self.q_dot = np.zeros((n_dofs), np.float64)
        self.q = np.zeros((n_dofs), np.float64) 

        self.Jc = np.zeros((m, n_dofs), np.float64) if not lagrange else None
        self.Jc_dot = np.zeros((m, n_dofs), np.float64) if not lagrange else None
        
        # self.R0q_k = ti.Matrix.field(3,3, float, shape = (n_cubes * 3))
        # self.R0q_pk = ti.Matrix.field(3,3, float, shape = (n_cubes * 3))

globals = Globals()

@ti.func
def skew(r):
    ret = ti.Matrix.zero(float, 3, 3)
    ret[0, 1] = -r[2]
    ret[1, 0] = +r[2]
    ret[0, 2] = +r[1]
    ret[2, 0] = -r[1]
    ret[1, 2] = -r[0]
    ret[2, 1] = +r[0]
    return ret

@ti.func
def unskew(R):
    ret = ti.Vector.zero(float, 3)
    ret[0] = R[2, 1]
    ret[1] = - R[2, 0]
    ret[2] = R[1, 0]
    return ret

@ti.func
def rotation(a, b, c):
    '''
    Tait-Bryan angle in ZYX order
    '''
    s1 = sin(a)
    s2 = sin(b)
    s3 = sin(c)

    c1 = cos(a)
    c2 = cos(b)
    c3 = cos(c)

    R = ti.Matrix([
        [c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
        [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
        [-s2, c2 * s3, c2 * c3]
    ])
    return R


@ti.func
def rotation_dot(a, b, c, d1, d2, d3):
    s1 = sin(a)
    s2 = sin(b)
    s3 = sin(c)

    c1 = cos(a)
    c2 = cos(b)
    c3 = cos(c)

    R = ti.Matrix([
        [c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
        [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
        [-s2, c2 * s3, c2 * c3]
    ])
    pR_pa = ti.Matrix([
        [-s1 * c2, -s1 * s2 * s3 - c3 * c1, c1 * s3 + -s1 * c3 * s2],
        [c2 * c1, -s1 * c3 + c1 * s2 * s3, c3 * c1 * s2 - -s1 * s3],
        [0, 0, 0]
    ])
    pR_pb = ti.Matrix([
        [c1 * -s2, c1 * c2 * s3, c1 * c3 * c2],
        [-s2 * s1, s1 * c2 * s3, c3 * s1 * c2],
        [-c2, -s2 * s3, -s2 * c3]
    ])
    pR_pc = ti.Matrix([
        [0, c1 * s2 * c3 - -s3 * s1, s1 * c3 + c1 * -s3 * s2],
        [0, c1 * -s3 + s1 * s2 * c3, -s3 * s1 * s2 - c1 * c3],
        [0, c2 * c3, c2 * -s3]
    ])
    R_dot = pR_pa * d1 + pR_pb * d2 + pR_pc * d3
    return R_dot

@ti.kernel
def skew_Rr(R0: ti.template(), 
    r0: float, 
    r1: float, 
    r2: float, 
    ret: ti.types.ndarray()):
    _r = ti.Vector([r0, r1, r2])

    R = R0[None]
    M = skew(R @ _r)

    for i, j in ti.static(ti.ndrange(3, 3)):
        ret[i, j] = M[i, j]

@ti.kernel
def wR_dot_r(R0: ti.template(), q_dot: ti.template(), 
    r0: float, 
    r1: float, 
    r2: float, 
    ret: ti.types.ndarray()):
    '''
    
    '''
    omega = ti.Vector([q_dot[None][3], q_dot[None][4], q_dot[None][5]])
    r = ti.Vector([r0, r1, r2])
    R = R0[None]
    M = skew(skew(omega) @ R @ r)

    for i, j in ti.static(ti.ndrange(3, 3)):
        ret[i, j] = M[i, j]

# @ti.kernel
# def field_Mvp(M: ti.template(), v: ti.Vector, ret: ti.types.ndarray()):
#     '''
#     multiply the whole Matrix field M by the same vector v
#     '''

#     for i in M:
#         Miv = M[i] @ v
#         for k in ti.static(range(3)):
#             ret[k, i * 2  + 1] = Miv[k]
        
@ti.func
def Jw(a, b, c):
    '''
    returns the right half of J_omega as the left half is constant 0
    '''
    s1 = sin(a)
    s2 = sin(b)
    s3 = sin(c)

    c1 = cos(a)
    c2 = cos(b)
    c3 = cos(c)

    R = ti.Matrix([
        [c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
        [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
        [-s2, c2 * s3, c2 * c3]
    ])
    pR_pa = ti.Matrix([
        [-s1 * c2, -s1 * s2 * s3 - c3 * c1, c1 * s3 + -s1 * c3 * s2],
        [c2 * c1, -s1 * c3 + c1 * s2 * s3, c3 * c1 * s2 - -s1 * s3],
        [0, 0, 0]
    ])
    pR_pb = ti.Matrix([
        [c1 * -s2, c1 * c2 * s3, c1 * c3 * c2],
        [-s2 * s1, s1 * c2 * s3, c3 * s1 * c2],
        [-c2, -s2 * s3, -s2 * c3]
    ])
    pR_pc = ti.Matrix([
        [0, c1 * s2 * c3 - -s3 * s1, s1 * c3 + c1 * -s3 * s2],
        [0, c1 * -s3 + s1 * s2 * c3, -s3 * s1 * s2 - c1 * c3],
        [0, c2 * c3, c2 * -s3]
    ])
    ja = unskew(pR_pa @ R.transpose())
    jb = unskew(pR_pb @ R.transpose())
    jc = unskew(pR_pc @ R.transpose())
    return ti.Matrix.cols([ja, jb, jc])


@ti.func
def J_dot(a, b, c, d1, d2, d3):
    s1 = sin(a)
    s2 = sin(b)
    s3 = sin(c)

    c1 = cos(a)
    c2 = cos(b)
    c3 = cos(c)

    R = ti.Matrix([
        [c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
        [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
        [-s2, c2 * s3, c2 * c3]
    ])
    pR_pa = ti.Matrix([
        [-s1 * c2, -s1 * s2 * s3 - c3 * c1, c1 * s3 + -s1 * c3 * s2],
        [c2 * c1, -s1 * c3 + c1 * s2 * s3, c3 * c1 * s2 - -s1 * s3],
        [0, 0, 0]
    ])
    pR_pb = ti.Matrix([
        [c1 * -s2, c1 * c2 * s3, c1 * c3 * c2],
        [-s2 * s1, s1 * c2 * s3, c3 * s1 * c2],
        [-c2, -s2 * s3, -s2 * c3]
    ])
    pR_pc = ti.Matrix([
        [0, c1 * s2 * c3 - -s3 * s1, s1 * c3 + c1 * -s3 * s2],
        [0, c1 * -s3 + s1 * s2 * c3, -s3 * s1 * s2 - c1 * c3],
        [0, c2 * c3, c2 * -s3]
    ])
    R_dot = pR_pa * d1 + pR_pb * d2 + pR_pc * d3

    pR2_paa = ti.Matrix([
        [-c1 * c2, -c1 * s2 * s3 - c3 * -s1, -s1 * s3 + -c1 * c3 * s2],
        [c2 * -s1, -c1 * c3 + -s1 * s2 * s3, c3 * -s1 * s2 - -c1 * s3],
        [0, 0, 0]
    ])

    pR2_pab = ti.Matrix([
        [-s1 * -s2, -s1 * c2 * s3, -s1 * c3 * c2],
        [-s2 * c1, c1 * c2 * s3, c3 * c1 * c2],
        [0, 0, 0]
    ])
    pR2_pbb = ti.Matrix([
        [c1 * -c2, c1 * -s2 * s3, c1 * c3 * -s2],
        [-c2 * s1, s1 * -s2 * s3, c3 * s1 * -s2],
        [s2, -c2 * s3, -c2 * c3]
    ])

    pR2_pac = ti.Matrix([
        [0, -s1 * s2 * c3 - -s3 * c1, c1 * c3 + -s1 * -s3 * s2],
        [0, -s1 * -s3 + c1 * s2 * c3, -s3 * c1 * s2 - -s1 * c3],
        [0, 0, 0]
    ])
    pR2_pbc = ti.Matrix([
        [0, c1 * c2 * c3, c1 * -s3 * c2],
        [0, s1 * c2 * c3, -s3 * s1 * c2],
        [0, -s2 * c3, -s2 * -s3]
    ])
    pR2_pcc = ti.Matrix([
        [0, c1 * s2 * -s3 - -c3 * s1, s1 * -s3 + c1 * -c3 * s2],
        [0, c1 * -c3 + s1 * s2 * -s3, -c3 * s1 * s2 - c1 * -s3],
        [0, c2 * -s3, c2 * -c3]
    ])

    dpR_pa = pR2_paa * d1 + pR2_pab * d2 + pR2_pac * d3
    dpR_pb = pR2_pab * d1 + pR2_pbb * d2 + pR2_pbc * d3
    dpR_pc = pR2_pac * d1 + pR2_pbc * d2 + pR2_pcc * d3

    ja_dot = unskew(dpR_pa @ R.transpose() + pR_pa @ R_dot.transpose())
    jb_dot = unskew(dpR_pb @ R.transpose() + pR_pb @ R_dot.transpose())
    jc_dot = unskew(dpR_pc @ R.transpose() + pR_pc @ R_dot.transpose())
    return ti.Matrix.cols([ja_dot, jb_dot, jc_dot])

# @ti.func
# def fill_3x3(J, A, i):
#     '''
#     J: 3 * 6n
#     A: 3 * 3
#     i: 
#     '''
#     I = 3 * i
#     for i, j in ti.ndrange(3, 3):
#         J[i, I + j] = A[i, j]

# @ti.func
# def load_3x3(A):
#     ret = ti.Matrix.zero(float, 3,3)
#     # for i, j in ti.ndrange(3, 3):
#     #     ret[i, j] = A[i, j]
#         # print(A[i, j])
#     return ret


@ti.func
def load_3x3(A):
    return A[None]


@ti.data_oriented
class Cube:
    def __init__(self, id, scale=[1.0, 1.0, 1.0], omega=[0., 0., 0.], pos=[0., 0., 0.], vc = [0.0, 0.0, 0.0], parent=None, Newton_Euler=False, mass = 1.0):

        # generalized coordinates
        self.p = ti.Vector.field(3, float, shape=())
        self.v = ti.Vector.field(3, float, shape=())
        self.R = ti.Matrix.field(3, 3, float, shape=())

        self.R0 = ti.Matrix.field(3, 3, float, shape=())
        self.R0_dot = ti.Matrix.field(3, 3, float, shape=())
        # self.R0 = np.zeros((3, 3), dtype = np.float64)
        # self.R0_dot = np.zeros((3, 3), dtype = np.float64)
        self.a1 = np.zeros((3, 3), dtype=np.float64)
        self.a2 = np.zeros((3, 3), dtype=np.float64)

        self.q = ti.Vector.field(6, float, shape=())
        self.q_dot = ti.Vector.field(6, float, shape=())
        # self.J_dot = ti.Matrix.field(6, 6, float, shape = ())
        self.omega = ti.Vector.field(3, float, shape=())

        self.initial_state = [pos, omega, vc]
        # constants
        self.scale = scale
        self.m = mass
        # self.Ic = ti.Matrix.diag(3, self.m / 12 * scale[0] ** 2)
        self.Ic = self.m / 12 * scale[0] ** 2

        self.id = id

        self.v_transformed = ti.Vector.field(3, ti.f32, shape=(8))
        self.vertices = ti.Vector.field(3, float, shape=(8))
        self.indices = ti.field(ti.i32, shape=(3 * 12))
        self.faces = ti.Vector.field(4, ti.i32, shape=(6))

        self.gen_v()
        self.gen_id()

        self.parent = parent
        self.r_pkl_hat = self.vertices[0]
        # parent center to link
        # self.r_lk_hat = -self.vertices[7]
        self.r_lk_hat = ti.Vector([0.0, 0.0, 0.0], float) if centered else -self.vertices[7]
        # link to center
        self.children = []
        if self.parent is not None:
            self.parent.children.append(self)

        self.reset()
        self.substep = self.midpoint if Newton_Euler else self.lagrange_midpoint
        self.top_down = self.top_down_lagrange if lagrange else self.top_down_constrained

    @ti.kernel
    def initialize(self):
        self.R[None] = ti.Matrix.identity(float, 3)

        self.R0[None] = ti.Matrix.identity(float, 3)
        self.R0_dot[None] = ti.Matrix.zero(float, 3, 3)

        # self.set_Mc()
        # self.q_dot[None][0] = 1.0
        for k in ti.static(range(3)):
            self.q[None][k] = self.p[None][k]
            self.q[None][k + 3] = 0.0
            self.q_dot[None][k] = 0.0
        self.q_dot[None][3] = self.omega[None][0]
        self.q_dot[None][4] = self.omega[None][1]
        self.q_dot[None][5] = self.omega[None][2]

    @ti.kernel
    def midpoint(self):
        '''
        Newton-Euler Formulation
        midpoint time integral
        order of 2

        '''
        dt = 1e-3
        dv = f(self.p[None], self.R[None]) * dt / 2 / self.m
        dp = self.v[None] * dt
        domega = tau(self.p[None], self.R[None]) * dt / 2 / self.Ic
        dq = skew(self.omega[None]) @ self.R[None] * dt / 2

        v_mid = dv + self.v[None]
        p_mid = dp + self.p[None]
        omega_mid = domega + self.omega[None]
        q_mid = dq + self.R[None]

        self.v[None] += f(p_mid, q_mid) * dt / self.m
        self.p[None] += v_mid * dt
        self.omega[None] += tau(p_mid, q_mid) * dt / self.Ic
        self.R[None] += skew(omega_mid) @ q_mid * dt

        for i in range(8):
            self.v_transformed[i] = self.R[None] @ self.vertices[i] + self.p[None]

    @ti.func
    def lagrange_explicit_euler(self, q_dot, q):
        '''
        ret = dydt
        treat rotation and translation seperately in two 3*3 jacobian matrix
        '''
        _Jw = Jw(q[3], q[4], q[5])

        # M = block_diag(ti.Matrix.identity(float, 3) * self.m, _Jw.transpose() @ _Jw * self.Ic)
        # M = J.transpose() @ self.Mc[None] @ J
        tiled_omega_BR = skew(_Jw @ ti.Vector([q_dot[3], q_dot[4], q_dot[5]]))
        # tiled_omega = block_diag(ti.Matrix.zero(float, 3,3), tiled_omega_BR)
        # C = dot_block_diag(block_diag(ti.Matrix.zero(float, 3, 3), _Jw.transpose() * self.Ic @ J_dot(q[3], q[4], q[5], q_dot[3], q_dot[4], q_dot[5]) + _Jw.transpose() @ tiled_omega_BR * self.Ic @ _Jw), q_dot)

        # sovle M q.. + C = Q

        # q += q_dot
        # q_dot += solve_block_diag(M, -C)

        q_dot_omega = ti.Vector([q_dot[3], q_dot[4], q_dot[5]])
        # q_dot_v = ti.Vector([q_dot[0], q_dot[1], q_dot[2]])
        C_omega = (_Jw.transpose() * self.Ic @ J_dot(q[3], q[4], q[5], q_dot[3], q_dot[4],
                   q_dot[5]) + _Jw.transpose() @ tiled_omega_BR * self.Ic @ _Jw) @ q_dot_omega
        M_omega = _Jw.transpose() @ _Jw * self.Ic

        __q = M_omega.inverse() @ -C_omega
        # FIXME: add translation term
        return q_dot, ti.Vector([0, 0, 0, __q[0], __q[1], __q[2]])

    @ti.kernel
    def lagrange_midpoint(self):
        '''
        Lagrange Formulation
        6 DoFs all stacked together as q
        3 COM velocities + 3 euler angles (rotation X1 Z2 X3)
        '''
        dt = 1e-4
        dq, dq_dot = self.lagrange_explicit_euler(
            self.q_dot[None], self.q[None])

        q_mid = self.q[None] + dq * dt / 2
        q_dot_mid = self.q_dot[None] + dq_dot * dt / 2

        dq, dq_dot = self.lagrange_explicit_euler(q_dot_mid, q_mid)
        self.q[None] += dq * dt
        self.q_dot[None] += dq_dot * dt
        r = rotation(self.q[None][3], self.q[None][4], self.q[None][5])

        if ti.static(self.parent is not None):
            R0_pk = load_3x3(self.parent.R0)
            r = R0_pk @ r
        p = ti.Vector([self.q[None][0], self.q[None][1], self.q[None][2]])

        for i in ti.static(range(8)):
            self.v_transformed[i] = r @ self.vertices[i] + p

    @ti.kernel
    def gen_v(self):
        for i, j, k in ti.ndrange(2, 2, 2):
            I = i * 4 + j * 2 + k
            self.vertices[I] = ti.Vector([i - 0.5, j - 0.5, k - 0.5])

    @ti.kernel
    def gen_id(self):
        self.faces[0] = ti.Vector([0, 1, 3, 2])
        self.faces[1] = ti.Vector([4, 5, 1, 0])
        self.faces[2] = ti.Vector([2, 3, 7, 6])
        self.faces[3] = ti.Vector([4, 0, 2, 6])
        self.faces[4] = ti.Vector([1, 5, 7, 3])
        self.faces[5] = ti.Vector([5, 4, 6, 7])
        for i in range(6):
            self.indices[i * 6 + 0] = self.faces[i][0]
            self.indices[i * 6 + 1] = self.faces[i][1]
            self.indices[i * 6 + 2] = self.faces[i][2]
            self.indices[i * 6 + 3] = self.faces[i][2]
            self.indices[i * 6 + 4] = self.faces[i][3]
            self.indices[i * 6 + 5] = self.faces[i][0]

    @ti.kernel
    def fill_Jwk(self, Jw_k: ti.types.ndarray()):
        '''
        side effect: update R0 
        '''
        dJw = ti.Matrix.zero(float, 3, 3)
        R0 = ti.Matrix.zero(float, 3, 3)
        if ti.static(self.parent is not None):
            R0_pk = load_3x3(self.parent.R0)

            dJw = R0_pk @ Jw(self.q[None][3], self.q[None][4], self.q[None][5])
            R0 = R0_pk @ rotation(self.q[None][3], self.q[None]
                          [4], self.q[None][5])
        else:
            # root node
            dJw = Jw(self.q[None][3], self.q[None][4], self.q[None][5])
            R0 = rotation(self.q[None][3], self.q[None][4], self.q[None][5])

        self.R0[None] = R0
        for i, j in ti.static(ti.ndrange(3, 3)):
            Jw_k[i, j + 3 * (self.id + 1)] = dJw[i, j]

    @ti.kernel
    def coeff_Jw_pk_Jw_k(self, Jv_k: ti.types.ndarray(), a1: ti.types.ndarray()):
        R0_pk = ti.Matrix.identity(float, 3) if ti.static(
            self.parent is None) else load_3x3(self.parent.R0)
        R0_k = load_3x3(self.R0)

        q = self.q[None]
        R = rotation(q[3], q[4], q[5])
        dJv = -R0_pk @ skew(R @ self.r_lk_hat) @ Jw(q[3], q[4], q[5])
        _a1 = skew(R0_pk @ self.r_pkl_hat + R0_k @ self.r_lk_hat)
        
        for i, j in ti.static(ti.ndrange(3, 3)):
            a1[i, j] = _a1[i, j]
            Jv_k[i, j + 3 * (self.id + 1)] += dJv[i, j]


    def fill_Jvk(self):
        global globals
        if self.parent is not None:
            self.coeff_Jw_pk_Jw_k(globals.Jv_k, self.a1)
            globals.Jv_k -= self.a1 @ globals.Jw_pk
        else:
            globals.Jv_k[:, : 3] = np.identity(3, np.float64)

    @ti.kernel
    def fill_J_dot_related(self, a1_dot: ti.types.ndarray(), a2_dot: ti.types.ndarray(), Jw_hat: ti.types.ndarray(), Jw_dot_hat: ti.types.ndarray(), tiled_omega_BR: ti.types.ndarray()):
        '''
        side effect: update R0_dot
        '''
        q = self.q[None]
        q_dot = self.q_dot[None]

        _Jw_hat = Jw(q[3], q[4], q[5])
        _Jw_dot_hat = J_dot(q[3], q[4], q[5], q_dot[3], q_dot[4], q_dot[5])
        # fill_3x3(Jw_hat, _Jw_hat, 0)
        # fill_3x3(Jw_dot_hat, _Jw_dot_hat, 0)

        R_dot = rotation_dot(q[3], q[4], q[5], q_dot[3], q_dot[4], q_dot[5])
        R = rotation(q[3], q[4], q[5])

        # R0_pk = load_3x3(self.parent.R0) if ti.static(self.parent is not None) else ti.Matrix.identity(float, 3)
        # R0_dot_pk = load_3x3(self.parent.R0_dot) if ti.static(self.parent is not None) else ti.Matrix.zero(float,3, 3)
        R0_pk = ti.Matrix.identity(float, 3)
        R0_dot_pk = ti.Matrix.zero(float, 3, 3)
        if ti.static(self.parent is not None):
            R0_pk = load_3x3(self.parent.R0)
            R0_dot_pk = load_3x3(self.parent.R0_dot)

        R0_k_dot = R0_pk @ R_dot + R0_dot_pk @ R
        # fill_3x3(self.R0_dot, R0_k_dot, 0)

        _a1_dot = skew(R0_dot_pk @ self.r_pkl_hat)
        _a2_dot = skew(R0_k_dot @ self.r_lk_hat)
        # fill_3x3(a1_dot, _a1_dot, 0)
        # fill_3x3(a2_dot, _a2_dot, 0)

        _tiled_omega_BR = skew(
            _Jw_hat @ ti.Vector([q_dot[3], q_dot[4], q_dot[5]]))
        # fill_3x3(tiled_omega_BR, _tiled_omega_BR, 0)

        self.R0_dot[None] = R0_k_dot
        for i, j in ti.static(ti.ndrange(3, 3)):
            Jw_hat[i, j] = _Jw_hat[i, j]
            Jw_dot_hat[i, j] = _Jw_dot_hat[i, j]
            a1_dot[i, j] = _a1_dot[i, j]
            a2_dot[i, j] = _a2_dot[i, j]
            tiled_omega_BR[i, j] = _tiled_omega_BR[i, j]

    def aggregate_JkT_Mck_Jk(self):
        global globals
        ul = globals.Jv_k.T @ globals.Jv_k * self.m
        br = globals.Jw_k.T @ globals.Jw_k * self.Ic
        # M q.. + C = Q
        globals.M += ul + br
        # print(globals.M, ul, br)

    def aggregate_JkT_Mck_Jk_dot(self, ):
        '''
        fill Jw_dot, Jv_dot

        '''
        global globals
        a1_dot = np.zeros_like(self.a1)
        a2_dot = np.zeros_like(self.a2)
        Jw_dot_hat = np.zeros((3, 3), np.float64)
        Jw_hat = np.zeros((3, 3), np.float64)
        tiled_omega_BR = np.zeros((3, 3), np.float64)

        self.fill_J_dot_related(a1_dot, a2_dot, Jw_hat,
                                Jw_dot_hat, tiled_omega_BR)
        R0_pk_dot = np.zeros(
            (3, 3), np.float64) if self.parent is None else self.parent.R0_dot.to_numpy()
        R0_pk = np.identity(
            3, dtype=np.float64) if self.parent is None else self.parent.R0.to_numpy()
        globals.Jw_k_dot = globals.Jw_pk_dot
        globals.Jw_k_dot[:, (self.id + 1) * 3: (self.id + 2)
                         * 3] += R0_pk_dot @ Jw_hat + R0_pk @ Jw_dot_hat
        # FIXME: offset, fixed
        # FIXME: change R0 and R0 dot to numpy arrays, fixed: not possible
        globals.Jv_k_dot = globals.Jv_k_dot - self.a1 @ globals.Jw_pk_dot - \
            self.a2 @ globals.Jw_k_dot - a1_dot @ globals.Jw_pk - a2_dot @ globals.Jw_k

        ul = globals.Jv_k.T @ globals.Jv_k_dot * self.m
        br = globals.Jw_k.T @ (globals.Jw_k_dot +
                               tiled_omega_BR @ globals.Jw_k) * self.Ic
        globals.C += ul
        globals.C += br

    @ti.kernel
    def update_q_dot(self, q__: ti.types.ndarray()):
        if ti.static(lagrange):
            i0 = (self.id + 1) * 3
            for i in ti.static(range(3)):
                self.q_dot[None][i + 3] += q__[i0 + i, 0]
        else :
            i0 = self.id * 6
            for i in ti.static(range(6)):
                self.q_dot[None][i] += q__[i0 + i]

    @ti.kernel
    def update_q(self, dt: float):
        q_ = self.q_dot[None]
        self.q[None] += self.q_dot[None] * dt
        omega = ti.Vector([q_[3], q_[4], q_[5]])
        self.R0[None] += skew(omega) @ self.R0[None] * dt

    def traverse(self, q__, dt=1e-4):
        '''
        recursively apply q..
        try explicit first 
        '''
        self.update_q(dt)
        self.update_q_dot(q__)
        for c in self.children:
            c.traverse(q__, dt)

    def q_dot_assemble(self):
        _q_dot = self.q_dot.to_numpy()

        q_dot_arr = _q_dot if not lagrange or self.parent is None else _q_dot[3:]
        for c in self.children:
            arr = c.q_dot_assemble()
            q_dot_arr = np.hstack([q_dot_arr, arr])
        return q_dot_arr

    @ti.kernel
    def project_vertices(self, dx: ti.types.ndarray()):
        # dt = 1e-4
        for i in ti.static(range(3)):
            if ti.static(lagrange):
                self.p[None][i] += dx[i, 0]
            else:
                self.p[None][i] = self.q[None][i]

        for i in ti.static(range(8)):
            self.v_transformed[i] = self.R0[None] @ self.vertices[i] + self.p[None]


    def aggregate_force(self):
        global globals
        
        df = self.m * gravity @ globals.Jv_k
        # print(df)
        if self.id > 0:
            globals.f += df



    def top_down_lagrange(self):
        '''
        unknowns layout:
        (q_0[0:5], q_1[3:5], q_2[3:5],..., q_n[3:5])
        '''
        global globals
        dt = 3e-4
        if self.parent is None:
            globals.q_dot = self.q_dot_assemble().reshape((-1, 1))
            # print(globals.q_dot)
            globals.Jv_k = np.zeros_like(globals.Jv_k)
            globals.Jw_k = np.zeros_like(globals.Jw_k)
            globals.Jw_pk = np.zeros_like(globals.Jw_pk)

            globals.Jv_k_dot = np.zeros_like(globals.Jv_k_dot)
            globals.Jw_k_dot = np.zeros_like(globals.Jw_k_dot)
            globals.Jw_pk_dot = np.zeros_like(globals.Jw_pk_dot)

            globals.M = np.zeros_like(globals.M)
            globals.C = np.zeros_like(globals.C)
            globals.f = np.zeros_like(globals.f)

        globals.Jw_k = 0 + globals.Jw_pk
        self.fill_Jwk(globals.Jw_k)
        self.fill_Jvk()
        self.aggregate_JkT_Mck_Jk()
        self.aggregate_JkT_Mck_Jk_dot()
        self.aggregate_force()
        globals.Jw_pk_dot = globals.Jw_k_dot
        globals.Jw_pk = globals.Jw_k

        # self.substep()
        dxc = globals.Jv_k @ globals.q_dot * dt
        
        self.project_vertices(dxc)
        # if self.id == 1 or self.id == 0:
        #     print(self.id)
        #     # print(dxc.reshape((1, -1)))
        #     print(globals.Jv_k[:, -3:])
        #     # print(globals.q_dot.reshape((1, -1)))
        #     print("")

        # FIXME: support for tree (now only suitable for chain)
        for c in self.children:
            c.top_down_lagrange()

        if self.parent is None:
            # root do the finish-up
            q__ = np.linalg.solve(globals.M, -globals.C @ globals.q_dot + globals.f.reshape((-1, 1)))
            # print("q.. = ", q__.reshape(1, -1))
            # print(globals.f - (globals.C @ globals.q_dot).reshape(1, -1))
            # print("f = ", globals.f)
            self.traverse(q__ * dt, dt)

    def reset(self):
        self.p[None] = ti.Vector(self.initial_state[0])
        self.omega[None] = ti.Vector(self.initial_state[1])
        self.v[None] = ti.Vector(self.initial_state[2])
        self.initialize()
        for c in self.children:
            c.reset()

    def mesh(self, scene):
        scene.mesh(self.v_transformed, self.indices, two_sided=True, show_wireframe=False)
        for c in self.children:
            c.mesh(scene)
            
    def particles(self, scene):
        scene.particles(self.v_transformed, radius=0.05, color=(1.0, 0.0, 0.0))
        for c in self.children:
            c.particles(scene)

    def fill_Jc(self):
        global globals
        '''
        add link to parent

        q layout:
        q_1[0:6], q_2[0:6],..., q_n[0:6], 
        '''
        
        pk = self.parent.id
        k = self.id
        # q_pk = self.parent.q
        # q_k = self.q
        Rr_pk = np.zeros((3,3), np.float64)
        Rr_k = np.zeros((3,3), np.float64)
        lines = np.zeros((3, n_dofs), np.float64)

        # lines = globals.Jc[3 * (k -1) * 3 : 3 * k, :]

        skew_Rr(self.parent.R0, 
            self.r_pkl_hat[0], 
            self.r_pkl_hat[1], 
            self.r_pkl_hat[2], 
            Rr_pk)

        skew_Rr(self.R0, 
            -self.r_lk_hat[0],
            -self.r_lk_hat[1],
            -self.r_lk_hat[2],
            Rr_k)
        lines[:, 6 * pk: 6 * pk + 3] = np.identity(3, np.float64) 
        lines[:, 6 * k: 6 * k + 3] = -np.identity(3, np.float64) 
        lines[:, 6 * pk + 3: 6 * pk + 6] = -Rr_pk
        lines[:, 6 * k + 3: 6 * k + 6] = +Rr_k

        globals.Jc[3 * (k -1) * 3 : 3 * k, :] = lines

    def fill_Jc_dot(self):
        global globals
        pk = self.parent.id
        k = self.id
        # q_pk = self.parent.q
        # q_k = self.q
        q_dot_pk = self.parent.q_dot
        q_dot_k = self.q_dot
        R_dot_r_pk = np.zeros((3,3), np.float64)
        R_dot_r_k = np.zeros((3,3), np.float64)
        lines = np.zeros((3, n_dofs), np.float64)

        wR_dot_r(self.parent.R0, q_dot_pk, 
            self.r_pkl_hat[0], 
            self.r_pkl_hat[1], 
            self.r_pkl_hat[2], 
            R_dot_r_pk)

        wR_dot_r(self.R0, q_dot_k, 
            -self.r_lk_hat[0], 
            -self.r_lk_hat[1], 
            -self.r_lk_hat[2], 
            R_dot_r_k)

        lines[:, 6 * pk + 3: 6 * pk + 6] = -R_dot_r_pk
        lines[:, 6 * k + 3: 6 * k + 6] = R_dot_r_k

        globals.Jc_dot[3 * (k -1) * 3 : 3 * k, :] = lines

    def fill_W(self, W):
        i0 = self.id * 6
        W[i0: i0 + 3] = np.ones(3, np.float64) / self.m
        W[i0 + 3: i0 + 6] = np.ones(3, np.float64) / self.Ic
        for c in self.children:
            c.fill_W(W)

    def solve_sytem(self):
        '''
        C.. = J. q. + J W Q
        q. = (v_c0, omega_0, ..., v_cn, omega_n)
        W = diag(1/m0, 1/Ic0, ..., )
        Q = (f0, tau0, ..., )

              p(k)              k
        J. = (0, -[[w]Rr], ..., 0 , -[[w]Rr])
        J  = (I, -[Rr],    ..., -I, -[[w]Rr])
        '''
        diag_W = np.zeros((n_dofs), np.float64)
        self.fill_W(diag_W)
        JcWJcT = globals.Jc @ np.diag(diag_W) @ globals.Jc.T

        C = np.zeros((3), np.float64)
        self.compute_C(C)
        lam = np.linalg.solve(JcWJcT, -globals.Jc_dot @ globals.q_dot - 100 * globals.Jc @ globals.q_dot - 1e4 * C) # - globals.Jc @ W @ Q)
        q__ = np.diag(diag_W) @ (globals.Jc.T @ lam) 
        # print(f"\nC.. = ")
        # print(globals.Jc_dot @ globals.q_dot + globals.Jc @ q__)

        # print(f"\nC. = ")
        # print(globals.Jc @ globals.q_dot)
        return q__

    @ti.kernel
    def compute_C(self, C: ti.types.ndarray()):
        xl_k = self.children[0].v_transformed[7]

        dx = self.v_transformed[0] - xl_k 
        for i in ti.static(range(3)):
            C[i] = dx[i]

        # print("C = ")
        # print(dx)

    def top_down_constrained(self):
        global globals
        dt = 3e-4

        if self.parent is None:
            globals.q_dot = self.q_dot_assemble()
            globals.Jc = np.zeros_like(globals.Jc)
            globals.Jc_dot = np.zeros_like(globals.Jc_dot)
        else :
            self.fill_Jc()
            self.fill_Jc_dot()

        self.project_vertices(np.zeros((1)))

        for c in self.children:
            c.top_down_constrained()

        if self.parent is None:
            q__ = self.solve_sytem()
            self.traverse(q__ * dt, dt)
        

arr = np.zeros(shape=(10, 8, 3))


def booknote(a):
    global arr
    arr[:-1] = arr[1:]
    arr[-1] = a
    trajectory.from_numpy(arr.reshape(-1, 3))
    # return arr.reshape(-1,3)


def main():
    window = ti.ui.Window(
        "Articulated Multibody Simualtion", (800, 800), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0, 0, 0))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera_pos = np.array([0.0, 0.0, 3.0])
    camera_dir = np.array([0.0, 0.0, -1.0])

    cube = Cube(0, omega=[10.0, 10.0, 10.0])
    link = None if n_cubes < 2 else Cube(1, omega=[10., 0., 0.], pos = [-1., -1., -1.] if not centered else [-0.5, -0.5, -0.5], parent= cube) 
    link3 = None if n_cubes < 3 else Cube(2, pos = [-2., -2., -2.], parent = link)
    root = cube

    mouse_staled = np.zeros(2, dtype=np.float64)
    ts = 0
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
            root.reset()
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
        for i in range(10):
            root.top_down()
            # cube.substep()
            # link.substep()

        root.mesh(scene)

        if ts % per_trace == 0:
            t = booknote(cube.v_transformed.to_numpy())
        scene.particles(trajectory, radius=0.01, color=(1.0, 0.0, 0.0))

        root.particles(scene)
        canvas.scene(scene)
        window.show()
        ts += 1
        # quit()

if __name__ == "__main__":
    main()
