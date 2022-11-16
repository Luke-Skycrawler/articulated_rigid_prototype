from re import L
import taichi as ti
from taichi import cos, sin
import numpy as np

# FIXME: drifting of the R matrix

ti.init(arch=ti.cuda, default_fp=ti.f32)

delta = 0.08
per_trace = 10
trajectory = ti.Vector.field(3, float, shape=(80))

n_cubes = 2
n_dofs = 3 * n_cubes + 3
n_3x3blocks = 5 * n_cubes - 3
# Jw_k = ti.linalg.SparseMatrix(n = 3, m = n_dofs, dtype = float)
# Jv_k = ti.linalg.SparseMatrix(n = 3, m = n_dofs, dtype = float)

# triplets = ti.Vector.ndarray(n = 3, dtype = float, shape = n_3x3blocks * 9, layout=ti.Layout.AOS)

Jw_k = np.zeros((3, n_dofs), dtype = np.float32)
Jw_pk = np.zeros((3, n_dofs), dtype = np.float32)
Jv_k = np.zeros((3, n_dofs), dtype = np.float32)

Jw_pk_dot =np.zeros((3, n_dofs), dtype = np.float32)
Jw_k_dot = np.zeros((3, n_dofs), dtype = np.float32)
Jv_k_dot = np.zeros((3, n_dofs), dtype = np.float32)

M_ul = np.zeros((n_dofs, n_dofs), np.float32)
M_br = np.zeros_like(M_ul)
C_ul = np.zeros_like(M_ul)
C_br = np.zeros_like(M_ul)


# @ti.func
# def dot_block_diag(A, x):
#     A1 = ti.Matrix.zero(float, 3, 3)
#     A2 = ti.Matrix.zero(float, 3, 3)
#     x1 = ti.Vector.zero(float, 3)
#     x2 = ti.Vector.zero(float, 3)
#     ret =ti.Vector.zero(float, 6)
#     for i, j in ti.static(ti.ndrange(3,3)):
#         A1[i, j] = A[i, j]
#         A2[i, j] = A[i + 3, j + 3]
#     for i in ti.static(range(3)):
#         x1[i] = x[i]
#         x2[i] = x[i + 3]

#     r1 = A1 @ x1
#     r2 = A2 @ x2

#     for i in ti.static(range(3)):
#         ret[i] = r1[i]
#         ret[i + 3] = r2[i]

#     return ret


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

# @ti.func
# def block_diag(A, B):
#     ret = ti.Matrix.zero(float, 6, 6)
#     for i, j in ti.static(ti.ndrange(3, 3)):
#         ret[i, j] = A[i, j]
#         ret[i + 3, j + 3] = B[i, j]
#     return ret


@ti.func
def unskew(R):
    ret = ti.Vector.zero(float, 3)
    ret[0] = R[2, 1]
    ret[1] = - R[2, 0]
    ret[2] = R[1, 0]
    return ret


@ti.func
def f(p, q):
    return ti.Vector.zero(float, 3)


@ti.func
def tau(p, q):
    return ti.Vector.zero(float, 3)

# @ti.func
# def solve_block_diag(A, b):
#     # assert A.shape[0] == 6 and A.shape[1] == 6 and b.shape[0] == 6
#     # assert A is block diagonal matrix
#     A1 = ti.Matrix.zero(float, 3, 3)
#     A2 = ti.Matrix.zero(float, 3, 3)
#     for i, j in ti.static(ti.ndrange(3,3)):
#         A1[i, j] = A[i, j]
#         A2[i, j] = A[i + 3, j + 3]

#     b1 = ti.Vector.zero(float, 3)
#     b2 = ti.Vector.zero(float, 3)
#     for i in ti.static(range(3)):
#         b1[i] = b[i]
#         b2[i] = b[i + 3]
#     x1 = A1.inverse() @ b1
#     x2 = A2.inverse() @ b2
#     ret = ti.Vector.zero(float, 6)
#     for i in ti.static(range(3)):
#         ret[i] = x1[i]
#         ret[i + 3] = x2[i]

#     return ret


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
    # return block_diag(ti.Matrix.zero(float, 3, 3), ti.Matrix.cols([ja_dot, jb_dot, jc_dot]))

# @ti.func
# def J(a, b, c):
#     ret = ti.Matrix.zero(float, 6, 6)
#     jw = Jw(a, b, c)
#     return block_diag(ti.Matrix.identity(float, 3), jw)

@ti.func
def fill_3x3(J, A, i):
    '''
    J: 3 * 6n
    A: 3 * 3
    i: 
    '''
    I = 3 * i
    for i, j in ti.ndrange(3,3):
        J[i, I + j] = A[i, j]

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
    def __init__(self, id, scale=[1.0, 1.0, 1.0], omega=[0., 0., 0.], pos=[0., 0., 0.], parent=None, Newton_Euler=False):
        
        # generalized coordinates
        self.p = ti.Vector.field(3, float, shape=())
        self.v = ti.Vector.field(3, float, shape=())
        self.R = ti.Matrix.field(3, 3, float, shape=())

        self.R0 = ti.Matrix.field(3, 3, float, shape=())
        self.R0_dot = ti.Matrix.field(3,3, float, shape = ())
        # self.R0 = np.zeros((3, 3), dtype = np.float32)
        # self.R0_dot = np.zeros((3, 3), dtype = np.float32)
        self.a1 = np.zeros((3, 3), dtype = np.float32)
        self.a2 = np.zeros((3,3), dtype = np.float32)

        self.q = ti.Vector.field(6, float, shape=())
        self.q_dot = ti.Vector.field(6, float, shape=())
        # self.J_dot = ti.Matrix.field(6, 6, float, shape = ())
        self.omega = ti.Vector.field(3, float, shape=())
        # self.euler = ti.field(float, shape = (3))
        # self.euler_dot = ti.field(float, shape = (3))
        # self.Jw = ti.Matrix.field(3,3,float, shape=())
        # self.Mc = ti.field(float, shape=(6, 6))

        self.p[None] = ti.Vector(pos)
        self.omega[None] = ti.Vector(omega)
        # constants
        self.scale = scale
        self.m = 1.0
        # self.Ic = ti.Matrix.diag(3, self.m / 12 * scale[0] ** 2)
        self.Ic = self.m / 12 * scale[0] ** 2

        self.id = id

        self.v_transformed = ti.Vector.field(3, float, shape=(8))
        self.vertices = ti.Vector.field(3, float, shape=(8))
        self.indices = ti.field(ti.i32, shape=(3 * 12))
        self.faces = ti.Vector.field(4, ti.i32, shape=(6))

        self.initialize()
        # self.set_Ic()
        # self.set_M()
        self.gen_v()
        self.gen_id()

        self.parent = parent
        self.r_pkl_hat = self.vertices[0]
        # parent center to link
        self.r_lk_hat = -self.vertices[7]
        # link to center
        self.children = []
        if self.parent is not None:
            self.parent.children.append(self)
        
        self.substep = self.midpoint if Newton_Euler else self.lagrange_midpoint

    @ti.kernel
    def initialize(self):
        self.v[None] = ti.Vector.zero(float, 3)
        self.R[None] = ti.Matrix.identity(float, 3)
        # self.set_Mc()
        # self.q_dot[None][0] = 1.0
        for k in ti.static(range(3)):
            self.q[None][k] = self.p[None][k]
        self.q_dot[None][3] = self.omega[None][0]
        self.q_dot[None][4] = self.omega[None][1]
        self.q_dot[None][5] = self.omega[None][2]

    # @ti.func
    # def set_Mc(self):
    #     for i in ti.static(range(3)):
    #         self.Mc[i, i] = self.m
    #         self.Mc[i + 3, i + 3] = self.Ic

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
            r = r @ R0_pk
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
        dJw = ti.Matrix.zero(float, 3,3)
        R0 = ti.Matrix.zero(float, 3,3)
        if ti.static(self.parent is not None):
            R0_pk = load_3x3(self.parent.R0)

            dJw = R0_pk @ Jw(self.q[None][3], self.q[None][4], self.q[None][5])
            R0 = rotation(self.q[None][3], self.q[None][4], self.q[None][5]) @ R0_pk
        else :
            # root node
            dJw = Jw(self.q[None][3], self.q[None][4], self.q[None][5])
            R0 = rotation(self.q[None][3], self.q[None][4], self.q[None][5])
            for i in ti.static(range(3)):
                Jw_k[i, i] = 1.0

        self.R0[None] = R0
        for i, j in ti.static(ti.ndrange(3,3)):
            Jw_k[i, j + 3 * (self.id + 1)] = dJw[i, j]
        # fill_3x3(Jw_k, dJw, self.id + 1)
        # fill_3x3(self.R0, R0, 0)

    @ti.kernel
    def coeff_Jw_pk_Jw_k(self, a1: ti.types.ndarray(), a2: ti.types.ndarray()):
        R0_pk = ti.Matrix.identity(float, 3) if ti.static(self.parent is None) else load_3x3(self.parent.R0)
        R0_k = load_3x3(self.R0)

        _a1 = skew(R0_pk @ self.r_pkl_hat)
        _a2 = skew(R0_k @ self.r_lk_hat)
        for i, j in ti.static(ti.ndrange(3,3)):
            a1[i, j] = _a1[i,j]
            a2[i, j] = _a2[i,j]

        # fill_3x3(a1, _a1, 0)
        # fill_3x3(a2, _a2, 0)

    def fill_Jvk(self, Jvk):
        if self.parent is not None:
            self.coeff_Jw_pk_Jw_k(self.a1, self.a2)
            Jvk = Jvk - self.a1 @ Jw_pk - self.a2 @ Jw_k
        else:
            pass
    
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
        R0_dot_pk = ti.Matrix.zero(float,3, 3)
        if ti.static(self.parent is not None):
            R0_pk = load_3x3(self.parent.R0) 
            R0_dot_pk = load_3x3(self.parent.R0_dot) 
            
        R0_k_dot = R_dot @ R0_pk + R @ R0_dot_pk
        # fill_3x3(self.R0_dot, R0_k_dot, 0)
        
        _a1_dot = skew(R0_dot_pk @ self.r_pkl_hat)
        _a2_dot = skew(R0_k_dot @ self.r_lk_hat)
        # fill_3x3(a1_dot, _a1_dot, 0)
        # fill_3x3(a2_dot, _a2_dot, 0)

        _tiled_omega_BR = skew(_Jw_hat @ ti.Vector([q_dot[3], q_dot[4], q_dot[5]]))
        # fill_3x3(tiled_omega_BR, _tiled_omega_BR, 0)

        self.R0_dot[None] = R0_k_dot
        for i, j in ti.static(ti.ndrange(3,3)):
            Jw_hat[i, j] = _Jw_hat[i,j]
            Jw_dot_hat[i, j] = _Jw_dot_hat[i,j]
            a1_dot[i, j] = _a1_dot[i,j]
            a2_dot[i, j] = _a2_dot[i,j]
            tiled_omega_BR[i, j] = _tiled_omega_BR[i,j]
        
    def aggregate_JkT_Mck_Jk(self):
        global M_ul, M_br
        ul = Jv_k.T @ Jv_k * self.m
        br = Jw_k.T @ Jw_k * self.Ic
        # M q.. + C = Q
        M_ul += ul
        M_br += br


    def aggregate_JkT_Mck_Jk_dot(self, ):
        '''
        fill Jw_dot, Jv_dot
        
        '''
        global M_ul, M_br, C_ul, C_br, Jw_pk_dot, Jw_k_dot, Jv_k_dot, Jw_k, Jw_pk, Jv_k
        a1_dot = np.zeros_like(self.a1)
        a2_dot = np.zeros_like(self.a2)
        Jw_dot_hat = np.zeros((3, 3), np.float32)
        Jw_hat = np.zeros((3, 3), np.float32)
        tiled_omega_BR = np.zeros((3, 3), np.float32)

        self.fill_J_dot_related(a1_dot, a2_dot, Jw_hat, Jw_dot_hat, tiled_omega_BR)
        R0_pk_dot = np.zeros((3,3), np.float32) if self.parent is None else self.parent.R0_dot.to_numpy()
        R0_pk = np.identity(3, dtype = np.float32) if self.parent is None else self.parent.R0.to_numpy()
        Jw_k_dot = Jw_pk_dot
        Jw_k_dot[:, (self.id + 1) * 3: (self.id + 2) * 3] += R0_pk_dot @ Jw_hat + R0_pk @ Jw_dot_hat
        # FIXME: offset, fixed
        # FIXME: change R0 and R0 dot to numpy arrays, fixed: not possible
        Jv_k_dot = Jv_k_dot - self.a1 @ Jw_pk_dot - self.a2 @ Jw_k_dot - a1_dot @ Jw_pk - a2_dot @ Jw_k 

        ul = Jv_k.T @ Jv_k_dot * self.m
        br = Jw_k.T @ (Jw_k_dot + tiled_omega_BR @ Jw_k) * self.Ic
        C_ul += ul
        C_br += br

    def top_down(self):
        global M_ul, M_br, C_ul, C_br, Jw_pk_dot, Jw_k_dot, Jv_k_dot, Jw_k, Jw_pk, Jv_k

        Jw_k = Jw_pk
        self.fill_Jwk(Jw_k)
        self.fill_Jvk(Jv_k)
        self.aggregate_JkT_Mck_Jk()
        self.aggregate_JkT_Mck_Jk_dot()

        Jw_pk_dot = Jw_k
        Jw_pk = Jw_k

        self.substep()
        # FIXME: support for tree (now only suitable for chain)
        for c in self.children:
            c.top_down()


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

    cube = Cube(0, omega=[10.0, 10.0, 1.0])
    link = Cube(1, omega=[0., 0., 0.], pos = [-1., -1., -1.], parent= cube)
    root = cube

    mouse_staled = np.zeros(2, dtype=np.float32)
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

        scene.mesh(cube.v_transformed, cube.indices,
                   two_sided=True, show_wireframe=False)
        scene.mesh(link.v_transformed, link.indices, two_sided=True, show_wireframe=False)

        if ts % per_trace == 0:
            t = booknote(cube.v_transformed.to_numpy())
        scene.particles(trajectory, radius=0.01, color=(1.0, 0.0, 0.0))
        scene.particles(cube.v_transformed, radius=0.05, color=(1.0, 0.0, 0.0))
        scene.particles(link.v_transformed, radius=0.05, color=(1.0, 0.0, 0.0))
        canvas.scene(scene)
        window.show()
        ts += 1


main()