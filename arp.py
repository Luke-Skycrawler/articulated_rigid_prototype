from re import L
import taichi as ti
from taichi import cos, sin
import numpy as np

from fractal3d_ggui import dot
# FIXME: drifting of the R matrix

ti.init(arch = ti.cuda, default_fp = ti.f32)

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0]
delta = 0.08

@ti.func
def skew(r):
    ret = ti.Matrix.zero(float, 3, 3)
    ret[0,1] = -r[2]
    ret[1,0] = +r[2]
    ret[0,2] = +r[1]
    ret[2,0] = -r[1]
    ret[1,2] = -r[0]
    ret[2,1] = +r[0]
    return ret

@ti.func
def block_diag(A, B):
    ret = ti.Matrix.zero(float, 6, 6)
    for i, j in ti.static(ti.ndrange(3, 3)):
        ret[i, j] = A[i, j]
        ret[i + 3, j + 3] = B[i, j]
    return ret

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

@ti.func
def solve_block_diag(A, b):
    # assert A.shape[0] == 6 and A.shape[1] == 6 and b.shape[0] == 6
    # assert A is block diagonal matrix
    A1 = ti.Matrix.zero(float, 3, 3)
    A2 = ti.Matrix.zero(float, 3, 3)
    for i, j in ti.static(ti.ndrange(3,3)):
        A1[i, j] = A[i, j]
        A2[i, j] = A[i + 3, j + 3]

    b1 = ti.Vector.zero(float, 3) 
    b2 = ti.Vector.zero(float, 3) 
    for i in ti.static(range(3)):
        b1[i] = b[i]
        b2[i] = b[i + 3]
    x1 = A1.inverse() @ b1
    x2 = A2.inverse() @ b2
    ret = ti.Vector.zero(float, 6)
    for i in ti.static(range(3)):
        ret[i] = x1[i]
        ret[i + 3] = x2[i]

    return ret

@ti.func
def rotation(a, b, c):
    s1 = sin(a)
    s2 = sin(b)
    s3 = sin(c)

    c1 = cos(a)
    c2 = cos(b)
    c3 = cos(c)

    R = ti.Matrix([
        [c2, -c3 * s2, s2 * s3],
        [c1 * s2, c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3],
        [s1 * s2, c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3]
    ])
    return R

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
        [c2, -c3 * s2, s2 * s3],
        [c1 * s2, c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3],
        [s1 * s2, c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3]
    ])
    pR_pa = ti.Matrix([
        [0, 0, 0],
        [-s1 * s2, -s1 * c2 * c3 - c1 * s3, -c3 * c1 - -s1 * c2 * s3],
        [c1 * s2, -s1 * s3 + c2 * c3 * c1, -s1 * c3 - c2 * c1 * s3]
    ])
    pR_pb = ti.Matrix([
        [-s2, -c3 * c2, c2 * s3],
        [c1 * c2, c1 * -s2 * c3 - s1 * s3, -c3 * s1 - c1 * -s2 * s3],
        [s1 * c2, c1 * s3 + -s2 * c3 * s1, c1 * c3 - -s2 * s1 * s3]
    ])
    pR_pc = ti.Matrix([
        [0, s3 * s2, s2 * c3],
        [0, c1 * c2 * -s3 - s1 * c3, s3 * s1 - c1 * c2 * c3],
        [0, c1 * c3 + c2 * -s3 * s1, c1 * -s3 - c2 * s1 * c3]
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
        [c2, -c3 * s2, s2 * s3],
        [c1 * s2, c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3],
        [s1 * s2, c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3]
    ])
    pR_pa = ti.Matrix([
        [0, 0, 0],
        [-s1 * s2, -s1 * c2 * c3 - c1 * s3, -c3 * c1 - -s1 * c2 * s3],
        [c1 * s2, -s1 * s3 + c2 * c3 * c1, -s1 * c3 - c2 * c1 * s3]
    ])
    pR_pb = ti.Matrix([
        [-s2, -c3 * c2, c2 * s3],
        [c1 * c2, c1 * -s2 * c3 - s1 * s3, -c3 * s1 - c1 * -s2 * s3],
        [s1 * c2, c1 * s3 + -s2 * c3 * s1, c1 * c3 - -s2 * s1 * s3]
    ])
    pR_pc = ti.Matrix([
        [0, s3 * s2, s2 * c3],
        [0, c1 * c2 * -s3 - s1 * c3, s3 * s1 - c1 * c2 * c3],
        [0, c1 * c3 + c2 * -s3 * s1, c1 * -s3 - c2 * s1 * c3]
    ])
    R_dot = pR_pa * d1 + pR_pb * d2 + pR_pc * d3

    pR2_paa = ti.Matrix([
        [0, 0, 0],
        [-c1 * s2, -c1 * c2 * c3 - -s1 * s3, -c3 * -s1 - -c1 * c2 * s3],
        [-s1 * s2, -c1 * s3 + c2 * c3 * -s1, -c1 * c3 - c2 * -s1 * s3]
    ])

    pR2_pab = ti.Matrix([
        [0, 0, 0],
        [-s1 * c2, -s1 * -s2 * c3 - c1 * s3, -c3 * c1 - -s1 * -s2 * s3],
        [c1 * c2, -s1 * s3 + -s2 * c3 * c1, -s1 * c3 - -s2 * c1 * s3]
    ])
    pR2_pbb = ti.Matrix([
        [-c2, -c3 * -s2, -s2 * s3],
        [c1 * -s2, c1 * -c2 * c3 - s1 * s3, -c3 * s1 - c1 * -c2 * s3],
        [s1 * -s2, c1 * s3 + -c2 * c3 * s1, c1 * c3 - -c2 * s1 * s3]
    ])

    pR2_pac = ti.Matrix([
        [0, 0, 0],
        [0, -s1 * c2 * -s3 - c1 * c3, s3 * c1 - -s1 * c2 * c3],
        [0, -s1 * c3 + c2 * -s3 * c1, -s1 * -s3 - c2 * c1 * c3]
    ])
    pR2_pbc = ti.Matrix([
        [0, s3 * c2, c2 * c3],
        [0, c1 * -s2 * -s3 - s1 * c3, s3 * s1 - c1 * -s2 * c3],
        [0, c1 * c3 + -s2 * -s3 * s1, c1 * -s3 - -s2 * s1 * c3]
    ])
    pR2_pcc = ti.Matrix([
        [0, c3 * s2, s2 * -s3],
        [0, c1 * c2 * -c3 - s1 * -s3, c3 * s1 - c1 * c2 * -s3],
        [0, c1 * -s3 + c2 * -c3 * s1, c1 * -c3 - c2 * s1 * -s3]
    ])

    dpR_pa = pR2_paa * d1 + pR2_pab * d2 + pR2_pac * d3
    dpR_pb = pR2_pab * d1 + pR2_pbb * d2 + pR2_pbc * d3
    dpR_pc = pR2_pac * d1 + pR2_pbc * d2 + pR2_pcc * d3

    ja_dot = unskew(dpR_pa @ R.transpose() + pR_pa @ R_dot.transpose())
    jb_dot = unskew(dpR_pb @ R.transpose() + pR_pb @ R_dot.transpose())
    jc_dot = unskew(dpR_pc @ R.transpose() + pR_pc @ R_dot.transpose())
    return block_diag(ti.Matrix.zero(float, 3, 3), ti.Matrix.cols([ja_dot, jb_dot, jc_dot]))

@ti.func
def J(a, b, c):
    ret = ti.Matrix.zero(float, 6, 6)
    jw = Jw(a, b, c)
    return block_diag(ti.Matrix.identity(float, 3), jw)

@ti.data_oriented
class Cube:
    def __init__(self, scale = [1.0, 1.0, 1.0], omega = [0.,0.,0.], pos = [0.,0.,0.], parent = None, Newton_Euler = False):
        # generalized coordinates
        self.p = ti.Vector.field(3, float, shape = ())
        self.v = ti.Vector.field(3, float, shape = ())
        self.R = ti.Matrix.field(3,3, float, shape=())
        self.q = ti.Vector.field(6, float, shape = ())
        self.q_dot = ti.Vector.field(6, float, shape = ())
        # self.J_dot = ti.Matrix.field(6, 6, float, shape = ())
        self.omega = ti.Vector.field(3, float, shape = ())
        self.euler = ti.field(float, shape = (3))
        self.euler_dot = ti.field(float, shape = (3))
        # self.Jw = ti.Matrix.field(3,3,float, shape=())
        self.Mc = ti.Matrix.field(6, 6, float, shape=())
        
        self.p[None] = ti.Vector(pos)
        self.omega[None] = ti.Vector(omega)
        # constants
        self.scale = scale
        self.m = 1.0
        # self.Ic = ti.Matrix.diag(3, self.m / 12 * scale[0] ** 2)
        self.Ic = self.m / 12 * scale[0] ** 2
        self.parent = parent
        self.children = []

        self.v_transformed = ti.Vector.field(3, float, shape = (8))
        self.vertices = ti.Vector.field(3, float, shape = (8))
        self.indices = ti.field(ti.i32, shape = (3 * 12))
        self.faces = ti.Vector.field(4, ti.i32, shape = (6))

        
        self.initialize()
        # self.set_Ic()
        # self.set_M()
        self.gen_v()
        self.gen_id()
        self.substep = self.midpoint if Newton_Euler else self.lagrange_explicit_euler
        
    @ti.kernel
    def initialize(self):
        self.v[None] = ti.Vector.zero(float , 3)
        self.R[None] = ti.Matrix.identity(float, 3)
        self.set_Mc()
        # self.q_dot[None][3] = 100.0
        self.q_dot[None][4] = 100.0
        self.q_dot[None][5] = 0.0

    @ti.func
    def set_Mc(self):
        self.Mc[None] = ti.Matrix.identity(float, 6) * self.m        
        for i in ti.static(range(3, 6)):
            self.Mc[None][i, i] = self.Ic

            
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
        domega = tau(self.p[None], self.R[None]) * dt /2 / self.Ic
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

    @ti.kernel
    def lagrange_explicit_euler(self):
        dt = 1e-3
        J = J(self.q[None][3], self.q[None][4], self.q[None][5])
        
        M = J.transpose() @ self.Mc[None] @ J
        tiled_omega_BR = skew(Jw(self.q[None][3], self.q[None][4], self.q[None][5]) @ ti.Vector([self.q_dot[None][3], self.q_dot[None][4], self.q_dot[None][5]]))
        tiled_omega = block_diag(ti.Matrix.zero(float, 3,3), tiled_omega_BR)
        C = (J.transpose() @ self.Mc[None] @ J_dot(self.q[None][3], self.q[None][4], self.q[None][5], self.q_dot[None][3], self.q_dot[None][4], self.q_dot[None][5]) + J.transpose() @ tiled_omega @ self.Mc[None] @ J ) @ self.q_dot[None]

        # sovle M q.. + C = Q
        self.q[None] += self.q_dot[None] * dt
        self.q_dot[None] += solve_block_diag(M, -C) * dt
        
        r = rotation(self.q[None][3], self.q[None][4], self.q[None][5])
        p = ti.Vector([self.q[None][0], self.q[None][1], self.q[None][2]])
        for i in ti.static(range(8)):
            self.v_transformed[i] = r @ self.vertices[i] + p
        

    # @ti.kernel
    # def lagrange_midpoint(self):
    #     '''
    #     Lagrange Formulation
    #     6 DoFs all stacked together as q
    #     3 COM velocities + 3 euler angles (rotation X1 Z2 X3)
    #     '''
    #     J = J(self.q[3], self.q[4], self.q[5])
    #     M = J.T @ self.Mc[None] @ J
    #     tiled_omega_BR = skew(Jw(self.q[3], self.q[4], self.q[5]) @ self.q_dot[None])
    #     tiled_omega = block_diag(ti.Matrix.zero(float, 3,3), tiled_omega_BR)
    #     C = (J.T @ self.Mc[None] @ self.J_dot[None] + J.T @ tiled_omega @ self.Mc[None] @ J ) @ self.q_dot
    #     # Q = ti.Vector.zero(float, 6)
        
        
        
    def link(self, cube):
        # self.parent.append(cube)
        cube.children.append(self)
        
    @ti.kernel
    def gen_v(self ):
        for i,j,k in ti.ndrange(2,2,2):
            I = i * 4 + j * 2 + k
            self.vertices[I] = ti.Vector([i - 0.5, j - 0.5, k - 0.5])
    
        
    @ti.kernel
    def gen_id(self ):
        self.faces[0] = ti.Vector([0,1,3,2])
        self.faces[1] = ti.Vector([4,5,1,0])
        self.faces[2] = ti.Vector([2,3,7,6])
        self.faces[3] = ti.Vector([4,0,2,6])
        self.faces[4] = ti.Vector([1,5,7,3])
        self.faces[5] = ti.Vector([5,4,6,7])
        for i in range(6):
            self.indices[i * 6 + 0] = self.faces[i][0]
            self.indices[i * 6 + 1] = self.faces[i][1]
            self.indices[i * 6 + 2] = self.faces[i][2]
            self.indices[i * 6 + 3] = self.faces[i][2]
            self.indices[i * 6 + 4] = self.faces[i][3]
            self.indices[i * 6 + 5] = self.faces[i][0]
        

    
def main():
    window = ti.ui.Window("Articulated Multibody Simualtion", (800, 800), vsync = True)
    canvas = window.get_canvas()
    canvas.set_background_color((0, 0, 0))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera_pos = np.array([0.0,0.0,3.0])
    camera_dir = np.array([0.0, 0.0, -1.0])
    
    cube = Cube(omega = [100.0, 100.0, 0.0])
    
    mouse_staled = np.zeros(2, dtype = np.float32)
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
        
        scene.point_light(pos = (0,1,2), color = (1,1,1))
        scene.ambient_light((0.5, 0.5, 0.5))

        cube.substep()

        scene.mesh(cube.v_transformed, cube.indices, two_sided = True, show_wireframe = False)            
        canvas.scene(scene)
        window.show()

main()