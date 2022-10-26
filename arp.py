from re import L
import taichi as ti
import numpy as np
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
    ret[1,0] = +r[0]
    return ret

@ti.func
def f(p, q):
    return ti.Vector.zero(float , 3)
    

@ti.func
def tau(p, q):
    return ti.Vector.zero(float, 3)

@ti.data_oriented
class Cube:
    def __init__(self, scale = [1.0, 1.0, 1.0], omega = [0.,0.,0.], pos = [0.,0.,0.], parent = None):
        # generalized coordinates
        self.p = ti.Vector.field(3, float, shape = ())
        self.v = ti.Vector.field(3, float, shape = ())
        self.q = ti.Matrix.field(3,3, float, shape=())
        self.omega = ti.Vector.field(3, float, shape = ())

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
        self.substep = self.midpoint
    @ti.kernel
    def initialize(self):
        self.v[None] = ti.Vector.zero(float , 3)
        self.q[None] = ti.Matrix.identity(float, 3)

    @ti.kernel
    def midpoint(self):
        dt = 1e-3
        dv = f(self.p[None], self.q[None]) * dt / 2 / self.m
        dp = self.v[None] * dt 
        domega = tau(self.p[None], self.q[None]) * dt /2 / self.Ic
        dq = skew(self.omega[None]) @ self.q[None] * dt / 2

        v_mid = dv + self.v[None]
        p_mid = dp + self.p[None]
        omega_mid = domega + self.omega[None]
        q_mid = dq + self.q[None]

        self.v[None] += f(p_mid, q_mid) * dt / self.m
        self.p[None] += v_mid * dt
        self.omega[None] += tau(p_mid, q_mid) * dt / self.Ic
        self.q[None] += skew(omega_mid) @ q_mid * dt

        for i in range(8):
            self.v_transformed[i] = self.q[None] @ self.vertices[i] + self.p[None]

    def set_Ic(self):
        pass

    def set_M(self):
        pass
        
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
    
    cube = Cube(omega = [0.0, 100.0, 0.0])
    cube2 = Cube(pos = [0.5, 0.0, 0.0])
    
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
        cube2.substep()

        # scene.mesh(cube.v, cube.indices, two_sided = True, show_wireframe = False)            
        scene.mesh(cube.v_transformed, cube.indices, two_sided = True, show_wireframe = True)            
        scene.mesh(cube2.v_transformed, cube2.indices, two_sided = True, show_wireframe = True)            
        canvas.scene(scene)
        window.show()

main()