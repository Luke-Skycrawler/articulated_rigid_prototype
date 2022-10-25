import taichi as ti
import numpy as np

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

@ti.data_oriented
class Cube:
    def __init__(self, scale = [1.0, 1.0, 1.0], r = [0.,0.,0.], pos = [0.,0.,0.], parent = None):
        self.scale = scale
        self.r = ti.Vector(r)
        self.pos = ti.Vector.field(3, float, shape = ())
        self.rotation = ti.Matrix.field(3,3, float, shape=())
        self.parent = parent
        self.children = []
        self.vertices = ti.Vector.field(3, float, shape = (8))
        self.v = ti.Vector.field(3, float, shape = (8))
        self.indices = ti.field(ti.i32, shape = (3 * 12))
        self.faces = ti.Vector.field(4, ti.i32, shape = (6))

        self.set_Ic()
        self.set_M()
        self.gen_v()
        self.gen_id()
        self.initialize()
        
    @ti.kernel
    def substep(self):
        for i in range(8):
            self.v[i] = self.rotation[None] @ self.vertices[i] + self.pos[None]

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
    def initialize(self):
        self.rotation[None] = ti.Matrix.identity(float, 3)
        self.pos[None] = ti.Vector.zero(float, 3)
        

        
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
    
    cube = Cube()
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

        scene.mesh(cube.v, cube.indices, two_sided = True)            
        canvas.scene(scene)
        window.show()

main()