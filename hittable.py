from .vec3 import Vec3, Point3
from .ray import Ray

class HitRecord:
    def __init__(self):
        self.p = Point3()
        self.normal = Vec3()
        self.t = 0.0
        self.front_face = False
        self.material = None
    
    def set_face_normal(self, r, outward_normal):
        self.front_face = r.direction.dot(outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal

class Hittable:
    def hit(self, r, ray_t, rec):
        pass