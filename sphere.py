import math
from .vec3 import Point3
from .hittable import Hittable, HitRecord
from .interval import Interval

class Sphere(Hittable):
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material
    
    def hit(self, r, ray_t, rec):
        oc = r.origin - self.center
        a = r.direction.length_squared()
        h = oc.dot(r.direction)
        c = oc.length_squared() - self.radius * self.radius
        
        discriminant = h * h - a * c
        if discriminant < 0:
            return False
        
        sqrtd = math.sqrt(discriminant)
        
        # Find the nearest root that lies in the acceptable range
        root = (-h - sqrtd) / a
        if not ray_t.surrounds(root):
            root = (-h + sqrtd) / a
            if not ray_t.surrounds(root):
                return False
        
        rec.t = root
        rec.p = r.at(rec.t)
        outward_normal = (rec.p - self.center) / self.radius
        rec.set_face_normal(r, outward_normal)
        rec.material = self.material
        
        return True