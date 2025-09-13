from .hittable import Hittable, HitRecord
from .interval import Interval

class HittableList(Hittable):
    def __init__(self, objects=None):
        self.objects = objects or []
    
    def clear(self):
        self.objects.clear()
    
    def add(self, object):
        self.objects.append(object)
    
    def hit(self, r, ray_t, rec):
        hit_anything = False
        closest_so_far = ray_t.max
        
        for object in self.objects:
            temp_rec = HitRecord()
            if object.hit(r, Interval(ray_t.min, closest_so_far), temp_rec):
                hit_anything = True
                closest_so_far = temp_rec.t
                rec.p = temp_rec.p
                rec.normal = temp_rec.normal
                rec.t = temp_rec.t
                rec.front_face = temp_rec.front_face
                rec.material = temp_rec.material
        
        return hit_anything