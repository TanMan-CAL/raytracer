import random
import math
from .vec3 import Vec3, Color
from .ray import Ray
from .utils import reflect, refract

class Material:
    def scatter(self, r_in, rec, attenuation, scattered):
        return False

class Lambertian(Material):
    def __init__(self, albedo):
        self.albedo = albedo
    
    def scatter(self, r_in, rec, attenuation, scattered):
        scatter_direction = rec.normal + Vec3.random_unit_vector()
        
        # Catch degenerate scatter direction
        if scatter_direction.near_zero():
            scatter_direction = rec.normal
        
        scattered[0] = Ray(rec.p, scatter_direction)
        attenuation[0] = self.albedo
        return True

class Metal(Material):
    def __init__(self, albedo, fuzz=0):
        self.albedo = albedo
        self.fuzz = min(fuzz, 1.0)
    
    def scatter(self, r_in, rec, attenuation, scattered):
        reflected = reflect(r_in.direction.unit_vector(), rec.normal)
        reflected = reflected + self.fuzz * Vec3.random_in_unit_sphere()
        scattered[0] = Ray(rec.p, reflected)
        attenuation[0] = self.albedo
        return scattered[0].direction.dot(rec.normal) > 0

class Dielectric(Material):
    def __init__(self, refraction_index):
        self.refraction_index = refraction_index
    
    def scatter(self, r_in, rec, attenuation, scattered):
        attenuation[0] = Color(1.0, 1.0, 1.0)
        ri = 1.0 / self.refraction_index if rec.front_face else self.refraction_index
        
        unit_direction = r_in.direction.unit_vector()
        cos_theta = min(-unit_direction.dot(rec.normal), 1.0)
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
        
        cannot_refract = ri * sin_theta > 1.0
        
        if cannot_refract or self.reflectance(cos_theta, ri) > random.random():
            direction = reflect(unit_direction, rec.normal)
        else:
            direction = refract(unit_direction, rec.normal, ri)
        
        scattered[0] = Ray(rec.p, direction)
        return True
    
    def reflectance(self, cosine, ref_idx):
        # Use Schlick's approximation for reflectance
        r0 = (1 - ref_idx) / (1 + ref_idx)
        r0 = r0 * r0
        return r0 + (1 - r0) * pow((1 - cosine), 5)

# Our twist: Adding a Checkerboard material
class Checkerboard(Material):
    def __init__(self, odd_color, even_color, scale=1.0):
        self.odd_material = Lambertian(odd_color)
        self.even_material = Lambertian(even_color)
        self.scale = scale
    
    def scatter(self, r_in, rec, attenuation, scattered):
        # Determine if we're on an odd or even square
        sines = math.sin(self.scale * rec.p.x) * math.sin(self.scale * rec.p.y) * math.sin(self.scale * rec.p.z)
        if sines < 0:
            return self.odd_material.scatter(r_in, rec, attenuation, scattered)
        else:
            return self.even_material.scatter(r_in, rec, attenuation, scattered)