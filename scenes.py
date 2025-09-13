import random
import math
from .vec3 import Vec3, Point3, Color
from .hittable_list import HittableList
from .sphere import Sphere
from .box import Box
from .material import Lambertian, Metal, Dielectric, Checkerboard
from .utils import PI

def random_scene():
    """
    Create a random scene with many spheres (similar to the final scene in the book)
    """
    world = HittableList()
    
    # Ground
    ground_material = Checkerboard(Color(0.2, 0.3, 0.1), Color(0.9, 0.9, 0.9), 0.32)
    world.add(Sphere(Point3(0, -1000, 0), 1000, ground_material))
    
    # Random small spheres
    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random.random()
            center = Point3(a + 0.9 * random.random(), 0.2, b + 0.9 * random.random())
            
            if (center - Point3(4, 0.2, 0)).length() > 0.9:
                if choose_mat < 0.8:
                    # Diffuse
                    albedo = Color.random() * Color.random()
                    sphere_material = Lambertian(albedo)
                    world.add(Sphere(center, 0.2, sphere_material))
                elif choose_mat < 0.95:
                    # Metal
                    albedo = Color.random(0.5, 1)
                    fuzz = random.uniform(0, 0.5)
                    sphere_material = Metal(albedo, fuzz)
                    world.add(Sphere(center, 0.2, sphere_material))
                else:
                    # Glass
                    sphere_material = Dielectric(1.5)
                    world.add(Sphere(center, 0.2, sphere_material))
    
    # Three larger spheres
    material1 = Dielectric(1.5)
    world.add(Sphere(Point3(0, 1, 0), 1.0, material1))
    
    material2 = Lambertian(Color(0.4, 0.2, 0.1))
    world.add(Sphere(Point3(-4, 1, 0), 1.0, material2))
    
    material3 = Metal(Color(0.7, 0.6, 0.5), 0.0)
    world.add(Sphere(Point3(4, 1, 0), 1.0, material3))
    
    # Add a box (our twist)
    box_material = Lambertian(Color(0.2, 0.8, 0.4))
    world.add(Box(Point3(-1.5, 0.2, -1.5), Point3(-0.5, 1.2, -0.5), box_material))
    
    return world

def simple_scene():
    """
    Create a simple scene with a few spheres
    """
    world = HittableList()
    
    # Ground
    ground_material = Lambertian(Color(0.8, 0.8, 0.0))
    world.add(Sphere(Point3(0, -100.5, -1), 100, ground_material))
    
    # Center sphere
    center_material = Lambertian(Color(0.1, 0.2, 0.5))
    world.add(Sphere(Point3(0, 0, -1), 0.5, center_material))
    
    # Left sphere (glass)
    left_material = Dielectric(1.5)
    world.add(Sphere(Point3(-1.0, 0, -1), 0.5, left_material))
    # Inner sphere for hollow glass effect
    world.add(Sphere(Point3(-1.0, 0, -1), 0.4, Dielectric(1.0/1.5)))
    
    # Right sphere (metal)
    right_material = Metal(Color(0.8, 0.6, 0.2), 0.0)
    world.add(Sphere(Point3(1.0, 0, -1), 0.5, right_material))
    
    return world

def animated_scene(frame, total_frames):
    """
    Create an animated scene where objects move over time
    """
    world = HittableList()
    
    # Ground
    ground_material = Checkerboard(Color(0.2, 0.3, 0.1), Color(0.9, 0.9, 0.9), 0.32)
    world.add(Sphere(Point3(0, -1000, 0), 1000, ground_material))
    
    # Animation parameter (0 to 1)
    t = frame / (total_frames - 1)
    angle = 2 * PI * t
    
    # Moving sphere
    center = Point3(4 * math.cos(angle), 1 + 0.5 * math.sin(2 * angle), 4 * math.sin(angle))
    material1 = Dielectric(1.5)
    world.add(Sphere(center, 1.0, material1))
    
    # Static spheres
    material2 = Lambertian(Color(0.4, 0.2, 0.1))
    world.add(Sphere(Point3(-4, 1, 0), 1.0, material2))
    
    material3 = Metal(Color(0.7, 0.6, 0.5), 0.0)
    world.add(Sphere(Point3(0, 1, 0), 1.0, material3))
    
    # Moving box
    box_size = 0.5 + 0.3 * math.sin(angle)
    box_material = Lambertian(Color(0.2, 0.8, 0.4))
    min_point = Point3(-1.5, 0.2, -1.5 + 2 * math.sin(angle))
    max_point = Point3(-1.5 + box_size, 0.2 + box_size, -1.5 + 2 * math.sin(angle) + box_size)
    world.add(Box(min_point, max_point, box_material))
    
    return world