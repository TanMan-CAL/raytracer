import math
import time
import random
import multiprocessing as mp
import numpy as np
from PIL import Image
from tqdm import tqdm

from .vec3 import Vec3, Point3, Color
from .ray import Ray
from .interval import Interval
from .utils import degrees_to_radians, linear_to_gamma, INFINITY
from .hittable import HitRecord

class Camera:
    def __init__(self):
        # Image
        self.aspect_ratio = 16.0 / 9.0
        self.image_width = 400
        self.samples_per_pixel = 100
        self.max_depth = 50
        
        # Camera
        self.vfov = 90  # Vertical field of view (degrees)
        self.lookfrom = Point3(0, 0, 0)
        self.lookat = Point3(0, 0, -1)
        self.vup = Vec3(0, 1, 0)
        
        # Defocus blur
        self.defocus_angle = 0  # Variation angle of rays through each pixel
        self.focus_dist = 10  # Distance from camera lookfrom point to plane of perfect focus
        
        # Internal variables
        self.image_height = 0
        self.pixel_samples_scale = 0
        self.center = Point3()
        self.pixel00_loc = Point3()
        self.pixel_delta_u = Vec3()
        self.pixel_delta_v = Vec3()
        self.u = Vec3()
        self.v = Vec3()
        self.w = Vec3()
        self.defocus_disk_u = Vec3()
        self.defocus_disk_v = Vec3()
        
        # Initialize
        self.initialize()
    
    def initialize(self):
        # Calculate image height
        self.image_height = int(self.image_width / self.aspect_ratio)
        self.image_height = max(1, self.image_height)
        
        self.pixel_samples_scale = 1.0 / self.samples_per_pixel
        
        self.center = self.lookfrom
        
        # Determine viewport dimensions
        focal_length = (self.lookfrom - self.lookat).length()
        theta = degrees_to_radians(self.vfov)
        h = math.tan(theta / 2)
        viewport_height = 2 * h * self.focus_dist
        viewport_width = viewport_height * (self.image_width / self.image_height)
        
        # Calculate the u,v,w unit basis vectors for the camera coordinate frame
        self.w = (self.lookfrom - self.lookat).unit_vector()
        self.u = self.vup.cross(self.w).unit_vector()
        self.v = self.w.cross(self.u)
        
        # Calculate the vectors across the horizontal and down the vertical viewport edges
        viewport_u = viewport_width * self.u
        viewport_v = viewport_height * -self.v
        
        # Calculate the horizontal and vertical delta vectors from pixel to pixel
        self.pixel_delta_u = viewport_u / self.image_width
        self.pixel_delta_v = viewport_v / self.image_height
        
        # Calculate the location of the upper left pixel
        viewport_upper_left = self.center - (self.focus_dist * self.w) - viewport_u/2 - viewport_v/2
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)
        
        # Calculate the camera defocus disk basis vectors
        defocus_radius = self.focus_dist * math.tan(degrees_to_radians(self.defocus_angle / 2))
        self.defocus_disk_u = self.u * defocus_radius
        self.defocus_disk_v = self.v * defocus_radius
    
    def get_ray(self, i, j):
        # Get a randomly sampled camera ray for the pixel at location i,j
        
        # Get a random point in the square surrounding the pixel
        offset = self.sample_square()
        pixel_sample = self.pixel00_loc + ((i + offset.x) * self.pixel_delta_u) + ((j + offset.y) * self.pixel_delta_v)
        
        ray_origin = self.center if self.defocus_angle <= 0 else self.defocus_disk_sample()
        ray_direction = pixel_sample - ray_origin
        
        return Ray(ray_origin, ray_direction)
    
    def sample_square(self):
        return Vec3(random.random() - 0.5, random.random() - 0.5, 0)
    
    def defocus_disk_sample(self):
        p = Vec3.random_in_unit_disk()
        return self.center + (p.x * self.defocus_disk_u) + (p.y * self.defocus_disk_v)
    
    def ray_color(self, r, depth, world):
        # If we've exceeded the ray bounce limit, no more light is gathered
        if depth <= 0:
            return Color(0, 0, 0)
        
        rec = HitRecord()
        
        if world.hit(r, Interval(0.001, INFINITY), rec):
            scattered = [None]  # Using a list as a mutable container
            attenuation = [None]
            if rec.material.scatter(r, rec, attenuation, scattered):
                return attenuation[0] * self.ray_color(scattered[0], depth - 1, world)
            return Color(0, 0, 0)
        
        # Background - a simple gradient
        unit_direction = r.direction.unit_vector()
        a = 0.5 * (unit_direction.y + 1.0)
        return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0)
    
    def render_pixel(self, i, j, world):
        pixel_color = Color(0, 0, 0)
        for s in range(self.samples_per_pixel):
            r = self.get_ray(i, j)
            pixel_color = pixel_color + self.ray_color(r, self.max_depth, world)
        
        # Apply gamma correction
        r = linear_to_gamma(pixel_color.x * self.pixel_samples_scale)
        g = linear_to_gamma(pixel_color.y * self.pixel_samples_scale)
        b = linear_to_gamma(pixel_color.z * self.pixel_samples_scale)
        
        # Ensure color values are in [0, 1]
        intensity = Interval(0.0, 0.999)
        r = intensity.clamp(r)
        g = intensity.clamp(g)
        b = intensity.clamp(b)
        
        return (int(256 * r), int(256 * g), int(256 * b))
    
    def render(self, world, output_file="output.png", show_progress=True, use_multiprocessing=True):
        self.initialize()
        
        # Create an array to store the image
        img_array = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        start_time = time.time()
        
        if use_multiprocessing and mp.cpu_count() > 1:
            # Use multiprocessing to speed up rendering
            num_processes = mp.cpu_count()
            chunk_size = self.image_height // num_processes
            
            # Create shared array for the image
            shared_array = mp.RawArray('i', self.image_width * self.image_height * 3)
            
            # Function to process a chunk of the image
            def process_chunk(start_row, end_row):
                for j in range(start_row, end_row):
                    if show_progress:
                        print(f"\rRendering row {j+1}/{self.image_height}", end="", flush=True)
                    for i in range(self.image_width):
                        color = self.render_pixel(i, j, world)
                        idx = (j * self.image_width + i) * 3
                        shared_array[idx] = color[0]
                        shared_array[idx + 1] = color[1]
                        shared_array[idx + 2] = color[2]
            
            # Create and start processes
            processes = []
            for i in range(num_processes):
                start_row = i * chunk_size
                end_row = start_row + chunk_size if i < num_processes - 1 else self.image_height
                p = mp.Process(target=process_chunk, args=(start_row, end_row))
                processes.append(p)
                p.start()
            
            # Wait for all processes to finish
            for p in processes:
                p.join()
            
            # Copy data from shared array to image array
            for j in range(self.image_height):
                for i in range(self.image_width):
                    idx = (j * self.image_width + i) * 3
                    img_array[j, i, 0] = shared_array[idx]
                    img_array[j, i, 1] = shared_array[idx + 1]
                    img_array[j, i, 2] = shared_array[idx + 2]
        else:
            # Single-process rendering
            for j in tqdm(range(self.image_height), disable=not show_progress):
                for i in range(self.image_width):
                    color = self.render_pixel(i, j, world)
                    img_array[j, i, 0] = color[0]
                    img_array[j, i, 1] = color[1]
                    img_array[j, i, 2] = color[2]
        
        end_time = time.time()
        render_time = end_time - start_time
        
        # Create and save the image
        img = Image.fromarray(img_array)
        img.save(output_file)
        
        if show_progress:
            print(f"\nRendering completed in {render_time:.2f} seconds")
            print(f"Image saved to {output_file}")
        
        return img_array