import argparse
import os
import time

from .camera import Camera
from .scenes import random_scene, simple_scene, animated_scene
from .animation import Animation
from .vec3 import Vec3, Point3

def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('--scene', type=str, default='simple', choices=['simple', 'random', 'animated'],
                        help='Scene to render (default: simple)')
    parser.add_argument('--width', type=int, default=400,
                        help='Image width in pixels (default: 400)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Samples per pixel (default: 100)')
    parser.add_argument('--depth', type=int, default=50,
                        help='Maximum ray bounce depth (default: 50)')
    parser.add_argument('--output', type=str, default='output.png',
                        help='Output file name (default: output.png)')
    parser.add_argument('--no-multiprocessing', action='store_true',
                        help='Disable multiprocessing')
    parser.add_argument('--frames', type=int, default=30,
                        help='Number of frames for animation (default: 30)')
    parser.add_argument('--animation-dir', type=str, default='animation',
                        help='Directory for animation frames (default: animation)')
    
    args = parser.parse_args()
    
    # Create camera
    cam = Camera()
    cam.image_width = args.width
    cam.samples_per_pixel = args.samples
    cam.max_depth = args.depth
    
    # Set camera position and orientation
    if args.scene == 'random' or args.scene == 'animated':
        cam.lookfrom = Point3(13, 2, 3)
        cam.lookat = Point3(0, 0, 0)
        cam.vup = Vec3(0, 1, 0)
        cam.vfov = 20
        cam.defocus_angle = 0.6
        cam.focus_dist = 10.0
    else:  # simple scene
        cam.lookfrom = Point3(0, 0, 0)
        cam.lookat = Point3(0, 0, -1)
        cam.vup = Vec3(0, 1, 0)
        cam.vfov = 90
    
    # Render the scene
    start_time = time.time()
    
    if args.scene == 'simple':
        world = simple_scene()
        cam.render(world, args.output, use_multiprocessing=not args.no_multiprocessing)
    elif args.scene == 'random':
        world = random_scene()
        cam.render(world, args.output, use_multiprocessing=not args.no_multiprocessing)
    elif args.scene == 'animated':
        animation = Animation(cam, animated_scene, args.frames, args.animation_dir)
        animation.render(use_multiprocessing=not args.no_multiprocessing)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Total rendering time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()