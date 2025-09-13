import os

class Animation:
    def __init__(self, camera, world_func, num_frames=30, output_dir="animation"):
        self.camera = camera
        self.world_func = world_func  # Function that takes a frame number and returns a world
        self.num_frames = num_frames
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def render(self, show_progress=True, use_multiprocessing=True):
        for frame in range(self.num_frames):
            if show_progress:
                print(f"\nRendering frame {frame+1}/{self.num_frames}")
            
            # Get the world for this frame
            world = self.world_func(frame, self.num_frames)
            
            # Render the frame
            output_file = f"{self.output_dir}/frame_{frame:04d}.png"
            self.camera.render(world, output_file, show_progress, use_multiprocessing)