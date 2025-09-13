import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

from .camera import Camera
from .scenes import random_scene, simple_scene, animated_scene

class DenoiserDataGenerator:
    """
    Generates training data for the denoiser by rendering the same scene with different sample counts.
    """
    def __init__(self, output_dir="training_data"):
        """
        Initialize the data generator.
        
        Args:
            output_dir: Directory to save the generated data
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(os.path.join(output_dir, "noisy"))
            os.makedirs(os.path.join(output_dir, "clean"))
    
    def generate_scene_pair(self, scene_func, scene_id, low_samples=4, high_samples=100, 
                           image_width=200, use_multiprocessing=True):
        """
        Generate a pair of low-sample and high-sample renders of the same scene.
        
        Args:
            scene_func: Function that returns a scene
            scene_id: Identifier for the scene
            low_samples: Number of samples per pixel for the noisy image
            high_samples: Number of samples per pixel for the clean image
            image_width: Width of the rendered image
            use_multiprocessing: Whether to use multiprocessing for rendering
            
        Returns:
            Tuple of (noisy_image_path, clean_image_path)
        """
        # Create the scene
        world = scene_func() if callable(scene_func) and not isinstance(scene_func, type) else scene_func
        
        # Create camera for low-sample render
        low_cam = Camera()
        low_cam.image_width = image_width
        low_cam.samples_per_pixel = low_samples
        low_cam.max_depth = 50
        
        # Set camera position based on scene type
        if scene_func == random_scene or scene_func == animated_scene:
            low_cam.lookfrom = (13, 2, 3)
            low_cam.lookat = (0, 0, 0)
            low_cam.vup = (0, 1, 0)
            low_cam.vfov = 20
            low_cam.defocus_angle = 0.6
            low_cam.focus_dist = 10.0
        else:  # simple scene
            low_cam.lookfrom = (0, 0, 0)
            low_cam.lookat = (0, 0, -1)
            low_cam.vup = (0, 1, 0)
            low_cam.vfov = 90
        
        # Create camera for high-sample render (same settings except sample count)
        high_cam = Camera()
        high_cam.image_width = image_width
        high_cam.samples_per_pixel = high_samples
        high_cam.max_depth = 50
        
        # Copy camera settings
        high_cam.lookfrom = low_cam.lookfrom
        high_cam.lookat = low_cam.lookat
        high_cam.vup = low_cam.vup
        high_cam.vfov = low_cam.vfov
        high_cam.defocus_angle = low_cam.defocus_angle
        high_cam.focus_dist = low_cam.focus_dist
        
        # Render the images
        noisy_path = os.path.join(self.output_dir, "noisy", f"scene_{scene_id}.png")
        clean_path = os.path.join(self.output_dir, "clean", f"scene_{scene_id}.png")
        
        print(f"Rendering noisy image (samples={low_samples})...")
        low_cam.render(world, noisy_path, use_multiprocessing=use_multiprocessing)
        
        print(f"Rendering clean image (samples={high_samples})...")
        high_cam.render(world, clean_path, use_multiprocessing=use_multiprocessing)
        
        return noisy_path, clean_path
    
    def generate_dataset(self, num_scenes=10, low_samples=4, high_samples=100, 
                        image_width=200, use_multiprocessing=True):
        """
        Generate a dataset of noisy and clean image pairs.
        
        Args:
            num_scenes: Number of scenes to generate
            low_samples: Number of samples per pixel for the noisy images
            high_samples: Number of samples per pixel for the clean images
            image_width: Width of the rendered images
            use_multiprocessing: Whether to use multiprocessing for rendering
            
        Returns:
            List of tuples (noisy_image_path, clean_image_path)
        """
        scene_pairs = []
        
        # Generate simple scene
        print("Generating simple scene...")
        noisy_path, clean_path = self.generate_scene_pair(
            simple_scene, "simple", low_samples, high_samples, image_width, use_multiprocessing
        )
        scene_pairs.append((noisy_path, clean_path))
        
        # Generate random scenes
        for i in tqdm(range(num_scenes - 1), desc="Generating random scenes"):
            # Create a random scene with random parameters
            def random_scene_with_params():
                # Create a random scene with some variations
                world = random_scene()
                return world
            
            noisy_path, clean_path = self.generate_scene_pair(
                random_scene_with_params, f"random_{i}", low_samples, high_samples, 
                image_width, use_multiprocessing
            )
            scene_pairs.append((noisy_path, clean_path))
        
        return scene_pairs


class DenoiserDataset(Dataset):
    """
    Dataset for training the denoiser.
    """
    def __init__(self, data_dir="training_data", transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the training data
            transform: Optional transform to apply to the data
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all noisy images
        self.noisy_dir = os.path.join(data_dir, "noisy")
        self.clean_dir = os.path.join(data_dir, "clean")
        
        self.image_files = [f for f in os.listdir(self.noisy_dir) if f.endswith('.png')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load noisy and clean images
        noisy_path = os.path.join(self.noisy_dir, img_name)
        clean_path = os.path.join(self.clean_dir, img_name)
        
        noisy_img = Image.open(noisy_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')
        
        # Convert to numpy arrays and normalize to [0, 1]
        noisy_np = np.array(noisy_img) / 255.0
        clean_np = np.array(clean_img) / 255.0
        
        # Convert to tensors
        noisy_tensor = torch.from_numpy(noisy_np).permute(2, 0, 1).float()
        clean_tensor = torch.from_numpy(clean_np).permute(2, 0, 1).float()
        
        # Apply transform if provided
        if self.transform:
            noisy_tensor = self.transform(noisy_tensor)
            clean_tensor = self.transform(clean_tensor)
        
        return noisy_tensor, clean_tensor


def get_dataloader(data_dir="training_data", batch_size=4, shuffle=True, num_workers=4):
    """
    Create a DataLoader for the denoiser dataset.
    
    Args:
        data_dir: Directory containing the training data
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker threads for loading data
        
    Returns:
        DataLoader for the denoiser dataset
    """
    dataset = DenoiserDataset(data_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )
    return dataloader