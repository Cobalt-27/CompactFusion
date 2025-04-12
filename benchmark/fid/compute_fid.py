import argparse
import logging
import time
import torch
from cleanfid import fid
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio
from pathlib import Path
from tqdm import tqdm
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fid_computation.log')
        ]
    )

def compute_fid_score(ref_path: str, sample_path: str, device: str = "cuda") -> float:
    """
    Compute FID score
    
    Args:
        ref_path: Path to ref images directory
        sample_path: Path to sample images directory
        device: Computing device ('cuda' or 'cpu')
    
    Returns:
        float: FID score
    
    Raises:
        ValueError: If directory does not exist
    """
    # Check if paths exist
    ref_dir = Path(ref_path)
    gen_dir = Path(sample_path)
    
    if not ref_dir.exists():
        raise ValueError(f"ref images directory does not exist: {ref_path}")
    if not gen_dir.exists():
        raise ValueError(f"sample images directory does not exist: {sample_path}")
    
    logging.info(f"Starting FID score computation")
    logging.info(f"ref images directory: {ref_path}")
    logging.info(f"sample images directory: {sample_path}")
    logging.info(f"Using device: {device}")
    
    start_time = time.time()
    
    try:
        # Convert string device to torch.device
        torch_device = torch.device(device)
        
        score = fid.compute_fid(
            ref_path,
            sample_path,
            device=torch_device,
            num_workers=8,  # Can be adjusted as needed
            use_dataparallel=False  # Disable DataParallel to avoid device issues
        )
        
        elapsed_time = time.time() - start_time
        logging.info(f"FID computation completed, time elapsed: {elapsed_time:.2f} seconds")
        return score
        
    except Exception as e:
        logging.error(f"Error occurred during FID computation: {str(e)}")
        raise

def compute_lpips_score(ref_path: str, sample_path: str, device: str = "cuda") -> float:
    """
    Compute LPIPS score
    
    Args:
        ref_path: Path to ref images directory
        sample_path: Path to sample images directory
        device: Computing device ('cuda' or 'cpu')
    
    Returns:
        float: LPIPS score
    """
    ref_dir = Path(ref_path)
    gen_dir = Path(sample_path)
    
    if not ref_dir.exists():
        raise ValueError(f"ref images directory does not exist: {ref_path}")
    if not gen_dir.exists():
        raise ValueError(f"sample images directory does not exist: {sample_path}")
    
    # Convert string device to torch.device
    torch_device = torch.device(device)
    
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(torch_device)
    psnr = PeakSignalNoiseRatio().to(torch_device)

    ref_images = sorted(ref_dir.glob('*.png'))
    gen_images = sorted(gen_dir.glob('*.png'))
    
    # Define transform to convert PIL images to tensors
    to_tensor = transforms.ToTensor()
    
    lpips_scores = []
    psnr_scores = []
    
    # Create a list of paired images
    image_pairs = list(zip(ref_images, gen_images))
    
    # Use tqdm with a descriptive message and total count
    for ref_img_path, gen_img_path in tqdm(image_pairs, desc="Computing LPIPS & PSNR", total=len(image_pairs)):
        # Load images and resize to ensure dimensions match
        ref_img = Image.open(ref_img_path).convert('RGB')
        gen_img = Image.open(gen_img_path).convert('RGB')
        
        # Resize both images to a common size (1024x1024) to avoid dimension mismatch
        ref_img = ref_img.resize((1024, 1024), Image.LANCZOS)
        gen_img = gen_img.resize((1024, 1024), Image.LANCZOS)
        
        # Convert to tensors and move to device
        ref_tensor = to_tensor(ref_img).unsqueeze(0).to(torch_device)
        gen_tensor = to_tensor(gen_img).unsqueeze(0).to(torch_device)
        
        lpips_score = lpips(ref_tensor, gen_tensor)
        psnr_score = psnr(ref_tensor, gen_tensor)
        
        lpips_scores.append(lpips_score.item())
        psnr_scores.append(psnr_score.item())
    lpips_mean = sum(lpips_scores) / len(lpips_scores)
    psnr_mean = sum(psnr_scores) / len(psnr_scores)
    
    return lpips_mean, psnr_mean    

def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Compute FID score')
    parser.add_argument('--ref', type=str, required=True,
                      help='Path to ref images directory')
    parser.add_argument('--sample', type=str, required=True,
                      help='Path to sample images directory')
    parser.add_argument('--device', type=str, default="cuda",
                      choices=['cuda', 'cpu'], help='Computing device')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        # Compute FID
        # score = compute_fid_score(args.ref, args.sample, args.device)
        
        # Compute LPIPS
        lpips_mean, psnr_mean = compute_lpips_score(args.ref, args.sample, args.device)
        
        # Output result
        # logging.info(f"FID score: {score:.4f}")
        logging.info(f"LPIPS score: {lpips_mean:.4f}")
        logging.info(f"PSNR score: {psnr_mean:.4f}")
        
    except Exception as e:
        logging.error(f"Program execution failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())