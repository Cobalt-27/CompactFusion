import torch
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips

# ps: pixel value should be in [0, 1]!

NUMBER_OF_VIDEOS = 200
VIDEO_LENGTH = 49
CHANNEL = 3
TARGET_SIZE_HW = (480, 720) 
videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, TARGET_SIZE_HW[0], TARGET_SIZE_HW[1], requires_grad=False)
videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, TARGET_SIZE_HW[0], TARGET_SIZE_HW[1], requires_grad=False)


device = torch.device("cuda")
# device = torch.device("cpu")

import json
result = {}
only_final = True
# result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=only_final)
# result['fvd'] = calculate_fvd(videos1, videos2, device, method='videogpt', only_final=only_final)
# result['ssim'] = calculate_ssim(videos1, videos2, only_final=only_final)
# result['psnr'] = calculate_psnr(videos1, videos2, only_final=only_final)
result['lpips'] = calculate_lpips(videos1, videos2, device, only_final=only_final)
print(json.dumps(result, indent=4))