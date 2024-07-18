import numpy as np
import torch
from PIL import Image
import argparse

from unidepth.models import UniDepthV1, UniDepthV2
from unidepth.utils import colorize, image_grid

# time check
import math
import time

def resize_image(image, target_size):
    return np.array(Image.fromarray(image).resize(target_size[::-1], Image.BILINEAR))

def demo(model, image_path, x, y):
    rgb = np.array(Image.open(image_path))
    depth_gt = np.array(Image.open("/home/user/falcon/UniDepth/assets/demo/depth.png")).astype(float) / 1000.0
    # Resize depth_gt to match the RGB image size
    depth_gt_resized = resize_image(depth_gt, rgb.shape[:2])

    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    intrinsics_torch = torch.from_numpy(np.load("/home/user/falcon/UniDepth/assets/demo/intrinsics.npy"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_torch = rgb_torch.to(device)
    intrinsics_torch = intrinsics_torch.to(device)
    # predict
    predictions = model.infer(rgb_torch, intrinsics_torch)

    # get GT and pred
    depth_pred = predictions["depth"].squeeze().cpu().numpy()

    # compute error, you have zero divison where depth_gt == 0.0
    depth_arel = np.abs(depth_gt_resized - depth_pred) / depth_gt_resized
    depth_arel[depth_gt_resized == 0.0] = 0.0

    # colorize
    depth_pred_col = colorize(depth_pred, vmin=0.01, vmax=10.0, cmap="magma_r")
    depth_gt_col = colorize(depth_gt_resized, vmin=0.01, vmax=10.0, cmap="magma_r")
    depth_error_col = colorize(depth_arel, vmin=0.0, vmax=0.2, cmap="coolwarm")

    # save image with pred and error
    artifact = image_grid([rgb, depth_gt_col, depth_pred_col, depth_error_col], 2, 2)
    Image.fromarray(artifact).save("output.png")
    #print("Available predictions:", list(predictions.keys()))
    #print(f"ARel: {depth_arel[depth_gt_resized > 0].mean() * 100:.2f}%")

    # 특정 좌표(x, y)의 깊이 출력
    depth_at_point = depth_pred[y, x]
    #print(f"Depth at ({x}, {y}): {depth_at_point:.4f} meters")
    # print(f"Depth at ({x}, {y}): {depth_at_point:.4f} meters")
    print(depth_at_point)
    return depth_at_point


if __name__ == "__main__":
    
    start = time.time()
    math.factorial(100000)
    
    parser = argparse.ArgumentParser(description='Depth Prediction Demo')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--x', type=int, required=True, help='X coordinate')
    parser.add_argument('--y', type=int, required=True, help='Y coordinate')
    args = parser.parse_args()

    #print("Torch version:", torch.__version__)
    name = "unidepth-v2-vitl14"
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # time check
    end_point = time.time()
    # print(f"depth model lading time {end_point - start:.5f} sec")

    demo(model, args.image, args.x, args.y)

    # time check
    end_point2 = time.time()
    # print(f"depth model predict time {end_point2 - end_point:.5f} sec")

# python3 unidepth.py --image {path} --x 400 --y