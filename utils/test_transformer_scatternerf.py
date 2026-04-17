#!/usr/bin/env python3
"""
Quick smoke test for ScatterNeRF backbones.

This checks that:
1) `backbone_type="mlp"` forward works.
2) `backbone_type="transformer"` forward works.
3) output tensor shapes match for easy metric comparisons.
"""

import torch

from src.model.scatternerf.model import ScatterNeRF


def build_dummy_rays(num_rays=16):
    rays_o = torch.randn(num_rays, 3)
    rays_d = torch.randn(num_rays, 3)
    viewdirs = torch.nn.functional.normalize(rays_d, dim=-1)
    return {"rays_o": rays_o, "rays_d": rays_d, "viewdirs": viewdirs}


def run_one(backbone_type):
    rays = build_dummy_rays()
    model = ScatterNeRF(
        backbone_type=backbone_type,
        num_coarse_samples=16,
        num_fine_samples=16,
        noise_std=0.0,
    )
    outputs = model(rays, randomized=False, white_bkgd=True, near=0.1, far=2.0)
    coarse_rgb = outputs[0][0]
    fine_rgb = outputs[1][0]
    coarse_depth = outputs[0][3]
    fine_depth = outputs[1][3]
    return {
        "coarse_rgb": tuple(coarse_rgb.shape),
        "fine_rgb": tuple(fine_rgb.shape),
        "coarse_depth": tuple(coarse_depth.shape),
        "fine_depth": tuple(fine_depth.shape),
    }


def main():
    mlp_shapes = run_one("mlp")
    tr_shapes = run_one("transformer")

    for key in mlp_shapes:
        if mlp_shapes[key] != tr_shapes[key]:
            raise RuntimeError(
                f"Shape mismatch for {key}: mlp={mlp_shapes[key]} transformer={tr_shapes[key]}"
            )

    print("Smoke test passed.")
    print("Shared output shapes:", mlp_shapes)


if __name__ == "__main__":
    main()
