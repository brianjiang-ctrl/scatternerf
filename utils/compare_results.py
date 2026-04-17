#!/usr/bin/env python3
import argparse
import json


def load_metrics(path):
    with open(path, "r") as f:
        d = json.load(f)
    return {
        "PSNR": d["PSNR"]["test"],
        "SSIM": d["SSIM"]["test"],
        "LPIPS": d["LPIPS"]["test"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Path to baseline results.json")
    parser.add_argument("--candidate", required=True, help="Path to candidate results.json")
    args = parser.parse_args()

    base = load_metrics(args.baseline)
    cand = load_metrics(args.candidate)

    print("Metric comparison (candidate - baseline):")
    print(f"PSNR : {cand['PSNR']:.4f} - {base['PSNR']:.4f} = {cand['PSNR'] - base['PSNR']:+.4f}")
    print(f"SSIM : {cand['SSIM']:.4f} - {base['SSIM']:.4f} = {cand['SSIM'] - base['SSIM']:+.4f}")
    print(f"LPIPS: {cand['LPIPS']:.4f} - {base['LPIPS']:.4f} = {cand['LPIPS'] - base['LPIPS']:+.4f}")
    print("")
    print("Interpretation:")
    print("- Higher PSNR is better")
    print("- Higher SSIM is better")
    print("- Lower LPIPS is better")


if __name__ == "__main__":
    main()
