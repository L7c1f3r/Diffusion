import os
import subprocess
import shutil


def run_feature_extraction(image_path, exp_name, configs_dir, models_dir):
    """
    Step 1: Feature Extraction
    """
    cmd = [
        "python", "run_features_extraction.py",
        "--config", os.path.join(configs_dir, "pnp", "feature-extraction-real.yaml"),
        "--model_config", os.path.join(configs_dir, "stable-diffusion", "v1-inference.yaml"),
        "--ckpt", os.path.join(models_dir, "ldm", "stable-diffusion-v1", "model.ckpt"),
        "--precision", "autocast"
    ]
    print("Running Feature Extraction...")
    subprocess.run(cmd, check=True)
    print("Feature Extraction completed.\n")


def run_pnp(exp_name, configs_dir, models_dir):
    """
    Step 2: Plug-and-Play (PnP) Image Generation
    """
    cmd = [
        "python", "run_pnp.py",
        "--config", os.path.join(configs_dir, "pnp", "pnp-real.yaml"),
        "--model_config", os.path.join(configs_dir, "stable-diffusion", "v1-inference.yaml"),
        "--ckpt", os.path.join(models_dir, "ldm", "stable-diffusion-v1", "model.ckpt"),
        "--precision", "autocast"
    ]
    print("Running PnP Generation...")
    subprocess.run(cmd, check=True)
    print("PnP Generation completed.\n")


def copy_pnp_outputs(exp_output_root, content_images_dir):
    """
    Step 2.5: Copy PnP outputs (translations folder) to StyleID content folder
    """
    os.makedirs(content_images_dir, exist_ok=True)
    copied_count = 0

    for root, dirs, files in os.walk(exp_output_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg')) and "translations" in root:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(content_images_dir, file)
                shutil.copy(src_path, dst_path)
                copied_count += 1

    print(f"Copied {copied_count} images from PnP outputs to {content_images_dir}\n")
    if copied_count == 0:
        print("⚠️ Warning: No PnP images were found. Check if Step 2 generated images correctly.")
    return copied_count


def run_styleid(content_dir, style_dir, output_dir, configs_dir, models_dir, precomputed_dir):
    """
    Step 3: StyleID-based Style Transfer
    """
    cmd = [
        "python", "run_styleid.py",
        "--cnt", content_dir,
        "--sty", style_dir,
        "--output_path", output_dir,
        "--precomputed", precomputed_dir,
        "--model_config", os.path.join(models_dir, "ldm", "stable-diffusion-v1", "v1-inference.yaml"),
        "--ckpt", os.path.join(models_dir, "ldm", "stable-diffusion-v1", "model.ckpt"),
        "--precision", "autocast"
    ]
    print("Running StyleID...")
    subprocess.run(cmd, check=True)
    print("StyleID completed.\n")


def main():
    # ----------- Paths Configuration -----------
    input_image = "./input_images/example.jpg"
    experiment_name = "test_experiment"

    configs_directory = "./configs"
    models_directory = "./models"

    exp_output_root = f"./outputs/{experiment_name}"
    content_images_dir = f"{exp_output_root}/pnp_outputs"     # Step 2.5目标
    style_images_dir = "./style_images"                       # 需提前准备style图片
    final_output_dir = f"{exp_output_root}/styled_results"
    precomputed_features_dir = f"{exp_output_root}/precomputed_feats"

    os.makedirs(exp_output_root, exist_ok=True)
    os.makedirs(content_images_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)
    os.makedirs(precomputed_features_dir, exist_ok=True)

    # ----------- Step 1: Feature Extraction -----------
    run_feature_extraction(
        image_path=input_image,
        exp_name=experiment_name,
        configs_dir=configs_directory,
        models_dir=models_directory
    )

    # ----------- Step 2: PnP Generation -----------
    run_pnp(
        exp_name=experiment_name,
        configs_dir=configs_directory,
        models_dir=models_directory
    )

    # ----------- Step 2.5: Auto Copy PnP Outputs -----------
    copy_pnp_outputs(
        exp_output_root=exp_output_root,
        content_images_dir=content_images_dir
    )

    # ----------- Step 3: StyleID -----------
    run_styleid(
        content_dir=content_images_dir,
        style_dir=style_images_dir,
        output_dir=final_output_dir,
        configs_dir=configs_directory,
        models_dir=models_directory,
        precomputed_dir=precomputed_features_dir
    )

    print("All steps completed successfully.\nCheck outputs in:", final_output_dir)


if __name__ == "__main__":
    main()