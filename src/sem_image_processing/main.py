from pathlib import Path
import matplotlib.pyplot as plt

from .config import Config
from .io import load_image, save_image
from .preprocess import crop_bottom, denoise, remove_scale_bar, remove_background, clahe
from .segmentation import segment

def main():

    cfg = Config()

    

    image = load_image(cfg.image_path)
    save_image(cfg.output_dir / "01_original.png", image)

    #image = remove_scale_bar(image)
    #save_image(cfg.output_dir / "02_no_scalebar.png", image)


    image = crop_bottom(image, cfg.crop_bottom_pixels)
    save_image(cfg.output_dir / "03_cropped.png", image)

    image = denoise(image, cfg.gaussian_kernel)
    save_image(cfg.output_dir / "04_denoised.png", image)

    #image = clahe(image)
    #save_image(cfg.output_dir / "05_clahe.png", image)


    #corrected, background = remove_background(image)

    #save_image(cfg.output_dir / "06_background.png", background)
    #save_image(cfg.output_dir / "07_corrected.png", corrected)

    binary = segment(image)
    save_image(cfg.output_dir / "08_binary.png", binary)

    print("Done")
    image = load_image(cfg.image_path)

    print(f"Shape: {image.shape}")
    print(f"Dtype: {image.dtype}")
    print(f"Min: {image.min()}")
    print(f"Max: {image.max()}")
    print(f"Mean: {image.mean():.2f}")


    plt.hist(image.ravel(), bins=256)
    plt.show()


if __name__ == "__main__":
    main()