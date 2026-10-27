import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# ASTROSURFACE AI
# Synthetic Lunar Crater Dataset Generator (v2, higher quality)
# ==========================================

IMAGE_SIZE = 128          # final saved size (keep 128 to match your GAN)
RENDER_SCALE = 2          # render at 2x, then downsample (anti-aliasing)
IMAGES_PER_CLASS = 100
OUTPUT_DIR = "dataset_v2"

# Crater radius in FINAL image pixels
SCENARIOS = {
    "small": (8, 12),
    "medium": (16, 21),
    "large": (25, 31),
}


def smooth_noise(size, grid_size, rng):
    """Smooth, low-frequency noise (float, zero mean, unit std)."""
    small = rng.normal(0, 1, (grid_size, grid_size)).astype(np.float32)
    img = Image.fromarray(small).resize((size, size), Image.Resampling.BICUBIC)
    noise = np.array(img, dtype=np.float32)
    return (noise - noise.mean()) / (noise.std() + 1e-6)


def fractal_noise(size, rng, base_grid=4, persistence=0.5):
    """Multi-octave noise: large hills plus fine regolith texture."""
    total = np.zeros((size, size), dtype=np.float32)
    amplitude, grid = 1.0, base_grid
    while grid <= size // 4:
        total += amplitude * smooth_noise(size, grid, rng)
        amplitude *= persistence
        grid *= 2
    return total / (total.std() + 1e-6)


def crater_profile(radius, angle, r, depth, rng, central_peak=False):
    """Height contribution of one crater (in pixels)."""
    # Irregular rim shape
    wobble = np.ones_like(radius)
    for k in (2, 3, 5, 7):
        wobble += rng.uniform(0.005, 0.03) * np.sin(k * angle + rng.uniform(0, 2 * np.pi))

    d = radius / (r * wobble)                      # normalised distance (1.0 = rim)
    inside = d < 1.0

    # Bowl: parabolic, steep near the wall
    bowl = -depth * np.clip(1 - d**2, 0, 1) ** 0.8

    # Rim: sharp on the inside, softer falling off outside
    rim_h = depth * 0.35
    rim = rim_h * np.where(
        inside,
        np.exp(-((d - 1) / 0.10) ** 2),
        np.exp(-((d - 1) / 0.30) ** 2),
    )

    # Ejecta blanket thinning with distance
    ejecta = np.where(inside, 0.0, depth * 0.10 * np.power(np.maximum(d, 1.0), -3.0))

    height = bowl + rim + ejecta

    if central_peak:
        height += depth * 0.45 * np.exp(-(d / 0.10) ** 2)

    return height, d


def create_terrain(size, crater_radius, rng, scale):
    """Render one crater image at high resolution (size already scaled)."""
    S = size
    r = crater_radius * scale

    y, x = np.mgrid[0:S, 0:S].astype(np.float32)
    cx = S / 2 + rng.uniform(-3, 3) * scale
    cy = S / 2 + rng.uniform(-3, 3) * scale
    dx, dy = x - cx, y - cy
    radius = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    # 1. Base terrain (fractal)
    height = fractal_noise(S, rng) * (0.010 * S)

    # 2. Small secondary craters scattered around
    for _ in range(rng.integers(10, 20)):
        rs = min(S * 0.012 * (1 + rng.pareto(2.0)), S * 0.05)
        sx, sy = rng.uniform(0, S, 2)
        sr = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
        sa = np.arctan2(y - sy, x - sx)
        h, _ = crater_profile(sr, sa, rs, rs * 0.25, rng)
        height += h

    # 3. Main crater
    h, d = crater_profile(
        radius, angle, r, depth=r * 0.28, rng=rng,
        central_peak=crater_radius >= 22,
    )
    height += h

    # 4. Albedo: patchy brightness + bright ray streaks around the crater
    albedo = 1.0 + 0.05 * smooth_noise(S, 8, rng)
    rays = np.zeros_like(angle)
    for k in rng.integers(10, 40, 6):
        rays += rng.uniform(0.5, 1.0) * np.sin(k * angle + rng.uniform(0, 2 * np.pi))
    rays /= 3.0
    albedo += 0.08 * rays * np.where(d > 1, np.exp(-(d - 1) / 1.5), 0.0)

    # 5. Lighting from surface normals (low sun, upper-left)
    gy, gx = np.gradient(height)
    relief = 1.6
    nx, ny, nz = -gx * relief, -gy * relief, np.ones_like(gx)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)

    light = np.array([-0.6, -0.6, 0.55])
    light /= np.linalg.norm(light)
    shade = (nx * light[0] + ny * light[1] + nz * light[2]) / norm
    shade = np.clip(shade, 0, 1)

    image = albedo * shade

    # 6. Contrast stretch
    lo, hi = np.percentile(image, [0.5, 99.5])
    image = np.clip((image - lo) / (hi - lo + 1e-6), 0, 1)
    image = 0.04 + 0.92 * image
    return image.astype(np.float32)


def finalize(image_float, out_size, rng):
    """Downsample (anti-alias) and add slight sensor grain."""
    img = Image.fromarray(image_float).resize(
        (out_size, out_size), Image.Resampling.LANCZOS
    )
    arr = np.array(img, dtype=np.float32)
    arr = arr + rng.normal(0, 0.012, arr.shape)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def main():
    rng = np.random.default_rng(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preview_images = []

    print("Generating AstroSurface AI dataset...")
    print("--------------------------------------")

    render_size = IMAGE_SIZE * RENDER_SCALE

    for label, (min_r, max_r) in SCENARIOS.items():
        folder = os.path.join(OUTPUT_DIR, label)
        os.makedirs(folder, exist_ok=True)

        for index in range(IMAGES_PER_CLASS):
            crater_radius = rng.uniform(min_r, max_r)

            hi_res = create_terrain(render_size, crater_radius, rng, RENDER_SCALE)
            image = finalize(hi_res, IMAGE_SIZE, rng)

            Image.fromarray(image).save(os.path.join(folder, f"{label}_{index:04d}.png"))

            if index == 0:
                preview_images.append((label, image))

        print(f"Created {IMAGES_PER_CLASS} images for {label.upper()} craters.")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (label, image) in zip(axes, preview_images):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"{label.capitalize()} Crater")
        axis.axis("off")

    plt.suptitle("AstroSurface AI - Synthetic Crater Preview")
    plt.tight_layout()
    plt.show()

    print("--------------------------------------")
    print("Dataset generation completed!")
    print("Images saved in:", os.path.abspath(OUTPUT_DIR))
    print("Total images:", len(SCENARIOS) * IMAGES_PER_CLASS)
    print("Note: These are synthetic crater illustrations, not real lunar observations.")


if __name__ == "__main__":
    main()