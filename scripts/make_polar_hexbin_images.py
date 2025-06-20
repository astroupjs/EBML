import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def create_polar_hexbin(phase, flux, r_min=0.2, r_max=1., image_size=224, save_name=None):
    """
    Generates a polar hexbin plot from phase and flux data and saves it as an image.

    Args:
        phase: The phase values (in turns, where 1 turn is a full circle).
        flux: The flux values corresponding to the phase.
        r_min: The minimum radius for the plot (default is 0.2).
        r_max: The maximum radius for the plot (default is 1.0).
        image_size: The size of the output image in pixels (square, default is 224).
        save_name: The file name to save the generated plot (default is None).
    """
    angle = phase * 360 - 90 # Subtract 90 to start from vertical line

    # Convert fluxes to radius
    flux_min, flux_max = flux.min(), flux.max()
    if flux_min == flux_max:
        r = np.full_like(flux, r_max)  # Assign maximum radius if flux range is zero
    else:
        r = r_max - (r_max - r_min) * (flux_max - flux) / (flux_max - flux_min)

    # Calculate x and y coordinates in radial coordinates
    x = r * np.cos(np.deg2rad(angle))
    y = r * np.sin(np.deg2rad(angle))

    fig, ax = plt.subplots(figsize=(image_size / 100, image_size / 100), dpi=100)

    # Create the hexbin plot
    hb = ax.hexbin(x, y, gridsize=30, cmap="viridis", mincnt=1)

    # Remove axes, margins, and grid
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

    plt.xticks([])
    plt.yticks([])
    ax.grid(False)
    fig.tight_layout(pad=0)
    plt.savefig(save_name, dpi=100, bbox_inches='tight',pad_inches = 0.1)
    plt.close(fig)
    del fig

    return True

# Example usage:
def create_images_from_dataframe(df, out_dir, n_start=0, image_size=224, passband='gaia', name=''):
    os.makedirs(out_dir, exist_ok=True)
    phase = np.linspace(0., 1, 100, endpoint=True)
    n = n_start
    # Set parameters based on passband
    if passband == 'gaia':
        noise_std = 0.005
        outlier_std = 0.3
        remove_min, remove_max = 0, 80
    elif passband == 'tess':
        noise_std = 0.002
        outlier_std = 0.15
        remove_min, remove_max = 0, 1
    elif passband == 'ogle':
        noise_std = 0.01
        outlier_std = 0.5
        remove_min, remove_max = 0, 1
    else:
        noise_std = 0.005
        outlier_std = 0.3
        remove_min, remove_max = 0, 30
    for idx, row in df.iterrows():
        flux = row[:100].values.astype(float)
        # Add noise
        flux = flux + np.random.normal(0, noise_std, 100)
        # Add outlier
        if np.random.rand() > 0.6:
            outlier_val = np.random.normal(0, outlier_std)
            outlier_pos = np.random.randint(0, 100)
            flux[outlier_pos] += outlier_val
        # Remove random points
        n_remove = np.random.randint(remove_min, remove_max+1)
        if n_remove > 0:
            remove_idx = np.random.choice(100, n_remove, replace=False)
            mask = np.ones(100, dtype=bool)
            mask[remove_idx] = False
            phase_plot = phase[mask]
            flux_plot = flux[mask]
        else:
            phase_plot = phase
            flux_plot = flux
        if not name:
            name = os.path.join(out_dir, f"{str(n).zfill(5)}.png")
        create_polar_hexbin(phase_plot, flux_plot, image_size=image_size, save_name=name)
        n += 1

def create_polar_hexbin_from_ecsv(ecsv_path, out_dir=None, image_size=224):
    """
    Create a polar hexbin image from a real Gaia light curve ECSV file.
    Uses the 'Phase' and 'norm_FG' columns.
    Saves the image as a PNG with the same basename as the ECSV file.
    Args:
        ecsv_path (str): Path to the ECSV file.
        out_dir (str or None): Output directory for the PNG. If None, saves in the same directory as the ECSV.
        image_size (int): Size of the output image (square, in pixels).
    Returns:
        out_path (str): Path to the saved PNG file.
    """
    from astropy.table import Table
    import os
    table = Table.read(ecsv_path, format='ascii.ecsv')
    phase = np.array(table['Phase'])
    flux = np.array(table['norm_FG'])
    if out_dir is None:
        out_dir = os.path.dirname(ecsv_path)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(ecsv_path))[0]
    out_path = os.path.join(out_dir, base + '.png')
    create_polar_hexbin(phase, flux, image_size=image_size, save_name=out_path)
    return out_path

# Example DataFrame usage:
# df = pd.read_csv('your_lightcurve_file.csv')
# create_images_from_dataframe(df, './output_images')
