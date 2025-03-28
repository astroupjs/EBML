import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, Column, hstack, vstack, unique
import random

import sys, os

def add_edges(tab):
    mask=tab['Phase']<0.2
    new_tab=tab[mask].copy()
    new_tab['Phase']=new_tab['Phase']+1.

    final_tab=vstack([tab,new_tab])
    mask=tab['Phase']>0.8
    new_tab=tab[mask].copy()
    new_tab['Phase']=new_tab['Phase']-1.
    final_tab=vstack([final_tab,new_tab])
    final_tab.sort('Phase')
    return final_tab

def create_polar_hexbin(phase, flux, r_min=0.2, r_max=1., image_size=224, save_name=None):
    """
    Generates a polar hexbin plot from phase and flux data.

    Args:
        phase: The phase values (in turns, where 1 turn is a full circle).
        flux: The flux values corresponding to the phase.
        r_min: The minimum radius for the plot (default is 0.2).
        r_max: The maximum radius for the plot (default is 1.0).
        image_size: The size of the output image in pixels (square, default is 224).
        save_name: The file name to save the generated plot (default is None).

    Returns:
        A NumPy array representing the hexbin plot image.
    """

    angle = phase * 360 -90 # Subtract 90 to start from vertical line

    # Convert fluxes to radius
    if flux_min == flux_max:
        r = np.full_like(flux, r_max)  # Assign maximum radius if flux range is zero
    else:
        r = r_max - (r_max - r_min) * (flux_max - flux) / (flux_max - flux_min)

    flux_min, flux_max = flux.min(), flux.max()
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

N=5000

df=pd.read_csv("/home/parimucha/Virtual/Random_dataset/merged/period/overcontact/overcontact_nospot_period_gaia.csv")

df=df[df['period']<1.2]

sample_df = df.sample(n=N, random_state=42)

out_dir='/home/parimucha/Virtual/GaiaLC/dataset/train/nospot/'

phase=np.linspace(0.,1, 100, endpoint=True)
n=30000

#add niose and outliers to synthetic data
for index, row in sample_df.iterrows():
    flux_o=list(row[:100])
    noise1 = np.random.normal(0,0.005,100) #noise should be changed based on a passband
    flux=flux_o+noise1
    is_outlier=np.random.randint(0,5,1)
    if is_outlier>2:
        outlier_val=np.random.normal(0,0.3,1)
        outlier_pos=int(np.random.randint(0,100,1))
    else:
        outlier_pos=0
        outlier_val=0.

    flux[outlier_pos]=outlier_val+flux[outlier_pos]

    table=Table()
    table['Phase']=list(phase)
    table['Flux']=flux
    tab=table
    name=out_dir+str(n).zfill(5)+'.png'
    print(n, name)
    create_polar_hexbin(tab['Phase'], tab['Flux'],image_size=224, save_name=name)
    n=n+1

    # remove some points randomly
    number_of_points=np.random.randint(50, 80)
    random_indices = np.random.choice(len(table), size=number_of_points, replace=False)
   
    mask = np.ones(len(table), dtype=bool)  # Create a boolean mask
    mask[random_indices] = False  # Set mask to False for selected rows
    table = table[mask]

    tab=table

    name=out_dir+str(n).zfill(5)+'.png'
    print(n, name)

    create_polar_hexbin(tab['Phase'], tab['Flux'],image_size=224, save_name=name)

    del table
    del tab
    n=n+1


sys.exit()
dir='/home/parimucha/Virtual/GaiaLC/dataset/over_parameters/temperature/'

X = df

def remove_points(table, number=15):
    phase=list(table['Phase'])
    remove_phase=random.sample(phase, number)
    mask =np.invert( [item in remove_phase for item in list(table['Phase'])])
    final_tab=table[mask]


    return final_tab


tt=[]
ii=[]
qq=[]
pp=[]
n=0
for index, row in X.iterrows():
    flux_o=list(row[:50])
    t=row[50:]
    i=row[52]
    q=row[51]
    p=row[53]
    
    ii.append(i)
    qq.append(q)
    pp.append(p)


print(len(pp))
plt.hist(pp, bins=5)
plt.hist(qq, bins=5)

plt.show()
sys.exit()


n=0
for index, row in X.iterrows():


    is_outlier=np.random.randint(0,5,1)
    if is_outlier>3:
        outlier_val=np.random.normal(0,0.5,1)
        outlier_pos=int(np.random.randint(0,50,1))
    else:
        outlier_pos=0
        outlier_val=0.

    name=dir+str(n).zfill(5)+'_n.png'

sys.exit()
