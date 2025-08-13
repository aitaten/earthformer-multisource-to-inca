import os
import glob
import h5py
import zarr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.ndimage import zoom
from tqdm import tqdm
import random

import pdb


# =====================
# CONFIG
# =====================
DATA_DIR = "/scratch/evno/DATA_EF4INCA/test"
PATCH_SIZE = (40, 70)   # small patch for Colab
SUBDOMAIN_START = (80, 140)  # top-left of subdomain in full 400x700 grid
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Normalization constants (same as before)
normDict = {
    'ch05': {'mean': 260.0, 'std': 15.0},
    'ch06': {'mean': 250.0, 'std': 15.0},
    'ch07': {'mean': 270.0, 'std': 15.0},
    'ch09': {'mean': 290.0, 'std': 15.0},
    'lght': {'mean': 0.0, 'std': 1.0},
    'inca_cape': {'mean': 500.0, 'std': 300.0},
    'dem': {'mean': 800.0, 'std': 500.0},
    'lat': {'mean': 47.5, 'std': 1.0},
    'lon': {'mean': 13.0, 'std': 1.0},
}

def lght2rast(df4rast, event_raster, time_idx):
    xpixelsize = 1000.0
    ypixelsize = 1000.0
    x1 = 20000
    y2 = 620000
    df4rast = df4rast.copy()
    df4rast['pixel_x'] = ((df4rast['x_proj'] - x1) / xpixelsize).astype(int)
    df4rast['pixel_y'] = ((y2 - df4rast['y_proj']) / ypixelsize).astype(int)
    df4rast = df4rast[
        (df4rast['pixel_x'] >= 0) & (df4rast['pixel_x'] < 700) &
        (df4rast['pixel_y'] >= 0) & (df4rast['pixel_y'] < 400)
    ]
    event_counts = df4rast.groupby(['pixel_x', 'pixel_y']).size().reset_index(name='counts')
    for _, row in event_counts.iterrows():
        event_raster[time_idx, row['pixel_y'], row['pixel_x']] = row['counts']
    return event_raster


# =====================
# HELPER
# =====================
def process_file(file_path):
    """Read one H5 file and return (past, future) arrays cropped & normalized."""
    eventFile = h5py.File(file_path, 'r')

    # Get timestamp
    timestamp = file_path.split('_')[-1][:-3]
    timestamp = datetime.strptime(timestamp, '%Y%m%d%H%M')
    time_of_day = timestamp.hour + timestamp.minute / 60
    startTime_lght = timestamp - timedelta(minutes=130)

    # SEVIRI
    seviri = eventFile['seviri'][0:25,:,:,:]
    h_zoom_seviri = 400 / 86
    w_zoom_seviri = 700 / 150
    seviri = zoom(seviri, (1, h_zoom_seviri, w_zoom_seviri, 1), order=1)

    # Radar
    radar = eventFile['radar'][0:25,:,:]

    # Lightning
    lght_df = pd.DataFrame(eventFile['lght'][:,:], columns=['event_time', 'x_proj', 'y_proj'])
    lght_df['event_time'] = pd.to_datetime(lght_df['event_time'], unit='s')
    lght = np.zeros((25, 400, 700))
    for i in range(25):
        lght = lght2rast(
            lght_df[(lght_df['event_time'] > startTime_lght) &
                    (lght_df['event_time'] <= (startTime_lght + timedelta(minutes=5)))],
            lght, i
        )
        startTime_lght += timedelta(minutes=5)

    # Precipitation
    inca_precip_x = eventFile['inca_precip'][0:25,:,:]
    inca_precip_y = eventFile['inca_precip'][25:49,:,:]

    # Normalize SEVIRI
    for ch, key in enumerate(['ch05','ch06','ch07','ch09']):
        seviri[:, :, :, ch] = (seviri[:, :, :, ch] - normDict[key]['mean']) / normDict[key]['std']

    # Normalize lightning
    lght = (lght - normDict['lght']['mean']) / normDict['lght']['std']

    # Precip thresholds & log
    for arr in (inca_precip_x, inca_precip_y, radar):
        arr[arr < 0.1] = 0.02
        np.log10(arr, out=arr)

    # Crop subdomain patch
    y0, x0 = SUBDOMAIN_START
    h, w = PATCH_SIZE
    seviri = seviri[:, y0:y0+h, x0:x0+w, :]
    radar = radar[:, y0:y0+h, x0:x0+w]
    lght = lght[:, y0:y0+h, x0:x0+w]
    inca_precip_x = inca_precip_x[:, y0:y0+h, x0:x0+w]
    inca_precip_y = inca_precip_y[:, y0:y0+h, x0:x0+w]

    # Combine into past (with channels) and future
    past = np.concatenate([
        seviri,
        lght[:, :, :, np.newaxis],
        inca_precip_x[:, :, :, np.newaxis],
        radar[:, :, :, np.newaxis]
    ], axis=-1)
    future = inca_precip_y[:, :, :, np.newaxis]

    return past.astype(np.float32), future.astype(np.float32)


def has_precip(file_path):
    """Check if radar precipitation exists in subdomain."""
    with h5py.File(file_path, 'r') as f:
        radar = f['radar'][0:25,:,:]
        y0, x0 = SUBDOMAIN_START
        h, w = PATCH_SIZE
        sub_radar = radar[:, y0:y0+h, x0:x0+w]
        return np.any(sub_radar > 0.1)


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(DATA_DIR, "sampled_*.h5")))
    # files = [f for f in files if has_precip(f)]
    random.shuffle(files)

    n_total = len(files)
    n_train = int(TRAIN_FRAC * n_total)
    n_val = int(VAL_FRAC * n_total)

    split = {
        "train": files[:n_train],
        "val": files[n_train:n_train+n_val],
        "test": files[n_train+n_val:]
    }

    OUTPUT_DIR = "/scratch/evno/DATA_EF4INCA"

    for split_name, split_files in split.items():
        if not split_files:
            continue
        print(f"Creating Zarr for {split_name} with {len(split_files)} samples")

        # Create Zarr group
        first_past, first_future = process_file(split_files[0])
        n_past, h, w, c_past = first_past.shape
        n_future, _, _, c_future = first_future.shape

        out_path = os.path.join(OUTPUT_DIR, f"dataset_{split_name}.zarr")
        z = zarr.open(out_path, mode='w')

        past_arr = z.create_dataset(
            "past", shape=(len(split_files), n_past, h, w, c_past),
            chunks=(1, n_past, h, w, c_past), dtype='f4'
        )
        future_arr = z.create_dataset(
            "future", shape=(len(split_files), n_future, h, w, c_future),
            chunks=(1, n_future, h, w, c_future), dtype='f4'
        )

        # Streaming write
        for i, fpath in enumerate(tqdm(split_files, desc=f"Writing {split_name}")):
            past, future = process_file(fpath)
            past_arr[i] = past
            future_arr[i] = future

        print(f"✅ Saved {out_path} with {len(split_files)} samples")
