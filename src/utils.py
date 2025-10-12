import re
import os
import pandas as pd
import multiprocessing
from time import time as timer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import requests
import urllib

def download_image(image_link, savefolder, timeout=10):
    """Download a single image with proper timeout and error handling"""
    if not isinstance(image_link, str):
        return None

    filename = Path(image_link).name
    image_save_path = os.path.join(savefolder, filename)

    if os.path.exists(image_save_path):
        return image_save_path  # Already downloaded

    try:
        # Use requests with timeout for better control
        response = requests.get(image_link, timeout=timeout, stream=True)
        response.raise_for_status()  # Raise exception for bad status codes

        # Download in chunks to handle large files
        with open(image_save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return image_save_path

    except requests.exceptions.Timeout:
        print(f'Timeout ({timeout}s): {image_link}')
    except requests.exceptions.RequestException as e:
        print(f'Download failed: {image_link} - {e}')
    except Exception as e:
        print(f'Unexpected error: {image_link} - {e}')

    return None

def download_images(image_links, download_folder, max_workers=12, timeout=10):
    """Download multiple images in parallel with progress tracking"""
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Limit workers based on CPU cores and memory
    num_workers = min(max_workers, len(image_links), multiprocessing.cpu_count())

    print(f"Downloading {len(image_links)} images using {num_workers} workers...")

    download_func = partial(download_image, savefolder=download_folder, timeout=timeout)

    results = []
    with multiprocessing.Pool(num_workers) as pool:
        # Use imap_unordered for better performance with variable URL speeds
        for result in tqdm(pool.imap_unordered(download_func, image_links),
                          total=len(image_links), desc="Downloading images"):
            results.append(result)

    successful = sum(1 for r in results if r is not None)
    print(f"Downloaded {successful}/{len(image_links)} images successfully")

    return results