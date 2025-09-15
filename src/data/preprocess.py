import os
import shutil
import glob
import rasterio
import numpy as np
import json
from rasterio.merge import merge

import richdem as rd
from tqdm import tqdm

from config import project_config
from src.utils.log_config import setup_log

# ==============================================================================
# SECTION 1: CORE PREPROCESSING LOGIC (Your Refactored Functions)
# ==============================================================================

def create_padding_mask(output_dir, relative_path, data_type, original_height, original_width, target_height, target_width, transform, crs):
    """Creates and saves a padding mask for a single patch."""
    mask_dir = os.path.join(output_dir, 'mask', 'padding_mask', data_type, os.path.dirname(relative_path))
    os.makedirs(mask_dir, exist_ok=True)
    mask_filename = os.path.basename(relative_path)
    mask_path = os.path.join(mask_dir, mask_filename)
    mask_data = np.zeros((target_height, target_width), dtype=np.uint8)
    mask_data[:original_height, :original_width] = 1
    profile = {'driver': 'GTiff', 'height': target_height, 'width': target_width, 'count': 1, 'dtype': rasterio.uint8, 'crs': crs, 'transform': transform}
    with rasterio.open(mask_path, 'w', **profile) as dst:
        dst.write(mask_data, 1)

def pad_and_mask_all_patches(raw_dir, interim_dir, target_size):
    """Pads all .tif patches to a target size and creates padding masks."""
    log = setup_log(__name__, project_config.DATA_PIPELINE_LOG, console_output=False) # printing logs to console for main pipeline
    log.info("--- Step 1: Padding patches and creating padding masks ---")
    data_types = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    for data_type in data_types:
        # if data_type=='dem': # skip dem we did it while pre-computing features. Just before this function call
        #     continue
        source_data_path = os.path.join(raw_dir, data_type)
        output_data_path = os.path.join(interim_dir, data_type)
        patch_files = glob.glob(os.path.join(source_data_path, '**', '*.tif'), recursive=True)
        log.info(f"  Processing {len(patch_files)} patches for '{data_type}'...")
        padded_pixel_value = -99999  # unique for padding

        for patch_path in patch_files:
            try:
                with rasterio.open(patch_path) as src:
                    data, profile = src.read().astype(np.float32), src.profile.copy()
                    nodata_val = src.nodata if src.nodata is not None else np.nan
                    original_height, original_width = src.height, src.width

                    # Initialize with PAD_VAL
                    padded_data = np.full((src.count, target_size, target_size),
                                        padded_pixel_value, dtype=np.float32)

                    # Insert original data
                    padded_data[:, :original_height, :original_width] = data

                    # Update profile
                    profile.update({
                        'height': target_size,
                        'width': target_size,
                        'dtype': 'float32',
                        'nodata': nodata_val  # keep native nodata
                    })

                    relative_path = os.path.relpath(patch_path, source_data_path)
                    output_patch_path = os.path.join(output_data_path, relative_path)
                    os.makedirs(os.path.dirname(output_patch_path), exist_ok=True)

                    with rasterio.open(output_patch_path, 'w', **profile) as dst:
                        dst.write(padded_data)

                    # Not saving padding masks. Will just use the value -99999
                    # create_padding_mask( # Now creating padding mask is optional since we can just use the padded_data value. So do not need to load. Saving just for the sake of saving.
                    #     interim_dir, relative_path, data_type,
                    #     original_height, original_width,
                    #     target_size, target_size,
                    #     src.transform, src.crs
                    # )

            except Exception as e:
                log.info(f"    ERROR: Could not process {patch_path}: {e}")
    # Also copy all non-TIF files like metadata jsons
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if not file.lower().endswith('.tif'):
                src_file = os.path.join(root, file)
                relative_path = os.path.relpath(src_file, raw_dir)
                dst_file = os.path.join(interim_dir, relative_path)
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)


def generate_nodata_masks_for_all_patches(interim_dir):
    """Creates nodata masks for all processed .tif files."""
    log = setup_log(__name__, project_config.DATA_PIPELINE_LOG, console_output=False) # printing logs to console for main pipeline
    log.info("--- Step 2: Generating nodata masks ---")
    data_types = [d for d in os.listdir(interim_dir) if os.path.isdir(os.path.join(interim_dir, d)) and d != 'mask']
    for data_type in data_types:
        source_folder = os.path.join(interim_dir, data_type)
        patch_files = glob.glob(os.path.join(source_folder, '**', '*.tif'), recursive=True)
        log.info(f"  Processing {len(patch_files)} patches for '{data_type}'...")
        for patch_path in patch_files:
            relative_path = os.path.relpath(patch_path, source_folder)
            mask_dir = os.path.join(interim_dir, 'mask', 'nodata_mask', data_type, os.path.dirname(relative_path))
            os.makedirs(mask_dir, exist_ok=True)
            mask_path = os.path.join(mask_dir, os.path.basename(relative_path))
            with rasterio.open(patch_path) as src:
                profile, nodata_val = src.profile.copy(), src.nodata
                if nodata_val is not None:
                    mask_data = (src.read(1) != nodata_val).astype(np.uint8)
                else:
                    mask_data = np.ones((src.height, src.width), dtype=np.uint8)
                profile.update({'count': 1, 'dtype': rasterio.uint8, 'nodata': None})
                with rasterio.open(mask_path, 'w', **profile) as dst:
                    dst.write(mask_data, 1)


def create_permanent_water_masks(raw_dir, interim_dir, target_size, threshold):
    """Creates binary permanent water masks from raw occurrence data and pads them."""
    log = setup_log(__name__, project_config.DATA_PIPELINE_LOG, console_output=False) # printing logs to console for main pipeline
    log.info("--- Step 3: Generating permanent water masks ---")
    permanent_water_data_path = os.path.join(raw_dir, 'permanent_water')
    if not os.path.exists(permanent_water_data_path):
        log.info("  'permanent_water' directory not found. Skipping.")
        return
    mask_dir = os.path.join(interim_dir, 'mask', 'permanent_water_mask')
    patch_files = glob.glob(os.path.join(permanent_water_data_path, '**', '*.tif'), recursive=True)
    log.info(f"  Processing {len(patch_files)} permanent water patches...")
    for patch_path in patch_files:
        with rasterio.open(patch_path) as src:
            img, profile = src.read(1), src.profile.copy()
            original_height, original_width = src.height, src.width
            img = np.where(img < 0, np.nan, img)
            mask = np.where(img > threshold, 1, 0).astype(np.uint8)
            mask[np.isnan(img)] = 0
            padded_mask = np.full((target_size, target_size), 0, dtype=np.uint8)
            padded_mask[:original_height, :original_width] = mask
            profile.update({'dtype': 'uint8', 'count': 1, 'nodata': 0, 'height': target_size, 'width': target_size})
            relative_path = os.path.relpath(patch_path, permanent_water_data_path)
            output_mask_dir = os.path.join(mask_dir, os.path.dirname(relative_path))
            os.makedirs(output_mask_dir, exist_ok=True)
            mask_patch_path = os.path.join(output_mask_dir, os.path.basename(relative_path))
            with rasterio.open(mask_patch_path, 'w', **profile) as dst:
                dst.write(padded_mask, 1)


def _merge_and_save_mosaic(patch_paths, output_path, band_count=1):
    """
    A robust helper function to merge a list of GeoTIFFs and save the result.
    Closes file handles correctly to prevent errors.
    """
    log = setup_log(__name__, project_config.DATA_PIPELINE_LOG, console_output=False) # printing logs to console for main pipeline

    if not patch_paths:
        log.info(f"    - No patches found to create mosaic at: {output_path}")
        return

    log.info(f"    - Merging {len(patch_paths)} patches into -> {os.path.basename(output_path)}")
    
    sources = []
    try:
        # Open all source files
        for fp in patch_paths:
            sources.append(rasterio.open(fp))
        
        # Merge the datasets
        mosaic, out_trans = merge(sources)
        
        # Copy metadata from the first file and update it
        out_meta = sources[0].meta.copy()
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "count": mosaic.shape[0] # Update band count
        })
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write the mosaic to a new file
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
            
    except Exception as e:
        log.info(f"    ERROR: Could not create mosaic. Reason: {e}")
    finally:
        # IMPORTANT: Always close the file handles
        for src in sources:
            src.close()

def create_mosaics(raw_dir, interim_dir):
    """
    Creates mosaics from raw data patches and saves them to the interim directory.
    This function intelligently handles static and time-series data structures.
    """
    log = setup_log(__name__, project_config.DATA_PIPELINE_LOG, console_output=False) # printing logs to console for main pipeline

    log.info("\n--- Step 4: Creating mosaics for visualization ---")
    data_types = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

    for data_type in data_types:
        source_folder = os.path.join(raw_dir, data_type)
        mosaic_dir = os.path.join(interim_dir, data_type, 'mosaics')
        log.info(f"\n  Processing mosaics for data type: '{data_type}'")

        # --- Case 1: Static Data (DEM, Permanent Water) ---
        # Structure: .../{data_type}/{tile_name}/{patch_id}.tif
        if data_type in ['dem', 'hnd', 'permanent_water']:
        # if data_type in ['hnd1']:
            # REFACTORED: Correct glob pattern, no 'patches' folder
            patch_files = glob.glob(os.path.join(source_folder, '**', '*.tif'), recursive=True)
            
            output_mosaic_path = os.path.join(mosaic_dir, f'mosaic_{data_type}.tif')
            _merge_and_save_mosaic(patch_files, output_mosaic_path)

        # --- Case 2: Time-Series Data (Sentinel-1, Rainfall) ---
        # Structure: .../{data_type}/{tile_name}/{date}/{patch_id}.tif
        elif data_type in ['dry_sentinel1','monsoon_sentinel1', 'sentinel1', 'rainfall']:
        # elif data_type in ['sentinel1']:
            # First, find all unique date directories
            date_dirs = glob.glob(os.path.join(source_folder, '*', '*')) # Matches .../{tile_name}/{date}
            
            paths_by_date = {}
            for date_dir in date_dirs:
                # The date is the last part of the directory path
                date_str = os.path.basename(date_dir)
                if date_str not in paths_by_date:
                    paths_by_date[date_str] = []
                
                # Find all patches within this specific date directory
                patches_in_date_dir = glob.glob(os.path.join(date_dir, '*.tif'))
                paths_by_date[date_str].extend(patches_in_date_dir)

            if not paths_by_date:
                log.info(f"    - No date-based patches found for '{data_type}'.")
                continue

            # Now, create one mosaic for each date
            for date_str, patch_files in paths_by_date.items():
                output_mosaic_path = os.path.join(mosaic_dir, f'mosaic_{data_type}_{date_str}.tif')
                _merge_and_save_mosaic(patch_files, output_mosaic_path)

def compute_pad_and_mask_dem_features(source_dem_path, processed_dir, relative_path, target_size=270, nodata_fill_value=-9999):
    """
    Reads an UNPADDED DEM patch, computes features, pads the result, saves it,
    and then creates a corresponding padding mask.
    """
    try:
        # 1. Read the unpadded, raw data and get metadata
        with rasterio.open(source_dem_path) as src:
            if src.count != 1:
                print(f"WARNING: Skipping non-single-band file: {source_dem_path}")
                return
            
            profile = src.profile.copy()
            dem_array = src.read(1)
            nodata_val = src.nodata
            geotransform = src.transform.to_gdal()
            original_height, original_width = dem_array.shape
            original_transform = src.transform
            original_crs = src.crs

        # 2. Compute features on the UNPADDED data
        dem_rd = rd.rdarray(dem_array, no_data=nodata_val if nodata_val is not None else nodata_fill_value)
        dem_rd.geotransform = geotransform
        
        slope = rd.TerrainAttribute(dem_rd, attrib='slope_radians')
        aspect = rd.TerrainAttribute(dem_rd, attrib='aspect')
        curvature = rd.TerrainAttribute(dem_rd, attrib='curvature')
        sin_aspect = np.sin(np.deg2rad(aspect))
        cos_aspect = np.cos(np.deg2rad(aspect))
        
        unpadded_features = np.stack([
            dem_array, slope, sin_aspect, cos_aspect, curvature
        ]).astype(np.float32)

        # 3. Pad the multi-channel feature array
        padded_features = np.full(
            (5, target_size, target_size), 
            fill_value=nodata_fill_value, 
            dtype=np.float32
        )
        padded_features[:, :original_height, :original_width] = unpadded_features

        # 4. Update profile and save the PADDED features to the processed directory
        profile.update({
            'count': 5, 'dtype': 'float32', 'nodata': nodata_fill_value,
            'height': target_size, 'width': target_size
        })

        output_feature_path = os.path.join(processed_dir, 'dem', relative_path)
        os.makedirs(os.path.dirname(output_feature_path), exist_ok=True)
        with rasterio.open(output_feature_path, 'w', **profile) as dst:
            dst.write(padded_features)

        # ======================================================================
        # THE FIX: Call the create_padding_mask function immediately after saving
        # ======================================================================
        create_padding_mask(
            output_dir=processed_dir, # Base dir for all processed data
            relative_path=relative_path,
            data_type='dem', # Specify the data type
            original_height=original_height,
            original_width=original_width,
            target_height=target_size,
            target_width=target_size,
            transform=original_transform,
            crs=original_crs
        )
        # ======================================================================

    except Exception as e:
        print(f"ERROR processing {source_dem_path}: {e}")


# ==============================================================================
# SECTION 2: THE REUSABLE PREPROCESSING PIPELINE
# ==============================================================================

def run_preprocessing_dependent_pipeline(raw_district_dir, interim_district_dir, config):
    """
    Executes the full preprocessing pipeline for a single district.
    """
    log = setup_log(__name__, project_config.DATA_PIPELINE_LOG, console_output=False) # printing logs to console for main pipeline

    log.info(f"\n{'*'*70}")
    log.info(f"Starting preprocessing for: {raw_district_dir}")
    log.info(f"Output will be saved to: {interim_district_dir}")
    log.info(f"{'*'*70}")

    # log.info("--- Starting Pre-computation of DEM Features & Padding Mask Generation ---")
    
    # dem_patches = glob.glob(os.path.join(raw_district_dir, '**', 'dem', '**', '*.tif'), recursive=True)
    # dem_patches = [p for p in dem_patches if 'mosaics' not in p]
    
    # log.info(f"Found {len(dem_patches)} DEM patches to process.")

    # for source_path in tqdm(dem_patches, desc="Computing DEM Features"):
    #     # The relative path is key to maintaining the structure
    #     relative_path = os.path.relpath(source_path, os.path.join(raw_district_dir, 'dem'))
    #     compute_pad_and_mask_dem_features(source_path, interim_district_dir, relative_path, config["PADDED_PATCH_SIZE"])

    # log.info("\n--- Pre-computation for DEM Complete ---")
    
    """Here just pad the pathes. Do not create padding mask. Since we are assigning -999999 value to padded pixels. So we can just exclude using arr!=-999999"""
    # Step 1: Pad all patches to a consistent size and create padding masks.
    pad_and_mask_all_patches( # skip dem, we covered it above
        raw_dir=raw_district_dir,
        interim_dir=interim_district_dir,
        target_size=config["PADDED_PATCH_SIZE"]
    )


    """No need to generate and save nodata mask. We can just use nodata value by loading tif like arr!=nodata_value"""
    # Step 2: Generate nodata masks from the newly created interim (padded) data.
    # generate_nodata_masks_for_all_patches(
    #     interim_dir=interim_district_dir
    # )
    
    # # Step 3: Create binary permanent water masks from the original raw data.
    # create_permanent_water_masks(
    #     raw_dir=raw_district_dir,
    #     interim_dir=interim_district_dir,
    #     target_size=config["PADDED_PATCH_SIZE"],
    #     threshold=config["PERMANENT_WATER_THRESHOLD"]
    # )

    # Step 4: Create mosaics from the original raw patches for visualization.
    create_mosaics(
        raw_dir=raw_district_dir,
        interim_dir=interim_district_dir
    )
    
    log.info(f"\n--- Finished preprocessing for: {os.path.basename(raw_district_dir)} ---")

# ==============================================================================
# SECTION 3: MAIN ORCHESTRATOR
# ==============================================================================

def run_preprocessing_pipeline(TARGET_STATE=None, TARGET_DISTRICT=None):
    """
    Main orchestrator that reads the config, determines the scope (India/state/district),
    and runs the preprocessing pipeline accordingly.
    """
    log = setup_log(__name__, project_config.DATA_PIPELINE_LOG, console_output=False) # printing logs to console for main pipeline

    config = {
    "BASE_RAW_DIRECTORY": project_config.RAW_DATA_DIR,
    "BASE_INTERIM_DIRECTORY": project_config.INTERIM_DATA_DIR,
    "AOI_DATABASE_PATH": project_config.AOI_DATABASE_PATH,

    "TARGET_STATE": TARGET_STATE,
    "TARGET_DISTRICT": TARGET_DISTRICT,

    "PADDED_PATCH_SIZE": project_config.PADDED_PATCH_SIZE,
    "PERMANENT_WATER_THRESHOLD": project_config.PERMANENT_WATER_THRESHOLD
    }

    try:
        with open(config['AOI_DATABASE_PATH'], 'r') as f:
            aoi_database = json.load(f)
    except FileNotFoundError:
        log.info(f"FATAL ERROR: AOI database not found at '{config['AOI_DATABASE_PATH']}'.")
        return

    # Determine the list of districts to process
    target_state = config.get('TARGET_STATE')
    target_district = config.get('TARGET_DISTRICT')
    districts_to_process = []

    if not target_state or not target_state.strip():
        # Case 1: Whole India
        log.info("Scope: WHOLE INDIA. Preparing to process all states and districts.")
        for state, districts in aoi_database.items():
            for district in districts:
                districts_to_process.append({'state': state, 'district': district})
    else:
        all_districts_in_state = aoi_database.get(target_state)
        if not all_districts_in_state:
            log.info(f"FATAL ERROR: State '{target_state}' not found in AOI database.")
            return

        if not target_district or not target_district.strip():
            # Case 2: Single State
            log.info(f"Scope: STATE - {target_state}. Preparing to process all districts in this state.")
            for district in all_districts_in_state:
                districts_to_process.append({'state': target_state, 'district': district})
        else:
            # Case 3: Single District
            log.info(f"Scope: DISTRICT - {target_district}, {target_state}.")
            found_district = next((d for d in all_districts_in_state if d['district_name'] == target_district), None)
            if found_district:
                districts_to_process.append({'state': target_state, 'district': found_district})
            else:
                log.info(f"FATAL ERROR: District '{target_district}' not found in state '{target_state}'.")
                return

    if not districts_to_process:
        log.info("No districts selected for processing. Exiting.")
        return

    # Loop through the final list of districts and run the pipeline
    log.info(f"\n{'-'*70}")
    log.info(f"STARTING PREPROCESSING FOR {len(districts_to_process)} TOTAL DISTRICT(S).")
    log.info(f"{'-'*70}")

    for item in districts_to_process:
        state_name = item['state']
        district_name = item['district']['district_name']
        safe_district_name = district_name.replace(" ", "_").replace("/", "_")
        
        raw_district_path = os.path.join(config["BASE_RAW_DIRECTORY"], state_name, safe_district_name)
        interim_district_path = os.path.join(config["BASE_INTERIM_DIRECTORY"], state_name, safe_district_name)

        if not os.path.exists(raw_district_path):
            log.info(f"WARNING: Raw data directory not found for {district_name}, {state_name}. Skipping.")
            log.info(f"  (Checked path: {raw_district_path})")
            continue

        os.makedirs(interim_district_path, exist_ok=True)
        
        run_preprocessing_dependent_pipeline(raw_district_path, interim_district_path, config)

    log.info(f"\n\n{'*'*70}")
    log.info(f"PREPROCESSING BATCH COMPLETE.")
    log.info(f"{'*'*70}")