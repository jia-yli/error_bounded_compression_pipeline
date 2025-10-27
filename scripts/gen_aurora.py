import os
import re
import time
import torch
import shutil
import itertools
import fire

import xarray as xr
import numpy as np
import pandas as pd
import multiprocessing as mp

from error_bounded_compression_pipeline.compression import ErrorBoundedCompressionPipeline

import warnings
# warnings.filterwarnings("ignore")

def gen_ds_6h(ds):
  ds_6h = ds.sel(valid_time=ds.valid_time[::6])
  return ds_6h

def run_compression_pipeline(
  era5_path, output_path, year, month, variable, ratio, checkpoint_path1, checkpoint_path2,
):
  warnings.filterwarnings("ignore")
  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
  torch.use_deterministic_algorithms(True)
  current_proc_name = mp.current_process().name # "SpawnPoolWorker-{idx}"
  num_gpus = torch.cuda.device_count()
  try:
    worker_idx = int(current_proc_name.split('-')[-1]) % num_gpus
  except:
    worker_idx = 0
  print(f'Worker {current_proc_name} using GPU {worker_idx} for {variable} {year}-{month}')
  # Args
  if not checkpoint_path2:
    checkpoint_path2 = checkpoint_path1
  if isinstance(variable, tuple):
    reanalysis_file = os.path.join(era5_path, f'pressure_level/reanalysis/{year}/{month}/{variable[0]}/{variable[1]}.nc')
    interpolated_ensemble_spread_file = os.path.join(era5_path, f'pressure_level/interpolated_ensemble_spread/{year}/{month}/{variable[0]}/{variable[1]}.nc')
    output_file = os.path.join(output_path, f'pressure_level/reanalysis/{year}/{month}/{variable[0]}/{variable[1]}.nc')
    variable_str = f"{variable[1]}_{variable[0]}hPa"
  else:
    reanalysis_file = os.path.join(era5_path, f'single_level/reanalysis/{year}/{month}/{variable}.nc')
    interpolated_ensemble_spread_file = os.path.join(era5_path, f'single_level/interpolated_ensemble_spread/{year}/{month}/{variable}.nc')
    output_file = os.path.join(output_path, f'single_level/reanalysis/{year}/{month}/{variable}.nc')
    variable_str = variable
  if not os.path.exists(reanalysis_file):
    print(f"{variable} reanalysis file not exists")
    return
  
  reanalysis_dataset = xr.open_dataset(reanalysis_file)
  reanalysis_dataset = gen_ds_6h(reanalysis_dataset)
  if not os.path.exists(interpolated_ensemble_spread_file):
    print(f"{variable} interpolated ensemble spread file not exists")
    results = {
      'variable': variable,
      'year': year,
      'month': month,
      'ratio' : ratio, 
      'check_passed': 'file_not_exists',
    }
  else:
    interpolated_ensemble_spread_dataset = xr.open_dataset(interpolated_ensemble_spread_file)
    interpolated_ensemble_spread_dataset = gen_ds_6h(interpolated_ensemble_spread_dataset)

    assert len(reanalysis_dataset.data_vars) == len(interpolated_ensemble_spread_dataset.data_vars) == 1
    assert list(reanalysis_dataset.data_vars) == list(interpolated_ensemble_spread_dataset.data_vars)
    var = list(reanalysis_dataset.data_vars)[0]
    data = reanalysis_dataset[var].values
    shape = data.shape
    interpolated_ensemble_spread = interpolated_ensemble_spread_dataset[var].values
    if ratio >= 0:
      error_bound = interpolated_ensemble_spread * ratio
    else:
      error_bound = np.full_like(interpolated_ensemble_spread, np.nan)
    error_bound[error_bound < 0] = 0

    data = data.reshape(-1, shape[-2], shape[-1])
    error_bound = error_bound.reshape(-1, shape[-2], shape[-1])

    compression_pipeline = ErrorBoundedCompressionPipeline(
      checkpoint_path1, 
      checkpoint_path2,
      device=f'cuda:{worker_idx}')

    # Run Compression Pipeline
    compression_start_time = time.time()
    compressed_bitstream, info = compression_pipeline.compress(
      data, 
      error_bound, 
      batch_size = 16,
    )
    compression_end_time = time.time()
    compression_time = compression_end_time - compression_start_time

    decompression_start_time = time.time()
    data_hat = compression_pipeline.decompress(bit_stream=compressed_bitstream)
    decompression_end_time = time.time()
    decompression_time = decompression_end_time - decompression_start_time

    data = data.reshape(shape)
    data_hat = data_hat.reshape(shape)
    error_bound = error_bound.reshape(shape)

    reanalysis_dataset[var] = (reanalysis_dataset[var].dims, data_hat)

    # process results
    data_size_bytes = data.nbytes
    compressed_size_bytes = len(compressed_bitstream)
    compression_ratio = data_size_bytes/compressed_size_bytes
    compression_bandwidth = data_size_bytes/1e6/compression_time
    decompression_bandwidth = data_size_bytes/1e6/decompression_time
  
    # check
    nan_match = (np.isnan(data) == np.isnan(data_hat)).all()
    exclude_mask_d = np.isnan(data)
    exclude_mask_e = np.isnan(error_bound)
    exclude_mask = exclude_mask_d | exclude_mask_e
    data_match = (np.abs(data - data_hat) <= error_bound)[~exclude_mask].all()
    # assert nan_match and data_match, f"{variable} {year}-{month} data mismatch: {nan_match=}, {data_match=}"
    check_passed = nan_match and data_match
    compression_error = np.abs(data - data_hat)[~exclude_mask_d]
    max_error = compression_error.max()
    mse = np.mean(compression_error**2)
    data_range = data[~exclude_mask_d].max() - data[~exclude_mask_d].min()
    psnr = 10*np.log10(data_range**2/(mse + 1e-18))
    portion_inside = (np.abs(data - data_hat) <= error_bound)[~exclude_mask].sum() / (~exclude_mask).sum()

    data_min = data[~exclude_mask_d].min()
    data_max = data[~exclude_mask_d].max()
    data_nan_ratio = exclude_mask_d.sum()/exclude_mask_d.size

    error_bound_nan_ratio = exclude_mask_e.sum()/error_bound.size
    if error_bound_nan_ratio < 1:
      error_bound_min = error_bound[~exclude_mask_e].min()
      error_bound_max = error_bound[~exclude_mask_e].max()
      error_bound_zero_ratio = (error_bound == 0).sum()/error_bound.size
      error_scale = (error_bound_max - error_bound_min) / (data_max - data_min)
    else:
      error_bound_min = 0
      error_bound_max = 0
      error_bound_zero_ratio = 0
      error_scale = 0

    results = {
      'variable': variable_str,
      'year': year,
      'month': month,
      'ratio' : ratio, 
      'data_size_bytes' : data_size_bytes,
      'compressed_size_bytes' : compressed_size_bytes,
      'compression_time' : compression_time,
      'decompression_time' : decompression_time,
      'compression_ratio' : compression_ratio,
      'compression_time' : compression_time,
      'compression_bandwidth': compression_bandwidth,
      'decompression_bandwidth': decompression_bandwidth,
      'check_passed': check_passed,
      'max_error': max_error,
      'psnr': psnr,
      'portion_inside': portion_inside,
      'data_min': data_min,
      'data_max': data_max,
      'data_nan_ratio': data_nan_ratio,
      'error_bound_min': error_bound_min,
      'error_bound_max': error_bound_max,
      'error_bound_nan_ratio': error_bound_nan_ratio,
      'error_bound_zero_ratio': error_bound_zero_ratio,
      'error_scale': error_scale,
      **info,
    }

  os.makedirs(os.path.dirname(output_file), exist_ok=True)
  reanalysis_dataset.to_netcdf(output_file)
  return results

def main(ratio):
  print(f"Run start, ratio = {ratio}")
  static_variables = [
    "geopotential",
    "land_sea_mask",
    "soil_type",
  ]
  single_level_variables = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
  ]
  pressure_level_variables = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
  ]
  pressure_levels = [
    "50",
    "100",
    "150",
    "200",
    "250",
    "300",
    "400",
    "500",
    "600",
    "700",
    "850",
    "925",
    "1000",
  ]
  year_lst = [2024]
  month_lst = [12]

  era5_path = f'/capstor/scratch/cscs/ljiayong/datasets/ERA5_large'
  output_path = f'/capstor/scratch/cscs/ljiayong/workspace/ERA5_compressed_{ratio}'
  os.makedirs(output_path, exist_ok = True)

  checkpoint_path1 = '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_128_lambda_0.05.pth.tar'
  checkpoint_path2 = '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_128_lambda_0.05.pth.tar'

  num_gpus = torch.cuda.device_count()
  # num_gpus = 1
  ctx = mp.get_context('spawn')
  pool = ctx.Pool(processes=num_gpus)
  results = []
  func = run_compression_pipeline
  for year in year_lst:
    for month in month_lst:
      for static_variable in static_variables:
        args = (era5_path, output_path, year, month, static_variable, ratio, checkpoint_path1, checkpoint_path2)
        if num_gpus > 1:
          result = pool.apply_async(func, args = args)
        else:
          result = func(*args)
        results.append(result)
      
      for single_level_variable in single_level_variables:
        args = (era5_path, output_path, year, month, single_level_variable, ratio, checkpoint_path1, checkpoint_path2)
        if num_gpus > 1:
          result = pool.apply_async(func, args = args)
        else:
          result = func(*args)
        results.append(result)
      
      for pressure_level_variable in pressure_level_variables:
        for pressure_level in pressure_levels:
          args = (era5_path, output_path, year, month, (pressure_level, pressure_level_variable), ratio, checkpoint_path1, checkpoint_path2)
          if num_gpus > 1:
            result = pool.apply_async(func, args = args)
          else:
            result = func(*args)
          results.append(result)
  
  pool.close()
  for idx in range(len(results)):
    if num_gpus > 1:
      results[idx] = results[idx].get()

    results_df = pd.DataFrame(results[:idx+1])
    results_df.to_csv(f'./gen_aurora_results_{ratio}.csv', index=False)
  pool.join()

if __name__ == '__main__':
  fire.Fire(main)