import os
import re
import time
import torch
import itertools

import xarray as xr
import numpy as np
import pandas as pd
import multiprocessing as mp

from error_bounded_compression_pipeline.compression import ErrorBoundedCompressionPipeline

import warnings
# warnings.filterwarnings("ignore")

def run_compression_pipeline(
  variable, year, month,
  era5_path, output_path, ebcc_pointwise_max_error_ratio,
  checkpoint_path1, checkpoint_path2,
):
  warnings.filterwarnings("ignore")
  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
  torch.use_deterministic_algorithms(True)
  current_proc_name = mp.current_process().name
  try:
    worker_idx = int(current_proc_name.split('-')[-1]) % configs.num_gpus
  except:
    worker_idx = 0
  # Args
  if not checkpoint_path2:
    checkpoint_path2 = checkpoint_path1
  reanalysis_file = os.path.join(era5_path, f'single_level/reanalysis/{year}/{month}/{variable}.nc')
  interpolated_ensemble_spread_file = os.path.join(era5_path, f'single_level/interpolated_ensemble_spread/{year}/{month}/{variable}.nc')
  output_file = os.path.join(output_path, f'single_level/reanalysis/{year}/{month}/{variable}.compressed')

  # Extract data and error bound in np array format
  reanalysis_dataset = xr.open_dataset(reanalysis_file)
  interpolated_ensemble_spread_dataset = xr.open_dataset(interpolated_ensemble_spread_file)
  assert len(reanalysis_dataset.data_vars) == 1
  assert len(interpolated_ensemble_spread_dataset.data_vars) == 1
  assert list(reanalysis_dataset.data_vars)[0] == list(interpolated_ensemble_spread_dataset.data_vars)[0]
  data = reanalysis_dataset[list(reanalysis_dataset.data_vars)[0]].values
  interpolated_ensemble_spread = interpolated_ensemble_spread_dataset[list(interpolated_ensemble_spread_dataset.data_vars)[0]].values
  error_bound = interpolated_ensemble_spread * ebcc_pointwise_max_error_ratio

  # Run
  data = data[0:16]
  error_bound = error_bound[0:16]
  if np.isnan(data).any():
    return {
      'variable': variable,
      'year': year,
      'month': month,
      # 'ebcc_pointwise_max_error_ratio' : np.nan, 
      # 'compression_ratio' : np.nan,
      # 'compression_time' : np.nan,
      # 'compression_bandwidth': np.nan,
      # 'decompression_bandwidth': np.nan,
    }

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
    output_file=output_file,
  )
  compression_end_time = time.time()
  compression_time = compression_end_time - compression_start_time

  # Run Decompression Pipeline
  decompression_start_time = time.time()
  data_hat = compression_pipeline.decompress(file_path=output_file)
  decompression_end_time = time.time()
  decompression_time = decompression_end_time - decompression_start_time

  # process results
  data_size_bytes = data.nbytes
  compressed_size_bytes = len(compressed_bitstream)
  compression_ratio = data_size_bytes/compressed_size_bytes
  compression_bandwidth = data_size_bytes/1e6/compression_time
  decompression_bandwidth = data_size_bytes/1e6/decompression_time

  # import pdb;pdb.set_trace()

  results = {
    'variable': variable,
    'year': year,
    'month': month,
    'ebcc_pointwise_max_error_ratio' : ebcc_pointwise_max_error_ratio, 
    'data_size_bytes' : data_size_bytes,
    'compression_time' : compression_time,
    'decompression_time' : decompression_time,
    'compression_ratio' : compression_ratio,
    'compression_time' : compression_time,
    'compression_bandwidth': compression_bandwidth,
    'decompression_bandwidth': decompression_bandwidth,
    **info,
  }
  return results

if __name__ == '__main__':
  variable_lst = [
    # "100m_u_component_of_wind",
    # "100m_v_component_of_wind",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    # "2m_dewpoint_temperature",
    # "2m_temperature",
    # "ice_temperature_layer_1",
    # "ice_temperature_layer_2",
    # "ice_temperature_layer_3",
    # "ice_temperature_layer_4",
    # "maximum_2m_temperature_since_previous_post_processing",
    # "mean_sea_level_pressure",
    # "minimum_2m_temperature_since_previous_post_processing",
    # "sea_surface_temperature",
    # "skin_temperature",
    # "surface_pressure",
    # "total_precipitation",
  ]
  year_lst = [2024]
  month_lst = [12]

  era5_path = f'/capstor/scratch/cscs/ljiayong/datasets/ERA5_large'
  output_path = f'/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/ERA5_compressed'
  os.makedirs(output_path, exist_ok = True)

  pointwise_max_error_ratio_lst = [0.1, 0.5, 1]
  pointwise_max_error_ratio_lst = [1]

  checkpoint_path1 = '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_128_lambda_0.05.pth.tar'
  checkpoint_path2 = '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_128_lambda_0.05.pth.tar'

  param_combinations = list(itertools.product(
    variable_lst, year_lst, month_lst,
    [era5_path], [output_path], pointwise_max_error_ratio_lst,
    [checkpoint_path1], [checkpoint_path2],
  ))

  # loop
  results = []
  for params in param_combinations:
    results.append(run_compression_pipeline(*params))
  
    # Convert results to a structured DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'./error_bounded_compression_pipeline/compression_results.csv', index=False)

  num_gpus = torch.cuda.device_count()
  # num_gpus = 1
  ctx = mp.get_context('spawn')
  pool = ctx.Pool(processes=num_gpus)
  results = []
  for params in param_combinations:
    if num_gpus > 1:
      result = pool.apply_async(run_compression_pipeline, args = params)
    else:
      result = run_compression_pipeline(*params)
    results.append(result)
  
  pool.close()

  for idx in range(len(results)):
    if num_gpus > 1:
      results[idx] = results[idx].get()

    results_df = pd.DataFrame(results[:idx+1])
    results_df.to_csv(f'./compression_results.csv', index=False)
  
  pool.join()