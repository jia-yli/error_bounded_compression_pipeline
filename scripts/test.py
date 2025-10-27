import os
import re
import time
import torch
import itertools
import fire

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
  reanalysis_file = os.path.join(era5_path, f'single_level/reanalysis/{year}/{month}/{variable}.nc')
  interpolated_ensemble_spread_file = os.path.join(era5_path, f'single_level/interpolated_ensemble_spread/{year}/{month}/{variable}.nc')
  if not (os.path.exists(reanalysis_file) and os.path.exists(interpolated_ensemble_spread_file)):
    return {
      'variable': variable,
      'year': year,
      'month': month,
      'ebcc_pointwise_max_error_ratio' : ebcc_pointwise_max_error_ratio, 
      'check_passed': 'file_not_exists',
    }
  output_file = os.path.join(output_path, f'single_level/reanalysis/{year}/{month}/{variable}.compressed')

  # Extract data and error bound in np array format
  reanalysis_dataset = xr.open_dataset(reanalysis_file)
  interpolated_ensemble_spread_dataset = xr.open_dataset(interpolated_ensemble_spread_file)
  assert len(reanalysis_dataset.data_vars) == 1
  assert len(interpolated_ensemble_spread_dataset.data_vars) == 1
  assert list(reanalysis_dataset.data_vars)[0] == list(interpolated_ensemble_spread_dataset.data_vars)[0]
  data = reanalysis_dataset[list(reanalysis_dataset.data_vars)[0]].values
  interpolated_ensemble_spread = interpolated_ensemble_spread_dataset[list(interpolated_ensemble_spread_dataset.data_vars)[0]].values
  if ebcc_pointwise_max_error_ratio >= 0:
    error_bound = interpolated_ensemble_spread * ebcc_pointwise_max_error_ratio
  else:
    error_bound = np.full_like(interpolated_ensemble_spread, np.nan)
  error_bound[error_bound < 0] = 0

  # Run
  steps = 24*3
  data = data[0:steps]
  error_bound = error_bound[0:steps]

  _error_bound=error_bound.copy()
  # print("[WARNING] in MSE Mode")
  # error_mse = np.sqrt(np.nanmean(_error_bound**2))
  # _error_bound[~np.isnan(_error_bound)] = error_mse
  # if np.isnan(data).any():
  #   return {
  #     'variable': variable,
  #     'year': year,
  #     'month': month,
  #     # 'ebcc_pointwise_max_error_ratio' : np.nan, 
  #     # 'compression_ratio' : np.nan,
  #     # 'compression_time' : np.nan,
  #     # 'compression_bandwidth': np.nan,
  #     # 'decompression_bandwidth': np.nan,
  #   }

  compression_pipeline = ErrorBoundedCompressionPipeline(
    checkpoint_path1, 
    checkpoint_path2,
    device=f'cuda:{worker_idx}')

  # Run Compression Pipeline
  compression_start_time = time.time()
  compressed_bitstream, info = compression_pipeline.compress(
    data, 
    _error_bound, 
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

  # check
  nan_match = (np.isnan(data) == np.isnan(data_hat)).all()
  exclude_mask_d = np.isnan(data)
  exclude_mask_e = np.isnan(_error_bound)
  exclude_mask = exclude_mask_d | exclude_mask_e
  data_match = (np.abs(data - data_hat) <= _error_bound)[~exclude_mask].all()
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

  # import pdb;pdb.set_trace()
  # fail_mask = (np.abs(data - data_hat) >  error_bound)
  # error_bound[fail_mask]

  results = {
    'variable': variable,
    'year': year,
    'month': month,
    'ebcc_pointwise_max_error_ratio' : ebcc_pointwise_max_error_ratio, 
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
  return results

def main(ratio=1):
  print(f"Run start, ratio = {ratio}")
  variable_lst = [
    "2m_dewpoint_temperature",
    "2m_temperature",
    "ice_temperature_layer_1",
    "ice_temperature_layer_2",
    "ice_temperature_layer_3",
    "ice_temperature_layer_4",
    "maximum_2m_temperature_since_previous_post_processing",
    "mean_sea_level_pressure",
    "minimum_2m_temperature_since_previous_post_processing",
    "sea_surface_temperature",
    "skin_temperature",
    "surface_pressure",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "10m_u_component_of_neutral_wind",
    "10m_u_component_of_wind",
    "10m_v_component_of_neutral_wind",
    "10m_v_component_of_wind",
    "10m_wind_gust_since_previous_post_processing",
    "air_density_over_the_oceans",
    "angle_of_sub_gridscale_orography",
    "anisotropy_of_sub_gridscale_orography",
    "benjamin_feir_index",
    "boundary_layer_dissipation",
    "boundary_layer_height",
    "charnock",
    "clear_sky_direct_solar_radiation_at_surface",
    "cloud_base_height",
    "coefficient_of_drag_with_waves",
    "convective_available_potential_energy",
    "convective_inhibition",
    "convective_precipitation",
    "convective_rain_rate",
    "convective_snowfall",
    "convective_snowfall_rate_water_equivalent",
    "downward_uv_radiation_at_the_surface",
    "duct_base_height",
    "eastward_gravity_wave_surface_stress",
    "eastward_turbulent_surface_stress",
    "evaporation",
    "forecast_albedo",
    "forecast_logarithm_of_surface_roughness_for_heat",
    "forecast_surface_roughness",
    "free_convective_velocity_over_the_oceans",
    "friction_velocity",
    "geopotential",
    "gravity_wave_dissipation",
    "high_cloud_cover",
    "high_vegetation_cover",
    "instantaneous_10m_wind_gust",
    "instantaneous_eastward_turbulent_surface_stress",
    "instantaneous_large_scale_surface_precipitation_fraction",
    "instantaneous_moisture_flux",
    "instantaneous_northward_turbulent_surface_stress",
    "instantaneous_surface_sensible_heat_flux",
    "k_index",
    "lake_bottom_temperature",
    "lake_cover",
    "lake_depth",
    "lake_ice_depth",
    "lake_ice_temperature",
    "lake_mix_layer_depth",
    "lake_mix_layer_temperature",
    "lake_shape_factor",
    "lake_total_layer_temperature",
    "land_sea_mask",
    "large_scale_precipitation",
    "large_scale_precipitation_fraction",
    "large_scale_rain_rate",
    "large_scale_snowfall",
    "large_scale_snowfall_rate_water_equivalent",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
    "low_cloud_cover",
    "low_vegetation_cover",
    "maximum_individual_wave_height",
    "maximum_total_precipitation_rate_since_previous_post_processing",
    "mean_boundary_layer_dissipation",
    "mean_convective_precipitation_rate",
    "mean_convective_snowfall_rate",
    "mean_direction_of_total_swell",
    "mean_direction_of_wind_waves",
    "mean_eastward_gravity_wave_surface_stress",
    "mean_eastward_turbulent_surface_stress",
    "mean_evaporation_rate",
    "mean_gravity_wave_dissipation",
    "mean_large_scale_precipitation_fraction",
    "mean_large_scale_precipitation_rate",
    "mean_large_scale_snowfall_rate",
    "mean_northward_gravity_wave_surface_stress",
    "mean_northward_turbulent_surface_stress",
    "mean_period_of_total_swell",
    "mean_period_of_wind_waves",
    "mean_potential_evaporation_rate",
    "mean_runoff_rate",
    "mean_snow_evaporation_rate",
    "mean_snowfall_rate",
    "mean_snowmelt_rate",
    "mean_square_slope_of_waves",
    "mean_sub_surface_runoff_rate",
    "mean_surface_direct_short_wave_radiation_flux",
    "mean_surface_direct_short_wave_radiation_flux_clear_sky",
    "mean_surface_downward_long_wave_radiation_flux",
    "mean_surface_downward_long_wave_radiation_flux_clear_sky",
    "mean_surface_downward_short_wave_radiation_flux",
    "mean_surface_downward_short_wave_radiation_flux_clear_sky",
    "mean_surface_downward_uv_radiation_flux",
    "mean_surface_latent_heat_flux",
    "mean_surface_net_long_wave_radiation_flux",
    "mean_surface_net_long_wave_radiation_flux_clear_sky",
    "mean_surface_net_short_wave_radiation_flux",
    "mean_surface_net_short_wave_radiation_flux_clear_sky",
    "mean_surface_runoff_rate",
    "mean_surface_sensible_heat_flux",
    "mean_top_downward_short_wave_radiation_flux",
    "mean_top_net_long_wave_radiation_flux",
    "mean_top_net_long_wave_radiation_flux_clear_sky",
    "mean_top_net_short_wave_radiation_flux",
    "mean_top_net_short_wave_radiation_flux_clear_sky",
    "mean_total_precipitation_rate",
    "mean_vertical_gradient_of_refractivity_inside_trapping_layer",
    "mean_vertically_integrated_moisture_divergence",
    "mean_wave_direction",
    "mean_wave_direction_of_first_swell_partition",
    "mean_wave_direction_of_second_swell_partition",
    "mean_wave_direction_of_third_swell_partition",
    "mean_wave_period",
    "mean_wave_period_based_on_first_moment",
    "mean_wave_period_based_on_first_moment_for_swell",
    "mean_wave_period_based_on_first_moment_for_wind_waves",
    "mean_wave_period_based_on_second_moment_for_swell",
    "mean_wave_period_based_on_second_moment_for_wind_waves",
    "mean_wave_period_of_first_swell_partition",
    "mean_wave_period_of_second_swell_partition",
    "mean_wave_period_of_third_swell_partition",
    "mean_zero_crossing_wave_period",
    "medium_cloud_cover",
    "minimum_total_precipitation_rate_since_previous_post_processing",
    "minimum_vertical_gradient_of_refractivity_inside_trapping_layer",
    "model_bathymetry",
    "near_ir_albedo_for_diffuse_radiation",
    "near_ir_albedo_for_direct_radiation",
    "normalized_energy_flux_into_ocean",
    "normalized_energy_flux_into_waves",
    "normalized_stress_into_ocean",
    "northward_gravity_wave_surface_stress",
    "northward_turbulent_surface_stress",
    "ocean_surface_stress_equivalent_10m_neutral_wind_direction",
    "ocean_surface_stress_equivalent_10m_neutral_wind_speed",
    "peak_wave_period",
    "period_corresponding_to_maximum_individual_wave_height",
    "potential_evaporation",
    "precipitation_type",
    "runoff",
    "sea_ice_cover",
    "significant_height_of_combined_wind_waves_and_swell",
    "significant_height_of_total_swell",
    "significant_height_of_wind_waves",
    "significant_wave_height_of_first_swell_partition",
    "significant_wave_height_of_second_swell_partition",
    "significant_wave_height_of_third_swell_partition",
    "skin_reservoir_content",
    "slope_of_sub_gridscale_orography",
    "snow_albedo",
    "snow_density",
    "snow_depth",
    "snow_evaporation",
    "snowfall",
    "snowmelt",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "soil_type",
    "standard_deviation_of_filtered_subgrid_orography",
    "standard_deviation_of_orography",
    "sub_surface_runoff",
    "surface_latent_heat_flux",
    "surface_net_solar_radiation",
    "surface_net_solar_radiation_clear_sky",
    "surface_net_thermal_radiation",
    "surface_net_thermal_radiation_clear_sky",
    "surface_runoff",
    "surface_sensible_heat_flux",
    "surface_solar_radiation_downward_clear_sky",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downward_clear_sky",
    "surface_thermal_radiation_downwards",
    "temperature_of_snow_layer",
    "toa_incident_solar_radiation",
    "top_net_solar_radiation",
    "top_net_solar_radiation_clear_sky",
    "top_net_thermal_radiation",
    "top_net_thermal_radiation_clear_sky",
    "total_cloud_cover",
    "total_column_cloud_ice_water",
    "total_column_cloud_liquid_water",
    "total_column_ozone",
    "total_column_rain_water",
    "total_column_snow_water",
    "total_column_supercooled_liquid_water",
    "total_column_water",
    "total_column_water_vapour",
    "total_precipitation",
    "total_sky_direct_solar_radiation_at_surface",
    "total_totals_index",
    "trapping_layer_base_height",
    "trapping_layer_top_height",
    "type_of_high_vegetation",
    "type_of_low_vegetation",
    "u_component_stokes_drift",
    "uv_visible_albedo_for_diffuse_radiation",
    "uv_visible_albedo_for_direct_radiation",
    "v_component_stokes_drift",
    "vertical_integral_of_divergence_of_cloud_frozen_water_flux",
    "vertical_integral_of_divergence_of_cloud_liquid_water_flux",
    "vertical_integral_of_divergence_of_geopotential_flux",
    "vertical_integral_of_divergence_of_kinetic_energy_flux",
    "vertical_integral_of_divergence_of_mass_flux",
    "vertical_integral_of_divergence_of_moisture_flux",
    "vertical_integral_of_divergence_of_ozone_flux",
    "vertical_integral_of_divergence_of_thermal_energy_flux",
    "vertical_integral_of_divergence_of_total_energy_flux",
    "vertical_integral_of_eastward_cloud_frozen_water_flux",
    "vertical_integral_of_eastward_cloud_liquid_water_flux",
    "vertical_integral_of_eastward_geopotential_flux",
    "vertical_integral_of_eastward_heat_flux",
    "vertical_integral_of_eastward_kinetic_energy_flux",
    "vertical_integral_of_eastward_mass_flux",
    "vertical_integral_of_eastward_ozone_flux",
    "vertical_integral_of_eastward_total_energy_flux",
    "vertical_integral_of_eastward_water_vapour_flux",
    "vertical_integral_of_energy_conversion",
    "vertical_integral_of_kinetic_energy",
    "vertical_integral_of_mass_of_atmosphere",
    "vertical_integral_of_mass_tendency",
    "vertical_integral_of_northward_cloud_frozen_water_flux",
    "vertical_integral_of_northward_cloud_liquid_water_flux",
    "vertical_integral_of_northward_geopotential_flux",
    "vertical_integral_of_northward_heat_flux",
    "vertical_integral_of_northward_kinetic_energy_flux",
    "vertical_integral_of_northward_mass_flux",
    "vertical_integral_of_northward_ozone_flux",
    "vertical_integral_of_northward_total_energy_flux",
    "vertical_integral_of_northward_water_vapour_flux",
    "vertical_integral_of_potential_and_internal_energy",
    "vertical_integral_of_potential_internal_and_latent_energy",
    "vertical_integral_of_temperature",
    "vertical_integral_of_thermal_energy",
    "vertical_integral_of_total_energy",
    "vertically_integrated_moisture_divergence",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "wave_spectral_directional_width",
    "wave_spectral_directional_width_for_swell",
    "wave_spectral_directional_width_for_wind_waves",
    "wave_spectral_kurtosis",
    "wave_spectral_peakedness",
    "wave_spectral_skewness",
    "zero_degree_level"
  ]
  year_lst = [2024]
  month_lst = [12]

  era5_path = f'/capstor/scratch/cscs/ljiayong/datasets/ERA5_large'
  output_path = f'/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/compression_results_tmp_{ratio}'
  os.makedirs(output_path, exist_ok = True)

  pointwise_max_error_ratio_lst = [ratio]

  checkpoint_path1 = '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_128_lambda_0.05.pth.tar'
  checkpoint_path2 = '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_128_lambda_0.05.pth.tar'

  param_combinations = list(itertools.product(
    variable_lst, year_lst, month_lst,
    [era5_path], [output_path], pointwise_max_error_ratio_lst,
    [checkpoint_path1], [checkpoint_path2],
  ))

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
    results_df.to_csv(f'./compression_results_{ratio}.csv', index=False)
  
  pool.join()

if __name__ == '__main__':
  fire.Fire(main)