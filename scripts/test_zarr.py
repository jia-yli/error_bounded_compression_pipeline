import os
import time
import numpy as np
import zarr
import xarray as xr
from error_bounded_compression_pipeline.compression import ErrorBounded2DCodec
from error_bounded_compression_pipeline.compression import ErrorBoundedCompressionPipelineFullGPU

# configs
variable = "10m_u_component_of_wind"
year = 2024
month = 12
era5_path = f'/capstor/scratch/cscs/ljiayong/datasets/ERA5_large'
output_path = f'/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/zarr_test'
os.makedirs(output_path, exist_ok = True)
pointwise_max_error_ratio = 1
checkpoint_path1 = '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_128_lambda_0.05.pth.tar'
checkpoint_path2 = '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_128_lambda_0.05.pth.tar'

# run
reanalysis_file = os.path.join(era5_path, f'single_level/reanalysis/{year}/{month}/{variable}.nc')
interpolated_ensemble_spread_file = os.path.join(era5_path, f'single_level/interpolated_ensemble_spread/{year}/{month}/{variable}.nc')

# Extract data and error bound in np array format
reanalysis_dataset = xr.open_dataset(reanalysis_file)
interpolated_ensemble_spread_dataset = xr.open_dataset(interpolated_ensemble_spread_file)
assert len(reanalysis_dataset.data_vars) == 1
assert len(interpolated_ensemble_spread_dataset.data_vars) == 1
assert list(reanalysis_dataset.data_vars)[0] == list(interpolated_ensemble_spread_dataset.data_vars)[0]
data = reanalysis_dataset[list(reanalysis_dataset.data_vars)[0]].values
interpolated_ensemble_spread = interpolated_ensemble_spread_dataset[list(interpolated_ensemble_spread_dataset.data_vars)[0]].values
error_bound = interpolated_ensemble_spread * pointwise_max_error_ratio

# Run
data = data[0:16]
error_bound = error_bound[0:16]


def run_compression(method):
  if method == 'full_gpu':
    output_file = os.path.join(output_path, f'{year}_{month}_{variable}.compressed')

    compression_pipeline = ErrorBoundedCompressionPipelineFullGPU(
      checkpoint_path1, 
      checkpoint_path2,
      device=f'cuda:0'
    )

    compression_start_time = time.time()
    compressed_bitstream, info = compression_pipeline.compress(
      data, 
      error_bound, 
      batch_size = 16,
      output_file=output_file,
    )
    compression_end_time = time.time()
    compression_time = compression_end_time - compression_start_time

    decompression_start_time = time.time()
    data_hat = compression_pipeline.decompress(file_path=output_file)
    decompression_end_time = time.time()
    decompression_time = decompression_end_time - decompression_start_time

    data_size_bytes = data.nbytes
    compressed_size_bytes = len(compressed_bitstream)
    compression_ratio = data_size_bytes/compressed_size_bytes
    compression_bandwidth = data_size_bytes/1e6/compression_time
    decompression_bandwidth = data_size_bytes/1e6/decompression_time
    print(f"{data_size_bytes=}, {compressed_size_bytes=}, {compression_ratio=}")
    print(f"{compression_bandwidth=}, {decompression_bandwidth=}")
    print((np.abs(data_hat-data) <= error_bound).all())
    import pdb;pdb.set_trace()
  elif method == 'zarr':
    output_file = os.path.join(output_path, f'{year}_{month}_{variable}.zarr')

    error_bound_file = os.path.join(output_path, f'{year}_{month}_{variable}_error_bound.npy')
    np.save(error_bound_file, error_bound)

    codec = ErrorBounded2DCodec(
      error_bound_file=error_bound_file, 
      batch_size=16,
      compressor_kwargs={
        'checkpoint_path1': checkpoint_path1,
        'checkpoint_path2': checkpoint_path2,
        'device': f'cuda:0',
      }
    )

    # method 1
    zarr.array(
      data,
      chunks=data.shape,
      compressor=codec,
      store=output_file,
      overwrite=True,
    )
    # method 2
    # store = zarr.DirectoryStore(output_file)
    # root = zarr.group(store=store, overwrite=True)
    # z = root.create_dataset(
    #   "data",
    #   shape=data.shape,
    #   dtype=data.dtype,
    #   compressor=codec,
    #   chunks=data.shape,
    #   overwrite=True,
    #   order="C",
    # )
    # z[...] = data
    data_size_bytes = data.nbytes
    compressed_size_bytes = os.path.getsize(output_file+"/0.0.0")
    compression_ratio = data_size_bytes/compressed_size_bytes
    print(f"{data_size_bytes=}, {compressed_size_bytes=}, {compression_ratio=}")
    # for 64: data_size_bytes=265789440, compressed_size_bytes=16600632, compression_ratio=16.01080248029111

    data_hat = np.array(zarr.open(output_file, mode="r"))
    import pdb;pdb.set_trace()

  else:
    raise NotImplementedError


if __name__ == "__main__":
  # run_compression(method='zarr')
  run_compression(method='full_gpu')
