# error_bounded_compression_pipeline

## Get Started
1. Download Published Checkpoint From https://github.com/jmliu206/LIC_TCM
2. Setup Env:

```
# Torch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas compressai
pip install netCDF4 xarray # if needed for your dataset
pip install zarr # if use zarr interface

# install repo
pip install -e .
```

3.  Run Compression and Decompression Pipeline, Refer to `scripts/test.py`

- Setup
```
from error_bounded_compression_pipeline.compression import ErrorBoundedCompressionPipeline
data = ...
error_bound = ...
output_file = "example.compress"
compression_pipeline = ErrorBoundedCompressionPipeline(
  checkpoint_path1, 
  checkpoint_path2,
  device=f'cuda:0',
)
```

- Run Compressioin and Decompression
```
compressed_bitstream, info = compression_pipeline.compress(
  data, 
  error_bound, 
  batch_size = 16,
  output_file=output_file,
)
data_hat = compression_pipeline.decompress(file_path=output_file)
```

4. Run Compression and Decompression Pipeline with Zarr

- Setup
```
from error_bounded_compression_pipeline.compression import ErrorBounded2DCodec
data = ...
error_bound = ...
output_file = "example.zarr"
error_bound_file = "error_bound.npy"
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
```

- Run Compressioin and Decompression
```
zarr.array(
  data,
  chunks=data.shape,
  compressor=codec,
  store=output_file,
  overwrite=True,
)
data_hat = np.array(zarr.open(output_file, mode="r"))
```