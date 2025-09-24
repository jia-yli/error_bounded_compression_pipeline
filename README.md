# error_bounded_compression_pipeline

## Get Started
1. Download Published Checkpoint From https://github.com/jmliu206/LIC_TCM
2. Setup Env:

```
# Torch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas compressai
pip install netCDF4 xarray # if needed for your dataset

# install repo
pip install -e .
```

3.  Run Compression and Decompression Pipeline, Refer to `scripts/test.py`
```
from error_bounded_compression_pipeline.compression import ErrorBoundedCompressionPipeline
compressed_bitstream, info = compression_pipeline.compress(
  data, 
  error_bound, 
  batch_size = 16,
  output_file=output_file,
)
data_hat = compression_pipeline.decompress(file_path=output_file)
```