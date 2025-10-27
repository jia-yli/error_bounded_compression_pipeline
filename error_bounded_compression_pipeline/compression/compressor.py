import os
import re
import torch
import time
import itertools
import pickle
import zlib

import xarray as xr
import numpy as np
import pandas as pd

import torch.nn.functional as F

from ..models import TCM

class ErrorBoundedCompressionPipeline:
  def __init__(
    self, 
    checkpoint_path1, 
    checkpoint_path2,
    device='cuda:0'):
    self.device = device

    match1 = re.search(r'[nN]_(\d+)', checkpoint_path1)
    N1 = int(match1.group(1))
    match2 = re.search(r'[nN]_(\d+)', checkpoint_path2)
    N2 = int(match2.group(1))

    self.net1 = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=N1, M=320)
    self.net1 = self.net1.to(self.device)
    self.net1.eval()
    # load check point
    dictory = {}
    checkpoint = torch.load(checkpoint_path1, map_location=self.device, weights_only=True)
    for k, v in checkpoint["state_dict"].items():
      dictory[k.replace("module.", "")] = v
    self.net1.load_state_dict(dictory)
    self.net1.update()

    # net 2
    self.net2 = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=N2, M=320)
    self.net2 = self.net2.to(self.device)
    self.net2.eval()
    # load check point
    dictory = {}
    checkpoint = torch.load(checkpoint_path2, map_location=self.device, weights_only=True)
    for k, v in checkpoint["state_dict"].items():
      dictory[k.replace("module.", "")] = v
    self.net2.load_state_dict(dictory)
    self.net2.update()

  @staticmethod
  def get_padding(h, w, p):
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p

    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    
    padding = [padding_left, padding_right, padding_top, padding_bottom]
    return padding

  @staticmethod
  def pad(x, padding):
    x_padded = F.pad(
      x,
      padding,
      mode="constant",
      value=0,
    )
    return x_padded
  
  @staticmethod
  def crop(x, padding):
    return F.pad(
      x,
      (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

  def _compress(
    self,
    x, # [N, H, W] in numpy array
    error_bound,
    net_id,
    batch_size=1,
  ):
    # Normalize to [0, 1]
    xmin = float(x.min())
    xmax = float(x.max())
    scale = (xmax - xmin) if xmax > xmin else 1.0
    x01 = (x - xmin) / scale
    norm_info={"min": xmin, "scale": scale}

    # run comrpession
    padding_granularity = 128
    padding = self.get_padding(x.shape[-2], x.shape[-1], padding_granularity)
    num_slices = x01.shape[0]
    
    meta_data = {
      "net": net_id,
      "pad": padding,
      **norm_info,
    }
    results = []
    start_idx = 0
    while start_idx < num_slices:
      # print(f"[INFO] Compressing {start_idx}/{num_slices}")
      end_idx = min(start_idx + batch_size, num_slices)
      # shape [B, 1, H, W] -> replicate to 3 channels
      x_tensor = torch.from_numpy(x01[start_idx:end_idx]).unsqueeze(1).repeat(1, 3, 1, 1)
      x_tensor = x_tensor.to(self.device) #, dtype=torch.float32)
      x_tensor = self.pad(x_tensor, padding)

      with torch.no_grad():
        out_enc = eval(f"self.net{net_id}").compress(x_tensor)
        out_enc["shape"] = list(out_enc["shape"]) # for bitstream packet, original datatype is torch.Size
        results.append(out_enc)

      start_idx = end_idx
    
    return {
      **meta_data,
      "res": results,
    }

  def _decompress(
    self,
    nested,
  ):
    net_id = nested['net']
    padding = nested['pad']
    xmin = nested['min']
    scale = nested['scale']
    results = nested['res']

    data_hat_lst = []
    for out_enc in results:
      with torch.no_grad():
        out_dec = eval(f"self.net{net_id}").decompress(out_enc["strings"], out_enc["shape"])
        out_dec["x_hat"] = self.crop(out_dec["x_hat"], padding).mean(dim=-3)
        data_hat = out_dec["x_hat"].detach().cpu().numpy()*scale + xmin
        data_hat_lst.append(data_hat)
    data_hat = np.concatenate(data_hat_lst, axis=0)
    return data_hat

  @staticmethod
  def compress_fail_value(x, x_hat, error_bound, exclude_mask):
    error = np.abs(x - x_hat)
    fail_mask = error > error_bound
    fail_mask[exclude_mask] = False
    fail_idx = np.flatnonzero(fail_mask).astype(np.int32)
    fail_val = x.flat[fail_idx]

    # compress them
    packed_fail_mask = np.packbits(fail_mask.ravel())
    compressed_fail_mask = zlib.compress(packed_fail_mask.tobytes(), level=6)
    compressed_fail_idx = zlib.compress(fail_idx.tobytes(), level=6)
    compressed_fail_val = zlib.compress(fail_val.tobytes(), level=6)

    if len(compressed_fail_mask) <= len(compressed_fail_idx):
      # use mask
      compressed_fail_idx = None
      compressed_fail_info_size = len(compressed_fail_mask) + len(compressed_fail_val)
      compressed_fail_info = {
        "fail_mask": compressed_fail_mask,
        "fail_val": compressed_fail_val,
      }
    else:
      compressed_fail_mask = None
      compressed_fail_info_size = len(compressed_fail_idx) + len(compressed_fail_val)
      compressed_fail_info = {
        "fail_idx": compressed_fail_idx,
        "fail_val": compressed_fail_val,
      }

    return compressed_fail_info_size, fail_mask, fail_val, compressed_fail_info

  @staticmethod
  def decompress_fail_value(shape, compressed_fail_info):
    compressed_fail_mask = compressed_fail_info.get("fail_mask", None)
    compressed_fail_idx = compressed_fail_info.get("fail_idx", None)
    compressed_fail_val = compressed_fail_info["fail_val"]

    fail_val = np.frombuffer(zlib.decompress(compressed_fail_val), dtype=np.float32)
    if compressed_fail_mask:
      packed_fail_mask = np.frombuffer(zlib.decompress(compressed_fail_mask), dtype=np.uint8)
      fail_mask = np.unpackbits(packed_fail_mask)[:np.prod(shape)].reshape(shape).astype(bool)
      # fail_idx = np.flatnonzero(fail_mask).astype(np.int32)
    else:
      fail_idx = np.frombuffer(zlib.decompress(compressed_fail_idx), dtype=np.int32)
      fail_mask = np.zeros(shape, dtype=bool)
      fail_mask.flat[fail_idx] = True

    return fail_mask, fail_val
  
  @staticmethod
  def process_nan(x, run_compression, fill_by):
    nan_mask = np.isnan(x)

    # compression
    compressed_nan_info = {}
    if run_compression:
      has_nan = bool(nan_mask.any())
      compressed_nan_info['has_nan'] = has_nan
      if has_nan:
        packed_nan_mask = np.packbits(nan_mask.ravel())
        compressed_nan_mask = zlib.compress(packed_nan_mask.tobytes(), level=6)
        compressed_nan_info['compressed_nan_mask'] = compressed_nan_mask
    
    # process
    if np.all(nan_mask):
      if fill_by == 'min':
        fill_val = 0.0
      elif fill_by == 'max':
        fill_val = 1.0
      else:
        raise ValueError(f"Unsupported fill_by {fill_by} when all values are NaN")
    else:
      fill_val = eval(f"x[~nan_mask].{fill_by}()")
    x = x.copy()
    x[nan_mask] = fill_val
    return x, nan_mask, compressed_nan_info

  def compress(
    self, 
    data, 
    error_bound=None, 
    batch_size=1,
    max_residual_runs=-1,
    output_file=None,
  ):
    if not isinstance(data, np.ndarray):
      raise TypeError("arr must be a NumPy ndarray")
    if data.dtype != np.float32:
      data = data.astype(np.float32, copy=False)
    if data.ndim != 3:
      raise ValueError("arr must have exactly 3 dims; last two are [N, H, W]")
    
    # To [N, H, W]
    N, H, W = data.shape

    '''
    Step 1: handle NaN
    '''
    x, nan_mask_data, compressed_nan_info = self.process_nan(data, run_compression=True, fill_by='min')
    has_nan = compressed_nan_info['has_nan']
    if has_nan:
      compressed_nan_mask = compressed_nan_info['compressed_nan_mask']

    error_bound, nan_mask_error_bound, _ = self.process_nan(error_bound, run_compression=False, fill_by='max')

    # exclude mask = points donot care
    exclude_mask = nan_mask_data | nan_mask_error_bound

    '''
    Step 2: compression
    '''
    net_id = 1
    compression_start_time = time.time()
    compressed_x = self._compress(
      x,
      error_bound=error_bound,
      net_id=net_id,
      batch_size=batch_size,
    )
    compression_end_time = time.time()
    compression_time = compression_end_time - compression_start_time

    decompression_start_time = time.time()
    x_hat = self._decompress(compressed_x)
    decompression_end_time = time.time()
    decompression_time = decompression_end_time - decompression_start_time
    assert x_hat.dtype == np.float32
    compressed_fail_info_size, fail_mask, fail_val, compressed_fail_info = self.compress_fail_value(x, x_hat, error_bound, exclude_mask)

    '''
    Step 3: residual compression
    '''
    x_hat = x_hat
    num_residual_runs = 0
    compressed_residual_lst = []
    current_compressed_fail_info_size = compressed_fail_info_size
    current_compressed_fail_info = compressed_fail_info
    while True:
      num_residual_runs += 1 # num runs after current loop (first run not counted as residual run)
      # max residual runs
      if max_residual_runs >= 0:
        if num_residual_runs > max_residual_runs:
          num_residual_runs -= 1 # not count current run
          break

      residual = x - x_hat

      # residual compression
      net_id = 2
      compressed_residual = self._compress(
        residual,
        error_bound=error_bound,
        net_id=net_id,
        batch_size=batch_size,
      )
      residual_hat = self._decompress(compressed_residual)
      assert residual_hat.dtype == np.float32
      x_hat = x_hat + residual_hat

      # fail value
      compressed_fail_info_size, fail_mask, fail_val, compressed_fail_info = self.compress_fail_value(x, x_hat, error_bound, exclude_mask)

      compressed_residual_bitstream = pickle.dumps(compressed_residual)

      # stop condition
      # if num_residual_runs > 1: # at least run once
      if len(compressed_residual_bitstream) + compressed_fail_info_size >= current_compressed_fail_info_size:
        # stop as no further reduction
        num_residual_runs -= 1 # not count current run
        break

      # prep next run
      compressed_residual_lst.append(compressed_residual)
      current_compressed_fail_info_size = compressed_fail_info_size
      current_compressed_fail_info = compressed_fail_info

    # output
    header = {
      "shape": list(data.shape),
      "has_nan": has_nan,
      **({"nan_mask": compressed_nan_mask} if has_nan else {}),
      **current_compressed_fail_info,
    }

    compressed_obj = [header, compressed_x] + compressed_residual_lst
    if output_file:
      # save to file
      os.makedirs(os.path.dirname(output_file), exist_ok=True)
      with open(output_file, "wb") as f:   # 'wb' = write binary
        pickle.dump(compressed_obj, f)
      # compressed_file_size_bytes = len(compressed_bitstream)
      # compressed_file_size_bytes = os.path.getsize(output_file)

    compressed_bitstream = pickle.dumps(compressed_obj)

    logger_info = {
      'num_residual_runs' : num_residual_runs,
      'fail_bytes': current_compressed_fail_info_size,
      'compression_inference_time': compression_time,
      'decompression_inference_time': decompression_time,
    }
    return compressed_bitstream, logger_info

  def decompress(
    self,
    bit_stream=None,
    file_path=None,
  ):
    if file_path:
      with open(file_path, "rb") as f:   # 'wb' = write binary
        compressed_obj = pickle.load(f)
    else:
      assert bit_stream is not None
      compressed_obj = pickle.loads(bit_stream)
    header = compressed_obj[0]
    compressed_x = compressed_obj[1]
    compressed_residual_lst = compressed_obj[2:]

    '''
    Step 1: Decode Header
    '''
    shape = header["shape"]
    N, H, W = shape
    has_nan = header["has_nan"]
    if has_nan:
      packed_nan_mask = np.frombuffer(zlib.decompress(header["nan_mask"]), dtype=np.uint8)
      nan_mask = np.unpackbits(packed_nan_mask)[:N*H*W].reshape((N, H, W)).astype(bool)
    fail_mask, fail_val = self.decompress_fail_value([N, H, W], header)

    x_hat = self._decompress(compressed_x).reshape(shape)

    for compressed_residual in compressed_residual_lst:
      residual_hat = self._decompress(compressed_residual).reshape(shape)
      x_hat = x_hat + residual_hat
    
    x_hat[fail_mask] = fail_val
    if has_nan:
      x_hat[nan_mask] = np.nan

    return x_hat

class ErrorBoundedCompressionPipelineFullGPU(ErrorBoundedCompressionPipeline):
  pass
#   def _compress(
#     self,
#     x, # [N, H, W] in numpy array
#     error_bound,
#     net_id,
#     batch_size=1,
#   ):
#     # Normalize to [0, 1]
#     xmin = float(x.min().item())
#     xmax = float(x.max().item())
#     scale = (xmax - xmin) if xmax > xmin else 1.0
#     x = (x - xmin) / scale
#     norm_info={"min": xmin, "scale": scale}

#     # run comrpession
#     padding_granularity = 128
#     padding = self.get_padding(x.shape[-2], x.shape[-1], padding_granularity)
#     num_slices = x.shape[0]
    
#     meta_data = {
#       "net": net_id,
#       "pad": padding,
#       **norm_info,
#     }

#     x = self.pad(x, padding).unsqueeze(1).repeat(1, 3, 1, 1)
#     out_enc = eval(f"self.net{net_id}").compress(x)
#     out_enc["shape"] = list(out_enc["shape"]) # for bitstream packet, original datatype is torch.Size
    
#     return {
#       **meta_data,
#       "res": out_enc,
#     }
  
#   def _decompress(
#     self,
#     nested,
#   ):
#     net_id = nested['net']
#     padding = nested['pad']
#     xmin = nested['min']
#     scale = nested['scale']
#     out_enc = nested['res']

#     out_dec = eval(f"self.net{net_id}").decompress(out_enc["strings"], out_enc["shape"])
#     out_dec["x_hat"] = self.crop(out_dec["x_hat"], padding).mean(dim=-3)
#     data_hat = out_dec["x_hat"]*scale + xmin
#     return data_hat
  
#   @staticmethod
#   def compress_fail_value(x, x_hat, error_bound, exclude_mask):
#     error = torch.abs(x - x_hat)
#     fail_mask = error > error_bound
#     fail_mask[exclude_mask] = False
#     fail_idx = torch.nonzero(fail_mask, as_tuple=False).view(-1).to(torch.int32)
#     fail_val = x.view(-1)[fail_idx]

#     fail_mask = fail_mask.detach().cpu().numpy()
#     fail_idx  = fail_idx .detach().cpu().numpy()
#     fail_val  = fail_val .detach().cpu().numpy()

#     # compress them
#     packed_fail_mask = np.packbits(fail_mask.ravel())
#     compressed_fail_mask = zlib.compress(packed_fail_mask.tobytes(), level=6)
#     compressed_fail_idx = zlib.compress(fail_idx.tobytes(), level=6)
#     compressed_fail_val = zlib.compress(fail_val.tobytes(), level=6)

#     if len(compressed_fail_mask) <= len(compressed_fail_idx):
#       # use mask
#       compressed_fail_idx = None
#       compressed_fail_info_size = len(compressed_fail_mask) + len(compressed_fail_val)
#       compressed_fail_info = {
#         "fail_mask": compressed_fail_mask,
#         "fail_val": compressed_fail_val,
#       }
#     else:
#       compressed_fail_mask = None
#       compressed_fail_info_size = len(compressed_fail_idx) + len(compressed_fail_val)
#       compressed_fail_info = {
#         "fail_idx": compressed_fail_idx,
#         "fail_val": compressed_fail_val,
#       }

#     return compressed_fail_info_size, fail_mask, fail_val, compressed_fail_info

#   def compress_slice(
#     self,
#     data, 
#     error_bound, 
#     batch_size,
#     max_residual_runs,
#   ):
#     data = torch.from_numpy(data).to(self.device)
#     error_bound = torch.from_numpy(error_bound).to(self.device)
#     N, H, W = data.shape

#     '''
#     Step 1: handle NaN
#     '''
#     x, nan_mask_data, compressed_nan_info = self.process_nan(data, run_compression=True, fill_by='min')
#     has_nan = compressed_nan_info['has_nan']
#     if has_nan:
#       compressed_nan_mask = compressed_nan_info['compressed_nan_mask']

#     error_bound, nan_mask_error_bound, _ = self.process_nan(error_bound, run_compression=False, fill_by='max')

#     # exclude mask = points donot care
#     exclude_mask = nan_mask_data | nan_mask_error_bound

#     x = torch.from_numpy(x).to(self.device)
#     error_bound = torch.from_numpy(error_bound).to(self.device)
#     '''
#     Step 2: compression
#     '''
#     net_id = 1
#     compressed_x = self._compress(
#       x,
#       error_bound=error_bound,
#       net_id=net_id,
#       batch_size=batch_size,
#     )
#     x_hat = self._decompress(compressed_x)
#     assert x_hat.dtype == torch.float32
#     compressed_fail_info_size, fail_mask, fail_val, compressed_fail_info = self.compress_fail_value(x, x_hat, error_bound, exclude_mask)




#     '''
#     Step 3: residual compression
#     '''
#     x_hat = torch.zeros_like(x, dtype=x.dtype, device=self.device)
#     num_residual_runs = 0
#     compressed_results = []
#     num_fail_points = x.numel()
#     fail_info = {}
#     # _debug_x_hat_lst = []
#     while True:
#       residual = x - x_hat

#       # residual compression
#       net_id = 1 if num_residual_runs == 0 else 2
#       compressed_residual_nested = self._compress(
#         residual,
#         error_bound=error_bound,
#         net_id=net_id,
#         batch_size=batch_size,
#       )

#       residual_hat = self._decompress(compressed_residual_nested)
#       assert residual_hat.dtype == torch.float32
#       x_hat = x_hat + residual_hat

#       # error
#       error = torch.abs(x - x_hat)
#       fail_idx = torch.where((error > error_bound).flatten())[0].to(torch.int32)
#       fail_val = x.flatten().index_select(0, fail_idx)

#       # stop condition
#       if num_residual_runs > 0: # at least run once
#         prev_fail_bytes = num_fail_points * 4 * 2
#         current_fail_bytes = fail_idx.numel() * 4 * 2
#         compressed_residual_bitstream = pickle.dumps(compressed_residual_nested)
#         if len(compressed_residual_bitstream) + current_fail_bytes >= prev_fail_bytes:
#           num_residual_runs -= 1
#           break

#       # prep next run
#       compressed_results.append(compressed_residual_nested)
#       num_fail_points = fail_idx.numel()
#       fail_info = {
#         'fail_idx': fail_idx,
#         'fail_val': fail_val,
#       }

#       if max_residual_runs >= 0:
#         if num_residual_runs >= max_residual_runs:
#           # num_residual_runs: num runs after current loop (first run not counted as residual run)
#           break

#       num_residual_runs += 1

#     # output
#     fail_info = {k: v.cpu().numpy().tobytes() for k, v in fail_info.items()}
#     header = {
#       "shape": list(x.shape),
#       **fail_info,
#     }

#     nested = [header] + compressed_results
#     return nested

#   def compress(
#     self, 
#     data, 
#     error_bound=None, 
#     batch_size=1,
#     max_residual_runs=-1,
#     output_file=None,
#   ):
#     if not isinstance(data, np.ndarray):
#       raise TypeError("arr must be a NumPy ndarray")
#     if data.dtype != np.float32:
#       data = data.astype(np.float32, copy=False)
#     if data.ndim != 3:
#       raise ValueError("arr must have exactly 3 dims; last two are [N, H, W]")

#     # To [N, H, W]
#     N, H, W = data.shape

#     # To [N, H, W]
#     num_slices = N

#     start_idx = 0
#     results = []
#     while start_idx < num_slices:
#       # print(f"[INFO] Compressing {start_idx}/{num_slices}")
#       end_idx = min(start_idx + batch_size, num_slices)
#       with torch.no_grad():
#         result = self.compress_slice(
#           data[start_idx:end_idx],
#           error_bound[start_idx:end_idx],
#           batch_size=batch_size,
#           max_residual_runs=max_residual_runs,
#         )
#         results.append(result)
#       start_idx = end_idx
    
#     if output_file:
#       # save to file
#       os.makedirs(os.path.dirname(output_file), exist_ok=True)
#       with open(output_file, "wb") as f:   # 'wb' = write binary
#         pickle.dump(results, f)
#       # compressed_file_size_bytes = len(compressed_bitstream)
#       # compressed_file_size_bytes = os.path.getsize(output_file)

#     compressed_bitstream = pickle.dumps(results)

#     info = {
#     }
#     return compressed_bitstream, info

#   def decompress(
#     self,
#     bit_stream=None,
#     file_path=None,
#   ):
#     if file_path:
#       with open(file_path, "rb") as f:   # 'wb' = write binary
#         nested = pickle.load(f)
#     else:
#       assert bit_stream is not None
#       nested = pickle.loads(bit_stream)
    
#     decompressed_results = []
#     for result in nested:
#       with torch.no_grad():
#         header = result[0]
#         compressed_results = result[1:]

#         shape = header["shape"]
#         data_hat_slice = torch.zeros(shape, dtype=torch.float32, device=self.device)
#         for compressed_residual_nested in compressed_results:
#           residual_hat = self._decompress(compressed_residual_nested)
#           data_hat_slice += residual_hat.reshape(shape)
        
#         fail_idx = torch.from_numpy(np.frombuffer(header["fail_idx"], dtype=np.int32)).to(self.device)
#         fail_val = torch.from_numpy(np.frombuffer(header["fail_val"], dtype=np.float32)).to(self.device)
#         data_hat_slice.view(-1)[fail_idx] = fail_val
#         decompressed_results.append(data_hat_slice.cpu().numpy())
#     data_hat = np.concatenate(decompressed_results, axis=0)
  
#     return data_hat