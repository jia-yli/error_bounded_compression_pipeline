import os
import re
import torch
import time
import itertools
import pickle

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
      print(f"[INFO] Compressing {start_idx}/{num_slices}")
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
  def slice_dominant(vals):
    v = vals[~np.isnan(vals)]
    if v.size == 0:
      return np.nan, 0, 0
    u, counts = np.unique(v, return_counts=True)
    idx = np.argmax(counts)
    return u[idx], counts[idx], v.size

  @staticmethod
  def find_dominant(x, threshold=0.3):
    N, H, W = x.shape
    x_flat = x.reshape(N, -1)

    dom_counts = np.zeros(N, dtype=np.int32)

    results = [ErrorBoundedCompressionPipeline.slice_dominant(x_flat[i]) for i in range(N)]

    dom_vals = np.array([r[0] for r in results], dtype=x.dtype)
    dom_counts = np.array([r[1] for r in results], dtype=np.int32)
    valid_counts = np.array([r[2] for r in results], dtype=np.int32)

    # Threshold check
    has_dom = dom_counts > threshold * valid_counts

    # Build masks (vectorized where possible)
    dom_mask = (x == dom_vals[:, np.newaxis, np.newaxis]) & ~np.isnan(x)
    dom_mask[~has_dom] = False

    dom_vals = dom_vals[has_dom]
    reduced_masks = dom_mask[has_dom]

    return has_dom, dom_mask, dom_vals, reduced_masks

  @staticmethod
  def recover_from_reduced_masks(has_dom, dom_vals, reduced_masks, shape):
    N, H, W = shape
    recovered_mask = np.zeros(shape, dtype=bool)
    dom_filled = np.full(shape, np.nan)

    idxs = np.where(has_dom)[0]
    if idxs.size > 0:
        recovered_mask[idxs] = reduced_masks
        dom_filled[idxs] = reduced_masks * dominant_vals[idxs, None, None]

    return recovered_mask, dom_filled

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
    if data.ndim < 3:
      raise ValueError("arr must have at least 3 dims; last two are [H, W]")
    
    # To [N, H, W]
    H, W = data.shape[-2:]
    x = data.reshape(-1, H, W)

    # '''
    # Step 1. detect const / fill value
    # '''
    # nan_mask = np.isnan(x)
    # has_nan = bool(nan_mask.any())
    # if has_nan:
    #   packed_mask = np.packbits(nan_mask.ravel())
    #   # bits = np.unpackbits(packed_mask)
    #   # bits = bits[:num_bits]  # trim padding at the end
    
    # '''
    # Step 2. Find Dominant Value
    # '''
    # u, counts = np.unique(x[~np.isnan(x)], return_counts=True)
    # dominant_val = u[np.argmax(counts)]
    # dominant_mask = (x == dominant_val)
    # has_dominant = np.sum(dominant_mask) > 0.3*x.size
    # if has_dominant:
    #   packed_dominant_mask = np.packbits(dominant_mask.ravel())


    x_hat = np.zeros_like(x, dtype=x.dtype)
    num_residual_runs = 0
    compressed_results = []
    num_fail_points = x.size
    fail_info = {}
    # _debug_x_hat_lst = []
    while True:
      residual = x - x_hat

      # residual compression
      net_id = 1 if num_residual_runs == 0 else 2
      compressed_residual_nested = self._compress(
        residual,
        error_bound=error_bound,
        net_id=net_id,
        batch_size=batch_size,
      )

      residual_hat = self._decompress(compressed_residual_nested)
      assert residual_hat.dtype == np.float32
      x_hat = x_hat + residual_hat
      # _debug_x_hat_lst.append(x_hat)
      # check
      # compressed_residual_nested_1 = self._compress(residual,error_bound=error_bound,net_id=net_id,batch_size=1)
      # compressed_residual_nested_4 = self._compress(residual,error_bound=error_bound,net_id=net_id,batch_size=4)
      # residual_hat_1 = self._decompress(compressed_residual_nested_1)
      # residual_hat_4 = self._decompress(compressed_residual_nested_4)
      # # not same
      # import pdb;pdb.set_trace()

      # error
      error = np.abs(x - x_hat)
      fail_idx = np.flatnonzero(error > error_bound).astype(np.int32)
      fail_val = x.flat[fail_idx]

      # stop condition
      if num_residual_runs > 0: # at least run once
        prev_fail_bytes = num_fail_points * 4 * 2
        current_fail_bytes = fail_idx.size * 4 * 2
        compressed_residual_bitstream = pickle.dumps(compressed_residual_nested)
        if len(compressed_residual_bitstream) + current_fail_bytes >= prev_fail_bytes:
          num_residual_runs -= 1
          break

      # prep next run
      compressed_results.append(compressed_residual_nested)
      num_fail_points = fail_idx.size
      fail_info = {
        'fail_idx': fail_idx,
        'fail_val': fail_val,
      }

      if max_residual_runs >= 0:
        if num_residual_runs >= max_residual_runs:
          # num_residual_runs: num runs after current loop (first run not counted as residual run)
          break

      num_residual_runs += 1

    # output
    fail_info = {k: v.tobytes() for k, v in fail_info.items()}
    header = {
      "test_header": "test_header",
      "shape": list(data.shape),
      **fail_info,
    }

    nested = [header] + compressed_results
    if output_file:
      # save to file
      os.makedirs(os.path.dirname(output_file), exist_ok=True)
      with open(output_file, "wb") as f:   # 'wb' = write binary
        pickle.dump(nested, f)
      # compressed_file_size_bytes = len(compressed_bitstream)
      # compressed_file_size_bytes = os.path.getsize(output_file)

    compressed_bitstream = pickle.dumps(nested)

    # check
    # x_hat_c = _debug_x_hat_lst[-2].copy()
    # fail_idx = np.frombuffer(header["fail_idx"], dtype=np.int32)
    # fail_val = np.frombuffer(header["fail_val"], dtype=np.float32)
    # x_hat_c.flat[fail_idx] = fail_val
    # x_hat_d = self.decompress(bit_stream = compressed_bitstream)
    # print((x_hat_c == x_hat_d).all())
    # print((np.abs(data - x_hat_d) <= error_bound).all())
    # import pdb;pdb.set_trace()
    info = {
      'num_residual_runs' : num_residual_runs,
    }
    return compressed_bitstream, info

  def decompress(
    self,
    bit_stream=None,
    file_path=None,
  ):
    if file_path:
      with open(file_path, "rb") as f:   # 'wb' = write binary
        nested = pickle.load(f)
    else:
      assert bit_stream is not None
      nested = pickle.loads(bit_stream)
    header = nested[0]
    compressed_results = nested[1:]

    shape = header["shape"]
    data_hat = np.zeros(shape, dtype=np.float32)
    for compressed_residual_nested in compressed_results:
      residual_hat = self._decompress(compressed_residual_nested)
      data_hat += residual_hat.reshape(shape)
    
    fail_idx = np.frombuffer(header["fail_idx"], dtype=np.int32)
    fail_val = np.frombuffer(header["fail_val"], dtype=np.float32)
    data_hat.flat[fail_idx] = fail_val
  
    return data_hat

class ErrorBoundedCompressionPipelineFullGPU(ErrorBoundedCompressionPipeline):
  def _compress(
    self,
    x, # [N, H, W] in numpy array
    error_bound,
    net_id,
    batch_size=1,
  ):
    # Normalize to [0, 1]
    xmin = float(x.min().item())
    xmax = float(x.max().item())
    scale = (xmax - xmin) if xmax > xmin else 1.0
    x = (x - xmin) / scale
    norm_info={"min": xmin, "scale": scale}

    # run comrpession
    padding_granularity = 128
    padding = self.get_padding(x.shape[-2], x.shape[-1], padding_granularity)
    num_slices = x.shape[0]
    
    meta_data = {
      "net": net_id,
      "pad": padding,
      **norm_info,
    }

    x = self.pad(x, padding).unsqueeze(1).repeat(1, 3, 1, 1)
    out_enc = eval(f"self.net{net_id}").compress(x)
    out_enc["shape"] = list(out_enc["shape"]) # for bitstream packet, original datatype is torch.Size
    
    return {
      **meta_data,
      "res": out_enc,
    }
  
  def _decompress(
    self,
    nested,
  ):
    net_id = nested['net']
    padding = nested['pad']
    xmin = nested['min']
    scale = nested['scale']
    out_enc = nested['res']

    out_dec = eval(f"self.net{net_id}").decompress(out_enc["strings"], out_enc["shape"])
    out_dec["x_hat"] = self.crop(out_dec["x_hat"], padding).mean(dim=-3)
    data_hat = out_dec["x_hat"]*scale + xmin
    return data_hat

  def compress_slice(
    self,
    x, 
    error_bound, 
    batch_size,
    max_residual_runs,
  ):
    x_hat = torch.zeros_like(x, dtype=x.dtype, device=self.device)
    num_residual_runs = 0
    compressed_results = []
    num_fail_points = x.numel()
    fail_info = {}
    # _debug_x_hat_lst = []
    while True:
      residual = x - x_hat

      # residual compression
      net_id = 1 if num_residual_runs == 0 else 2
      compressed_residual_nested = self._compress(
        residual,
        error_bound=error_bound,
        net_id=net_id,
        batch_size=batch_size,
      )

      residual_hat = self._decompress(compressed_residual_nested)
      assert residual_hat.dtype == torch.float32
      x_hat = x_hat + residual_hat

      # error
      error = torch.abs(x - x_hat)
      fail_idx = torch.where((error > error_bound).flatten())[0].to(torch.int32)
      fail_val = x.flatten().index_select(0, fail_idx)

      # stop condition
      if num_residual_runs > 0: # at least run once
        prev_fail_bytes = num_fail_points * 4 * 2
        current_fail_bytes = fail_idx.numel() * 4 * 2
        compressed_residual_bitstream = pickle.dumps(compressed_residual_nested)
        if len(compressed_residual_bitstream) + current_fail_bytes >= prev_fail_bytes:
          num_residual_runs -= 1
          break

      # prep next run
      compressed_results.append(compressed_residual_nested)
      num_fail_points = fail_idx.numel()
      fail_info = {
        'fail_idx': fail_idx,
        'fail_val': fail_val,
      }

      if max_residual_runs >= 0:
        if num_residual_runs >= max_residual_runs:
          # num_residual_runs: num runs after current loop (first run not counted as residual run)
          break

      num_residual_runs += 1

    # output
    fail_info = {k: v.cpu().numpy().tobytes() for k, v in fail_info.items()}
    header = {
      "shape": list(x.shape),
      **fail_info,
    }

    nested = [header] + compressed_results
    return nested

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
    if data.ndim < 3:
      raise ValueError("arr must have at least 3 dims; last two are [H, W]")
    
    # To [N, H, W]
    H, W = data.shape[-2:]
    x = data.reshape(-1, H, W)
    num_slices = x.shape[0]

    start_idx = 0
    results = []
    while start_idx < num_slices:
      print(f"[INFO] Compressing {start_idx}/{num_slices}")
      end_idx = min(start_idx + batch_size, num_slices)
      with torch.no_grad():
        x_tensor = torch.from_numpy(x[start_idx:end_idx]).to(self.device)
        error_bound_tensor = torch.from_numpy(error_bound[start_idx:end_idx]).to(self.device)
        result = self.compress_slice(
          x_tensor,
          error_bound_tensor,
          batch_size=batch_size,
          max_residual_runs=max_residual_runs,
        )
        results.append(result)
      start_idx = end_idx
    
    if output_file:
      # save to file
      os.makedirs(os.path.dirname(output_file), exist_ok=True)
      with open(output_file, "wb") as f:   # 'wb' = write binary
        pickle.dump(results, f)
      # compressed_file_size_bytes = len(compressed_bitstream)
      # compressed_file_size_bytes = os.path.getsize(output_file)

    compressed_bitstream = pickle.dumps(results)

    # check
    # x_hat_c = _debug_x_hat_lst[-2].copy()
    # fail_idx = np.frombuffer(header["fail_idx"], dtype=np.int32)
    # fail_val = np.frombuffer(header["fail_val"], dtype=np.float32)
    # x_hat_c.flat[fail_idx] = fail_val
    # x_hat_d = self.decompress(bit_stream = compressed_bitstream)
    # print((x_hat_c == x_hat_d).all())
    # print((np.abs(data - x_hat_d) <= error_bound).all())
    # import pdb;pdb.set_trace()
    info = {
    }
    return compressed_bitstream, info

  def decompress(
    self,
    bit_stream=None,
    file_path=None,
  ):
    if file_path:
      with open(file_path, "rb") as f:   # 'wb' = write binary
        nested = pickle.load(f)
    else:
      assert bit_stream is not None
      nested = pickle.loads(bit_stream)
    
    decompressed_results = []
    for result in nested:
      with torch.no_grad():
        header = result[0]
        compressed_results = result[1:]

        shape = header["shape"]
        data_hat_slice = torch.zeros(shape, dtype=torch.float32, device=self.device)
        for compressed_residual_nested in compressed_results:
          residual_hat = self._decompress(compressed_residual_nested)
          data_hat_slice += residual_hat.reshape(shape)
        
        fail_idx = torch.from_numpy(np.frombuffer(header["fail_idx"], dtype=np.int32)).to(self.device)
        fail_val = torch.from_numpy(np.frombuffer(header["fail_val"], dtype=np.float32)).to(self.device)
        data_hat_slice.view(-1)[fail_idx] = fail_val
        decompressed_results.append(data_hat_slice.cpu().numpy())
    data_hat = np.concatenate(decompressed_results, axis=0)
  
    return data_hat