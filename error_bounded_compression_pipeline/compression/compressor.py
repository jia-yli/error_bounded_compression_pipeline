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

  # @staticmethod
  # def _detect_const_fill_2d(
  #   data,
  # ):
  #   H, W = data.shape
  #   n = H * W

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
