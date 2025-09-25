import json, struct
from typing import Any, Optional
import numpy as np
from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray, ndarray_copy
from numcodecs.registry import register_codec

from .compressor import ErrorBoundedCompressionPipeline

class ErrorBounded2DCodec(Codec):
  """
  Splits leading batch dims, encodes each [h,w] slice separately with its
  corresponding error_bound slice.
  """
  codec_id = "error_bounded_2d"

  def __init__(self, error_bound_file, batch_size, compressor_kwargs):
    try:
      self.error_bound = np.load(error_bound_file)
    except:
      self.error_bound = None
    self.error_bound_file = error_bound_file
    self.batch_size = batch_size
    self.compressor_kwargs = compressor_kwargs
    self.compressor = ErrorBoundedCompressionPipeline(**compressor_kwargs)

  def encode(self, buf):
    assert self.error_bound is not None
    data = ensure_ndarray(buf)

    if data.shape != self.error_bound.shape:
      raise ValueError("error_bound must have same shape as input")

    return self.compressor.compress(
      data, 
      self.error_bound, 
      batch_size = self.batch_size,
    )[0]

  def decode(self, buf, out = None):
    data_hat = self.compressor.decompress(bit_stream=buf)
    if out is not None:
      if out.shape != data_hat.shape or out.dtype != data_hat.dtype:
        raise ValueError("out has wrong shape/dtype")
      ndarray_copy(data_hat, out)
      return out
    return data_hat

  def get_config(self):
    # error_bound is not persisted — must be provided by user each time
    return {
      "id": self.codec_id,
      "error_bound_file": self.error_bound_file,
      "batch_size": self.batch_size,
      "compressor_kwargs": self.compressor_kwargs
    }

  @classmethod
  def from_config(cls, cfg):
    return cls(
      error_bound_file=cfg['error_bound_file'],
      batch_size=cfg['batch_size'],
      compressor_kwargs=cfg['compressor_kwargs'],
    )


register_codec(ErrorBounded2DCodec)
