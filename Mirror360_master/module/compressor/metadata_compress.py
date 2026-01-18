import pickle
import json
import gzip
import bz2
import lzma
import zlib
import numpy as np
from pathlib import Path
import time
import sys


import zstandard as zstd
import pickle

def save_metadata(metadata, path, level=9):
    cctx = zstd.ZstdCompressor(level=level)
    with open(path, 'wb') as f:
        with cctx.stream_writer(f) as w:
            pickle.dump(metadata, w, protocol=pickle.HIGHEST_PROTOCOL)

def load_metadata(path):
    dctx = zstd.ZstdDecompressor()
    with open(path, 'rb') as f:
        with dctx.stream_reader(f) as r:
            return pickle.load(r)
