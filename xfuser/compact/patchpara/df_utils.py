import torch

class PatchConfig:
    def __init__(self,
                 use_compact: bool,
                 async_comm: bool,
                 async_warmup: int,
                 ) -> None:
        # assert not use_compact, "Compact seems to be buggy, results are inconsistent with ring impl."
        self.use_compact = use_compact
        self.async_comm = async_comm # Enable DistriFusion
        self.async_warmup = async_warmup
        if use_compact:
            assert not async_comm, "Compact does not support async communication"
        if async_comm:
            assert not use_compact, "Async communication does not support compact"


