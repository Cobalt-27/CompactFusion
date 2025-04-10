import torch
import torch.distributed
import torch.distributed.rpc
"""
Cache for async allgather operations
"""
HANDLES_IDX = 0
RECV_BUF_IDX = 1
SEND_BUF_IDX = 2
ENTRY_VAL_LEN = 3

class DummyHandle:
    """
    A dummy handle that does nothing
    """
    def wait(self):
        pass

class AllGatherCache:
    def __init__(self):
        self.cache = {}
    
    def clear(self):
        self.cache = {}

    def put(self, key, handle, recv_buf_list, send_buf):
        assert isinstance(recv_buf_list, list)
        assert isinstance(send_buf, torch.Tensor)
        self.cache[key] = (handle, recv_buf_list, send_buf)
    
    def get(self, key):
        assert isinstance(self.cache[key][RECV_BUF_IDX], list)
        return self.cache[key]

    def contains(self, key):
        return key in self.cache
    
    def tensors_size(self):
        """
        return the total size of all tensors in bytes
        """
        size = 0
        for key, value in self.cache.items():
            if value[SEND_BUF_IDX] is not None:
                size += value[SEND_BUF_IDX].numel() * value[SEND_BUF_IDX].element_size()
            for tensor in value[RECV_BUF_IDX]:
                if tensor is not None:
                    size += tensor.numel() * tensor.element_size()
        return size