from typing import Literal, Optional

from pydantic import BaseModel


class AlgoBase(BaseModel):
    name: Optional[str] = None
    num_process: int = 3
    device: str = 'cuda'
    parallel_backend: Literal['dask', 'sequential', 'balanced_dask', 'balanced_multiprocessing', 'multiprocessing', 'batched'] = 'balanced_dask'
    batch_size: int = 64
    run_episode_func: str = 'default'

    seed: Optional[int] = 0
    preprocessing: Optional[str] = None
