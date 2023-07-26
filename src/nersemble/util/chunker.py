from math import ceil
from typing import Tuple

import torch


def chunked(max_chunk_size: int, *lists) -> Tuple:
    for value_list in lists:
        # All lists should have same length
        # There is the edge case where an empty tensor or None is passed. These are fine as well
        assert isinstance(value_list, torch.Tensor) and len(value_list.shape) == 0 \
               or value_list is None \
               or len(value_list) == len(lists[0])

    size = len(lists[0])
    for i_chunk in range(ceil(size / max_chunk_size)):
        sliced_lists = []
        for value_list in lists:
            if isinstance(value_list, torch.Tensor) and len(value_list.shape) == 0 or value_list is None:
                # Empty tensor or None
                sliced_lists.append(value_list)
            else:
                sliced_lists.append(value_list[i_chunk * max_chunk_size: (i_chunk + 1) * max_chunk_size])

        if len(sliced_lists) == 1:
            yield sliced_lists[0]
        else:
            yield tuple(sliced_lists)
