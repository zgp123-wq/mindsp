
import math
import itertools
import mindspore
import mindspore.context as context
from data.coco import COCODataset
import mindspore.dataset as ds
import bisect
import numpy as np
from data.dis_simper import DistributedSampler
from data.transforms import train_aug, val_aug

context.set_context(mode=context.GRAPH_MODE)

# init()  # Initialize MindSpore's distributed environment

class group_sampler:
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    """

    def __init__(self, sampler, group_ids, batch_size, drop_uneven=False):
        self.sampler = sampler
        self.group_ids = mindspore.Tensor(group_ids, mindspore.int32)
        assert self.group_ids.ndim == 1
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven
        # self.groups = self.group_ids.unique().sort()[0]
        
        # Convert tensor to numpy, get unique values and sort them
        unique_vals = np.unique(self.group_ids.asnumpy())
        sorted_vals = np.sort(unique_vals)
        
        # Convert sorted numpy array back to MindSpore tensor
        self.groups = mindspore.Tensor(sorted_vals, mindspore.int32)
        self._can_reuse_batches = False

    def _prepare_batches(self):  # TODO: get to understand this
        dataset_size = len(self.group_ids)
        # get the sampled indices from the sampler
        sampled_ids = mindspore.Tensor(list(self.sampler), mindspore.int32)
        # potentially not all elements of the dataset were sampled by the sampler (e.g., DistributedSampler).
        # construct a tensor which contains -1 if the element was not sampled, and a non-negative number
        # indicating the order where the element was sampled.
        # for example. if sampled_ids = [3, 1] and dataset_size = 5, the order is [-1, 1, -1, 0, -1]
        order = mindspore.Tensor([-1] * dataset_size, mindspore.int64)
        numpy_arr = np.arange(len(sampled_ids))
        order[sampled_ids] = mindspore.Tensor(numpy_arr, dtype=mindspore.int64)
        # order[sampled_ids] = mindspore.arange(len(sampled_ids), dtype=mindspore.int64)

        # get a mask with the elements that were sampled
        mask = order >= 0
        # find the elements that belong to each individual cluster
        clusters = [(self.group_ids == i) & mask for i in self.groups]
        # get relative order of the elements inside each cluster that follows the order from the sampler
        relative_order = [order[cluster] for cluster in clusters]
        # with the relative order, find the absolute order in the sampled space
        permutation_ids = [s[s.argsort()] for s in relative_order]
        # permute each cluster so that they follow the order from the sampler
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]

        # splits each cluster in batch_size, and merge as a list of tensors
        splits = [c.split(self.batch_size) for c in permuted_clusters]
        merged = tuple(itertools.chain.from_iterable(splits))

        # now each batch internally has the right order, but they are grouped by clusters.
        # Find the permutation between different batches that brings them as close as possible to
        # the order that we have in the sampler. For that, we will consider the ordering as coming from
        # the first element of each batch, and sort correspondingly
        first_element_of_batch = [t[0].item() for t in merged]
        # get and inverse mapping from sampled indices and the position where they occur (as returned by the sampler)
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.asnumpy().tolist())}
        # from the first element in each batch, get a relative ordering
        first_index_of_batch = mindspore.Tensor([inv_sampled_ids_map[s] for s in first_element_of_batch], mindspore.int64)

        # permute the batches so that they approximately follow the order from the sampler
        permutation_order = first_index_of_batch.sort()[1].asnumpy().tolist()
        # finally, permute the batches
        batches = [merged[i].asnumpy().tolist() for i in permutation_order]

        if self.drop_uneven:
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
            batches = kept
        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches

        return iter(batches)

    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True

        return len(self._batches)


class iteration_sampler:
    # Wraps a BatchSampler, resampling from it until a specified number of iterations have been sampled
    def __init__(self, batch_sampler, num_iters, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iters = num_iters
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iters:
            # if the underlying sampler has a set_epoch method, like DistributedSampler,
            # used for making each process see a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)

            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iters:
                    break

                yield batch

    def __len__(self):
        return self.num_iters


class BatchCollator:
    def __init__(self):
        self.size_divisible = 32

    def __call__(self, batch):
        batch_list = list(zip(*batch))
        img_list_batch, box_list_batch = batch_list[0], batch_list[1]
        max_size = tuple(max(s) for s in zip(*[img_list.img.shape for img_list in img_list_batch]))

        if self.size_divisible > 0:
            stride = self.size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(img_list_batch),) + max_size
        batched_imgs = img_list_batch[0].img.new(*batch_shape).zero_()

        for img_l, pad_img in zip(img_list_batch, batched_imgs):
            pad_img[: img_l.img.shape[0], : img_l.img.shape[1], : img_l.img.shape[2]].copy_(img_l.img)
            img_l.img = pad_img
            img_l.padded_size = (pad_img.shape[2], pad_img.shape[1])

        return img_list_batch, box_list_batch
    

import random

class MyRandomSampler():
    def __init__(self, data_source, num_samples=None, replacement=False):
        self.data_source = data_source
        self.num_samples = num_samples if num_samples is not None else len(data_source)
        self.replacement = replacement

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        if self.replacement:
            return iter([random.choice(indices) for _ in range(self.num_samples)])
        else:
            return iter(random.sample(indices, self.num_samples))


from mindspore.dataset import samplers

def make_data_loader(cfg, start_iter=0, during_training=False):
    valing = cfg.val_mode or during_training
    dataset = COCODataset(cfg, valing)

    if not valing:
        batch_size = cfg.train_bs
        # sampler = samplers.RandomSampler(num_samples=len(dataset))
        distributed_sampler = DistributedSampler(len(dataset))
        num_iters = cfg.max_iter
    else:
        batch_size = int(cfg.test_bs)
        sampler = ds.SequentialSampler(len(dataset))
        num_iters = None

    # group in two cases: those with width / height > 1, and the other way around,
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)

    # group_ids = list(map(lambda y: bisect.bisect_right([1], y), aspect_ratios))

    # batch_sampler = group_sampler(sampler, group_ids, batch_size, drop_uneven=False)  # same as drop_last

    # if num_iters is not None:
    #     batch_sampler = iteration_sampler(batch_sampler, num_iters, start_iter)

    data_loader = ds.GeneratorDataset(dataset, column_names=["data"], sampler=distributed_sampler)
    data_loader = data_loader.map(operations=train_aug, python_multiprocessing=True)
 
    data_loader =  data_loader.batch(32, drop_remainder=True)

    return data_loader


