from operator import xor

from torch.utils.data import ConcatDataset, DataLoader

import hw_code.augmentations
import hw_code.datasets
from hw_code import batch_sampler as batch_sampler_module
from hw_code.base.base_text_encoder import BaseTextEncoder
from hw_code.collate_fn import get_collate_fn
from hw_code.utils.parse_config import ConfigParser


def get_datasets_and_dataloaders(configs: ConfigParser, text_encoder: BaseTextEncoder):
  datasets_, dataloaders_= {}, {}
  for split, params in configs["data"].items():
    num_workers = params.get("num_workers", 1)

    # set train augmentations
    if split == 'train':
      wave_augs, spec_augs = hw_code.augmentations.from_configs(configs)
      drop_last = True
    else:
      wave_augs, spec_augs = None, None
      drop_last = False

    # create and join datasets
    datasets = []
    for ds in params["datasets"]:
      datasets.append(configs.init_obj(
        ds, hw_code.datasets, text_encoder=text_encoder, config_parser=configs,
        wave_augs=wave_augs, spec_augs=spec_augs))
    assert len(datasets)
    if len(datasets) > 1:
      dataset = ConcatDataset(datasets)
    else:
      dataset = datasets[0]

    # select batch size or batch sampler
    assert xor("batch_size" in params, "batch_sampler" in params), \
      "You must provide batch_size or batch_sampler for each split"
    if "batch_size" in params:
      bs = params["batch_size"]
      shuffle = True
      batch_sampler = None
    elif "batch_sampler" in params:
      batch_sampler = configs.init_obj(params["batch_sampler"], batch_sampler_module,
                       data_source=dataset)
      bs, shuffle = 1, False
    else:
      raise Exception()

    # Fun fact. An hour of debugging was wasted to write this line
    assert bs <= len(dataset), \
      f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

    # create dataloader
    dataloader = DataLoader(dataset, batch_size=bs, collate_fn=get_collate_fn(dataset), shuffle=shuffle,
                            num_workers=num_workers, batch_sampler=batch_sampler, drop_last=drop_last, pin_memory=True)
    datasets_[split] = dataset
    dataloaders_[split] = dataloader
  return datasets_, dataloaders_
