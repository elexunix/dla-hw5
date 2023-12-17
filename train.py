import argparse
import collections
import warnings
from copy import deepcopy

import numpy as np
import torch, torchaudio

import hw_code.losses as module_loss
import hw_code.metric as module_metric
import hw_code.model as module_arch
from hw_code.trainer import ASTrainer as Trainer
from hw_code.utils import prepare_device
from hw_code.utils.object_loading import get_datasets_and_dataloaders
from hw_code.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)
#torchaudio.set_audio_backend("sox")

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
  logger = config.get_logger("train")

  # text_encoder (absent)
  #text_encoder = config.get_text_encoder()
  text_encoder = None

  # setup data_loader instances
  datasets, dataloaders = get_datasets_and_dataloaders(config, text_encoder)

  # build model architecture, then print to console
  model = config.init_obj(config["arch"], module_arch)  # # n_possible_train_speakers=9027, total number of speakers in Librispeech
  print('model with', sum(p.numel() for p in model.parameters()), 'parameters')
  logger.info(model)

  # prepare for (multi-device) GPU training
  device, device_ids = prepare_device(config["n_gpu"])
  model = model.to(device)
  if len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)

  # get function handles of loss and metrics
  #loss_module = config.init_obj(config["loss"], module_loss).to(device)
  metrics = [
    config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
    for metric_dict in config["metrics"]
  ]

  trainable_params = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
  lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

  trainer = Trainer(
    model,
    metrics,
    optimizer,
    config=config,
    device=device,
    dataloaders=dataloaders,
    lr_scheduler=lr_scheduler,
    len_epoch=config["trainer"].get("len_epoch", None)
  )

  trainer.train()


if __name__ == "__main__":
  args = argparse.ArgumentParser(description="PyTorch Template")
  args.add_argument(
    "-c",
    "--config",
    #default="hw_code/configs/one_batch_test.json",
    default="hw_code/configs/config.json",
    type=str,
    help="config file path (default: None)",
  )
  args.add_argument(
    "-r",
    "--resume",
    default=None,
    type=str,
    help="path to latest checkpoint (default: None)",
  )
  args.add_argument(
    "-d",
    "--device",
    default=None,
    type=str,
    help="indices of GPUs to enable (default: all)",
  )

  # custom cli options to modify configuration from default values given in json file.
  CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
  options = [
    CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
    CustomArgs(
      ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
    ),
  ]
  config = ConfigParser.from_args(args, options)
  main(config)
