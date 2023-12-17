import os, json, argparse
from copy import deepcopy
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm

import hw_code.model as module_model
from hw_code.trainer import ASTrainer as Trainer
from hw_code.utils import ROOT_PATH
from hw_code.utils.object_loading import get_datasets_and_dataloaders
from hw_code.utils.parse_config import ConfigParser
import hw_code.metric as module_metric
import hw_code.model as module_arch

from calculate_eer import compute_eer

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default-test-folder" / "model_best.pth"


def main(config, out_file):
  logger = config.get_logger("test")

  # define cpu or gpu if possible
  device = 'cuda'  #torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # text_encoder (absent)
  #text_encoder = config.get_text_encoder()
  text_encoder = None

  # setup data_loader instances
  datasets, dataloaders = get_datasets_and_dataloaders(config, text_encoder)

  # build model architecture, then print to console
  model = config.init_obj(config["arch"], module_arch)
          # # n_possible_train_speakers=9027, total number of speakers in Librispeech
  print('model with', sum(p.numel() for p in model.parameters()), 'parameters')
  logger.info(model)

  # build metrics
  metrics = [
    config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
    for metric_dict in config["metrics"]
  ]

  logger.info("Loading checkpoint: {} ...".format(config.resume))
  checkpoint = torch.load(config.resume, map_location=device)
  state_dict = checkpoint["state_dict"]
  if config["n_gpu"] > 1:
    model = torch.nn.DataParallel(model)
  model.load_state_dict(state_dict)

  # prepare model for testing
  model = model.to(device)
  model.eval()

  results = []

  bonafide_scores, spoof_scores = [], []
  model.eval()
  with torch.inference_mode():
    for batch in tqdm(dataloaders["test"]):
      batch = Trainer.move_batch_to_device(batch, device)
      batch_logits = model(batch["audio"])
      bonafide_scores += batch_logits[batch["label"] == 1, 1].cpu().tolist()
      spoof_scores += batch_logits[batch["label"] == 0, 1].cpu().tolist()
  eer, threshold = compute_eer(bonafide_scores, spoof_scores)
  print('EER', eer)
  print('threshold', threshold)

  quit(0)

  #    for path, wav in zip(batch["save_to_path"], wavs):
  #      torchaudio.save(path, wav.cpu(), sample_rate=config["preprocessing"]["sr"])
  #    for i in []:  #range(len(batch["text"])):
  #      results.append(
  #        {
  #          "ground_truth": batch["text"][i],
  #          "pred_text_argmax": text_encoder.ctc_decode(argmax.cpu().numpy()),
  #          "pred_text_beam_search": text_encoder.ctc_beam_search_with_lm(
  #            batch["probs"][i][:batch["log_probs_length"][i]], beam_size=3
  #          )[:10],
  #        }
  #      )
  results.append([metric_object.toJSON() for metric_object in metrics])
  with Path(out_file).open("w") as f:
    json.dump(results, f, indent=2)


if __name__ == "__main__":
  args = argparse.ArgumentParser(description="PyTorch Template")
  args.add_argument(
    "-c",
    "--config",
    default=None,
    type=str,
    help="config file path (default: None)",
  )
  args.add_argument(
    "-r",
    "--resume",
    default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
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
  args.add_argument(
    "-o",
    "--output",
    default="output.json",
    type=str,
    help="File to write results (.json)",
  )
  args.add_argument(
    "-t",
    "--test-data-folder",
    default=None,
    type=str,
    help="Path to dataset",
  )
  args.add_argument(
    "-b",
    "--batch-size",
    default=1,
    type=int,
    help="Test dataset batch size",
  )
  args.add_argument(
    "-j",
    "--jobs",
    default=1,
    type=int,
    help="Number of workers for test dataloader",
  )

  args = args.parse_args()

  # set GPUs
  if args.device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

  # first, we need to obtain config with model parameters
  # we assume it is located with checkpoint in the same folder
  model_config = Path(args.resume).parent / "config-for-test.json"
  with model_config.open() as f:
    config = ConfigParser(json.load(f), resume=args.resume)

  # update with addition configs from `args.config` if provided
  if args.config is not None:
    with Path(args.config).open() as f:
      config.config.update(json.load(f))

  # if `--test-data-folder` was provided, set it as a default test set
  if args.test_data_folder is not None:
    test_data_folder = Path(args.test_data_folder).absolute().resolve()
    assert test_data_folder.exists()
    print('args', args)
    #config.config["data"] = {
    #  "test": {
    #    "batch_size": args.batch_size,
    #    "num_workers": args.jobs,
    #    "datasets": [
    #      {
    #        "type": "CustomDirAudioDataset",
    #        "args": {
    #          "audio_dir": str(test_data_folder / "audio"),
    #          "transcription_dir": str(
    #            test_data_folder / "transcriptions"
    #          ),
    #        },
    #      }
    #    ],
    #  }
    #}

  assert config.config.get("data", {}).get("test", None) is not None
  #config["data"]["test"]["batch_size"] = args.batch_size
  #config["data"]["test"]["n_jobs"] = args.jobs

  main(config, args.output)
