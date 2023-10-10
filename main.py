import configparser
import os
import traceback
import torch
from torch.utils.data import DataLoader
import utils
from utils import CONFIG_PATH
import model
import torchvision
from torchvision import transforms
from train_model import read_size

def model_factory(cfg):
   net_cfg = cfg["NETWORK_SETTINGS"]["network_size"].lower().strip()
   if net_cfg not in model.configs.keys():
      raise ValueError("network_size error. Expected one of {}".format(model.configs.keys()))
   net_cfg = model.configs[net_cfg]

   predictor = model.ConvNetModel(int(cfg["NETWORK_SETTINGS"]["img_channels"]), net_cfg, cfg["NETWORK_SETTINGS"]["n_classes"])
   print("CONSOLE: MODEL INTIALIZED")
   predictor.load_state_dict(torch.load(cfg["INFERENCE_MODE"]["model_weights_path"]))
   print("CONSOLE: MODEL LOADED")
   return predictor


def main():
   config = configparser.ConfigParser()
   config.read(os.path.join(CONFIG_PATH))

   predictor = model_factory(config)

   # If not custom dataset -> MNIST
   if config["NETWORK_SETTINGS"]["custom_dataset"] == "0":
      dataset = utils.FolderDataset("inference", utils.get_mnist_transform(), inference_folder_path = config["NETWORK_SETTINGS"]["inference_data_path"])
      labels_map = torchvision.datasets.MNIST.classes
   else:
      dataset = utils.FolderDataset("inference", 
                                    utils.get_default_transform(*read_size, n_channels = int(config["NETWORK_SETTINGS"]["img_channels"])), 
                                    inference_folder_path = config["NETWORK_SETTINGS"]["inference_data_path"])
      labels_map = utils.FolderDataset().labels_map
   
   inference_dataloader = DataLoader(dataset, int(config["INFERENCE_MODE"]["batch_size"]), False)

   output_path = os.path.join(config["INFERENCE_MODE"]["output_folder"])
   if not os.path.exists(output_path):
      os.mkdir(output_path)

   

   for X in inference_dataloader:
      prediction = predictor(X).softmax(-1).argmax(-1)
      print(labels_map[prediction])



if __name__ == "__main__":
   main()