import configparser
import os
import torch
from torch.utils.data import DataLoader
import utils
from utils import CONFIG_PATH
import model
import torchvision
from train_model import read_size

def model_factory(cfg, n_classes):
   net_cfg = cfg["NETWORK_SETTINGS"]["network_size"].lower().strip()
   if net_cfg not in model.configs.keys():
      raise ValueError("network_size error. Expected one of {}".format(model.configs.keys()))
   net_cfg = model.configs[net_cfg]

   predictor = model.ConvNetModel(int(cfg["NETWORK_SETTINGS"]["img_channels"]), net_cfg, n_classes)
   print("CONSOLE: MODEL INTIALIZED")
   predictor.load_state_dict(torch.load(cfg["INFERENCE_MODE"]["model_weights_path"]))
   print("CONSOLE: MODEL LOADED")
   return predictor


def main():
   config = configparser.ConfigParser()
   config.read(os.path.join(CONFIG_PATH))

   # If not custom dataset -> MNIST
   if config["NETWORK_SETTINGS"]["custom_dataset"] == "0":
      dataset = utils.FolderDataset("inference", utils.get_mnist_transform(), inference_folder_path = config["NETWORK_SETTINGS"]["inference_data_path"])
      labels_map = torchvision.datasets.MNIST.classes
      n_classes = len(torchvision.datasets.MNIST.classes)
   else:
      dataset = utils.FolderDataset("inference", 
                                    utils.get_default_transform(*read_size, n_channels = int(config["NETWORK_SETTINGS"]["img_channels"])), 
                                    inference_folder_path = config["NETWORK_SETTINGS"]["inference_data_path"])
      labels_map = utils.FolderDataset().labels_map
      n_classes = len(utils.FolderDataset().labels_map)
   
   inference_dataloader = DataLoader(dataset, int(config["INFERENCE_MODE"]["batch_size"]), False)

   output_path = os.path.join(config["INFERENCE_MODE"]["output_folder"])
   if not os.path.exists(output_path):
      os.mkdir(output_path)

   predictor = model_factory(config, n_classes)
   predictor.eval()
   predictor.to(model.DEVICE)
   with torch.no_grad():
      for X in inference_dataloader:
         X.to(model.DEVICE)
         prediction = predictor(X).softmax(-1).argmax(-1)
         for image, class_num in zip(X, prediction):
            torchvision.io.write_png(image, os.path.join(output_path, "{}".format(labels_map[class_num])))

if __name__ == "__main__":
   main()