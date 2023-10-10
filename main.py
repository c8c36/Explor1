import configparser
import os
import torch
from torch.utils.data import DataLoader
import utils
from utils import CONFIG_PATH
import model
import torchvision
from train_model import read_size
from PIL import Image
from torchvision import transforms
toImage = transforms.ToPILImage()

def model_factory(cfg, n_classes):
   net_cfg = cfg["NETWORK_SETTINGS"]["network_size"].lower().strip()
   if net_cfg not in model.configs.keys():
      raise ValueError("network_size error. Expected one of {}".format(model.configs.keys()))
   net_cfg = model.configs[net_cfg]

   predictor = model.ConvNetModel(int(cfg["NETWORK_SETTINGS"]["img_channels"]), net_cfg, n_classes)
   print("CONSOLE: MODEL INTIALIZED")
   predictor.load_state_dict(torch.load(cfg["INFERENCE_MODE"]["model_weights_path"], map_location = model.DEVICE))
   print("CONSOLE: MODEL LOADED")
   return predictor


def main():
   config = configparser.ConfigParser()
   config.read(os.path.join(CONFIG_PATH))

   # If not custom dataset -> MNIST
   if config["PATHS"]["custom_dataset"] == "0":
      dataset = utils.FolderDataset("inference", utils.get_mnist_transform(), inference_folder_path = config["INFERENCE_MODE"]["inference_data_path"])
      labels_map = torchvision.datasets.MNIST.classes
      n_classes = len(torchvision.datasets.MNIST.classes)
   else:
      dataset = utils.FolderDataset("inference", 
                                    utils.get_default_transform(*read_size, n_channels = int(config["NETWORK_SETTINGS"]["img_channels"])), 
                                    inference_folder_path = config["INFERENCE_MODE"]["inference_data_path"])
      labels_map = utils.FolderDataset().get_labels_map()
      n_classes = len(utils.FolderDataset().get_labels_map())
   
   inference_dataloader = DataLoader(dataset, int(config["INFERENCE_MODE"]["batch_size"]), False)

   output_path = os.path.join(config["INFERENCE_MODE"]["output_folder"])
   if not os.path.exists(output_path):
      os.mkdir(output_path)

   predictor = model_factory(config, n_classes)
   predictor.eval()
   predictor.to(model.DEVICE)
   with torch.no_grad():
      for i, X in enumerate(inference_dataloader):
         X = X.to(model.DEVICE)
         predictions = predictor(X).softmax(-1).argmax(-1)
         for j in range(len(predictions)):
            toImage(X[j]).save(os.path.join(output_path, "pic {} {}.png".format((i+1)*(j+1), labels_map[predictions[j]])))

if __name__ == "__main__":
   main()