import configparser
import os
import traceback
import torch
import utils
from utils import CONFIG_PATH
import model
import torchvision
from torchvision import transforms
from train_model import read_size

def model_factory(cfg):
   if cfg["NETWORK_SETTINGS"]["NETWORK_SIZE"] == "config_t":
      net_cfg = model.config_t
   elif cfg["NETWORK_SETTINGS"]["NETWORK_SIZE"] == "config_s": 
      net_cfg = model.config_s
   elif cfg["NETWORK_SETTINGS"]["NETWORK_SIZE"] == "config_b": 
      net_cfg = model.config_b
   elif cfg["NETWORK_SETTINGS"]["NETWORK_SIZE"] == "config_l": 
      net_cfg = model.config_l
   elif cfg["NETWORK_SETTINGS"]["NETWORK_SIZE"] == "config_xl":
      net_cfg = model.config_xl
   else:
      raise ValueError()

   predictor = model.ConvNetModel(int(cfg["NETWORK_SETTINGS"]["IMG_CHANNELS"]), net_cfg, cfg["NETWORK_SETTINGS"]["N_CLASSES"])
   print("CONSOLE: MODEL INTIALIZED")
   predictor.load_state_dict(torch.load(cfg["INFERENCE_MODE"]["MODEL_WEIGHTS_PATH"]))
   print("CONSOLE: MODEL LOADED")
   return predictor


def main():
   config = configparser.ConfigParser()
   config.read(os.path.join(CONFIG_PATH))
   
   assert config["NETWORK_SETTINGS"]["N_CLASSES"] != "", "Run the create_csv.py or specify the number of classes in {}".format(CONFIG_PATH)

   if "INFERENCE_MODE" in config.sections():
      predictor = model_factory(config)
   else:
      raise configparser.NoSectionError("INFERENCE_MODE")

   if config["NETWORK_SETTINGS"]["custom_dataset"] == "0":
      dataset = utils.InferenceFolder("example_data", 
                                      transforms.Compose([transforms.ToTensor(), 
                                                          transforms.Grayscale(), 
                                                          transforms.Resize((64, 64), antialias = False), 
                                                          transforms.Normalize([0.5], [0.5])]))
   else:
      dataset = utils.InferenceFolder(config["INFERENCE_MODE"]["inference_data_path"],
                                      transforms.Compose([transforms.ToTensor(), 
                                                          transforms.Resize(read_size(), antialias = False), 
                                                          transforms.Normalize([0.5 for i in range(int(config["NETWORK_SETTINGS"]["IMG_CHANNELS"]))],
                                                                               [0.5 for i in range(int(config["NETWORK_SETTINGS"]["IMG_CHANNELS"]))])]))
   
   for i in range(len(dataset)):
      torchvision.io.write_png(dataset[i], os.path.join())

if __name__ == "__main__":
   try:
      main()
   except FileNotFoundError as fe:
      print("Error. Config file not found. Expected {}".format(CONFIG_PATH))
      traceback.print_exc()
   
   except configparser.NoSectionError as cfge:
      print("Error. No such section. INFERENCE_MODE section not found in the config file provided")
      traceback.print_exc()
   
   except ValueError as ve:
      print("Error. Wrong network config value. Verify the spelling of NETWORK_SIZE field in NETWORK_SETTINGS section of {}".format(CONFIG_PATH))
   
   except Exception as e:
      print("Error. GENERAL EXCEPTION CAUGHT")
      traceback.print_exc()