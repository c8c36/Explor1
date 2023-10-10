import model
from model import *
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import configparser
import utils
import traceback

config = configparser.ConfigParser()
config.read(utils.CONFIG_PATH)

def read_size():
   height = int(config["NETWORK_SETTINGS"]["img_height"])
   width = int(config["NETWORK_SETTINGS"]["img_width"])
   return (height, width)

def factory_datasets():
   if config["PATHS"]["custom_dataset"] == "0":
      img_transforms = utils.get_mnist_transform()
      train_dataset = datasets.MNIST(os.path.join("data_train"), train = True, transform = img_transforms, download = True)
      test_dataset = datasets.MNIST(os.path.join("data_test"), False, img_transforms, download = True)
      return train_dataset, test_dataset, len(train_dataset.classes)
   else:
      img_transforms = utils.get_default_transform(*read_size(), n_channels = int(config["NETWORK_SETTINGS"]["img_channels"]))
      train_dataset = utils.FolderDataset("train", img_transforms)
      test_dataset = utils.FolderDataset("test", img_transforms)
      return train_dataset, test_dataset, len(train_dataset.labels_map.keys())

def train_loop(dataloader, model, loss_fn, optimizer):
   size = len(dataloader.dataset)
   for batch, (X, y) in enumerate(dataloader):
      X = X.to(DEVICE)
      y = y.to(DEVICE)

      pred = model(X)
      loss = loss_fn(pred, y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch % int(config["NETWORK_SETTINGS"]["print_frequency"]) == 0:
         loss, current = loss.item(), batch * len(X)
         print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
   size = len(dataloader.dataset)
   num_batches = len(dataloader)
   test_loss, correct = 0, 0

   with torch.no_grad():
      for X, y in dataloader:
         X = X.to(DEVICE)
         y = y.to(DEVICE)

         model.eval()
         pred = model(X)
         model.train()
         test_loss += loss_fn(pred, y).item()
         correct += (pred.argmax(1) == y).type(torch.float).sum().item()

   test_loss /= num_batches
   correct /= size
   print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
   # Intitialize datasets
   train_dataset, test_dataset, n_classes = factory_datasets()

   # Initialize loaders
   train_loader = DataLoader(train_dataset, int(config["NETWORK_SETTINGS"]["batch_size"]), True, drop_last = True)
   test_loader = DataLoader(test_dataset, int(config["NETWORK_SETTINGS"]["batch_size"]), True, drop_last = True)

   # Read config
   net_cfg = config["NETWORK_SETTINGS"]["network_size"].lower().strip()
   if net_cfg not in model.configs.keys():
      raise ValueError("network_size error. Expected one of {}".format(model.configs.keys()))
   net_cfg = model.configs[net_cfg]

   # Intitialize network, loss_fn, optimizer, and scheduler
   network = ConvNetModel(int(config["NETWORK_SETTINGS"]["img_channels"]), net_cfg, n_classes)
   loss_fn = nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(network.parameters(), float(config["NETWORK_SETTINGS"]["learning_rate"]))
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(config["NETWORK_SETTINGS"]["epochs"]))

   # Train the network
   try:
      for _ in range(int(config["NETWORK_SETTINGS"]["epochs"])):
         print("Epoch: ", _+1)
         train_loop(train_loader, network, loss_fn, optimizer)
         test_loop(test_loader, network, loss_fn)
         scheduler.step()
   except Exception as e:
      print("General exception encountered")
      traceback.print_exc()
   finally:
      # Save the network
      print("Save model? (y/n)")
      user_input = input()
      if user_input != "n":
         torch.save(network.state_dict(), os.path.join(config["NETWORK_SETTINGS"]["model_save_path"]))
         print("Model saved")

if __name__ == "__main__":
   main()