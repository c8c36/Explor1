import model
from model import *
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import configparser
import utils

config = configparser.ConfigParser()
config.read(utils.CONFIG_PATH)

def read_size():
  height = int(config["NETWORK_SETTINGS"]["IMG_HEIGHT"])
  width = int(config["NETWORK_SETTINGS"]["IMG_WIDTH"])
  return (height, width)

def factory_datasets():
  if config["PATHS"]["CUSTOM_DATASET"] == "0":
    img_transforms = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(), transforms.Resize((64, 64), antialias = False), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(os.path.join("data_train"), train = True, transform = img_transforms, download = True)
    test_dataset = datasets.MNIST(os.path.join("data_test"), False, img_transforms, download = True)
    return train_dataset, test_dataset
  else:
    img_transforms = transforms.Compose([transforms.ToTensor(), 
                                         transforms.Resize(read_size(), antialias = False), 
                                         transforms.Normalize([0.5 for i in range(int(config["NETWORK_SETTINGS"]["IMG_CHANNELS"]))],
                                                              [0.5 for i in range(int(config["NETWORK_SETTINGS"]["IMG_CHANNELS"]))])])
    train_dataset = utils.FolderDataset(True)
    test_dataset = utils.FolderDataset()
    return train_dataset, test_dataset

# Required loops
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

    if batch % int(config["NETWORK_SETTINGS"]["PRINT_FREQUENCY"]) == 0:
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

train_dataset, test_dataset = factory_datasets()

train_loader = DataLoader(train_dataset, int(config["NETWORK_SETTINGS"]["batch_size"]), True, drop_last = True)
test_loader = DataLoader(test_dataset, int(config["NETWORK_SETTINGS"]["batch_size"]), True, drop_last = True)

if config["NETWORK_SETTINGS"]["NETWORK_SIZE"] == "config_t":
    net_cfg = model.config_t
elif config["NETWORK_SETTINGS"]["NETWORK_SIZE"] == "config_s": 
    net_cfg = model.config_s
elif config["NETWORK_SETTINGS"]["NETWORK_SIZE"] == "config_b": 
    net_cfg = model.config_b
elif config["NETWORK_SETTINGS"]["NETWORK_SIZE"] == "config_l": 
    net_cfg = model.config_l
elif config["NETWORK_SETTINGS"]["NETWORK_SIZE"] == "config_xl":
    net_cfg = model.config_xl
else:
   raise ValueError()

network = ConvNetModel(1, net_cfg, len(train_dataset.classes))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), 5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(config["NETWORK_SETTINGS"]["N_EPOCHS"]))

for _ in range(int(config["NETWORK_SETTINGS"]["N_EPOCHS"])):
  print("Epoch: ", _+1)
  train_loop(train_loader, network, loss_fn, optimizer)
  test_loop(test_loader, network, loss_fn)
  scheduler.step()

print("Save model? (y/n)")
user_input = input()
if user_input != "n":
   torch.save(network.parameters(), os.path.join(config["NETWORK_SETTINGS"]["MODEL_SAVE_PATH"]))
   print("Model saved")