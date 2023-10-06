from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import pandas as pd

CONFIG_PATH = "config.ini"

class FolderDataset(Dataset):
   def __init__(self, train = False, transform = None, target_transform = None):
      if transform is not None:
         self.transform = transform
      else:
         self.transform = lambda x: x
      
      if target_transform is not None:
         self.target_transform = target_transform
      else:
         self.target_transform = lambda x: x

      self.file = "{}.csv".format("train_data" if train else "test_data")
      print(os.path.join(self.file))
      assert os.path.exists(os.path.join(self.file)), "Run create_csv.py with the system of files from docs"
      self.file = pd.read_csv(self.file)

   def __len__(self):
      return len(self.file)

   def __getitem__(self, idx):
      image = read_image(os.path.join(self.file.iloc[idx, 0]))
      label = self.file.iloc[idx, 1]
      image = self.transform(image)
      label = self.target_transform(label)
      return image, label
   

class InferenceFolder(Dataset):
   def __init__(self, pathing, transform = None):
      if transform is not None:
         self.transform = transform
      else:
         self.transform = lambda x: x
      
      self.pics_path = []
      for pic in os.listdir(os.path.join(pathing)):
         self.pics_path.append(os.path.join(pathing, pic))
      
   def __len__(self):
      return len(self.pics_path)

   def __getitem__(self, idx):
      return self.transform(read_image(self.pics_path[idx]))