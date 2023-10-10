from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import pandas as pd
import csv
from torchvision import transforms

CONFIG_PATH = "config.ini"
CORE_FOLDER = "temp"

def get_default_transform(height, width, n_channels, *additional_transforms):
   return transforms.Compose([transforms.ToTensor(), 
                                          transforms.Resize((height, width), antialias = False), 
                                          transforms.Normalize([0.5 for i in range(n_channels)], [0.5 for i in range(n_channels)]), 
                                          *additional_transforms])

def get_mnist_transform(*additional_transforms):
   return transforms.Compose([transforms.ToTensor(), 
                              transforms.Grayscale(), 
                              transforms.Resize((64, 64), antialias = False), 
                              transforms.Normalize([0.5], [0.5]), 
                              *additional_transforms])

class FolderDataset(Dataset):
   def _test_train_get_item(self, idx):
      image = read_image(os.path.join(self.file.iloc[idx, 0]))
      label = self.file.iloc[idx, 1]
      image = self.transform(image)
      label = self.target_transform(label)
      return image, label
   
   def _inference_get_item(self, idx):
      image = read_image(os.path.join(self.file[idx]))
      return self.transform(image)

   def __init__(self, regime = "train", transform = None, target_transform = None, inference_folder_path = None):
      regimes = ["train", "inference", "test"]
      if regime not in regimes:
         raise ValueError("Invalid regime. Expected one of: {}".format(regimes))

      if transform is not None:
         self.transform = transform
      else:
         self.transform = lambda x: x
      
      if target_transform is not None:
         self.target_transform = target_transform
      else:
         self.target_transform = lambda x: x


      if regime == "train" or regime == "test":
         self.file = "{}.csv".format("train_data" if regime == "train" else "test_data")
         self.file = os.path.join(CORE_FOLDER, self.file)
         assert os.path.exists(self.file), "Run create_csv.py to create csv files for data pathing"
         self.get_item_function = self._test_train_get_item
         self.file = pd.read_csv(self.file)
      
      else:
         if inference_folder_path is None:
            raise ValueError("Enter inference_folder_path to use inference regime")
         
         self.get_item_function = self._inference_get_item
         self.file = []
         for pic in os.listdir(os.path.join(inference_folder_path)):
            self.file.append(os.path.join(inference_folder_path, pic))
      
      self.labels_map = self.get_labels_map()

   def __len__(self):
      return len(self.file)

   def __getitem__(self, idx):
      return self.get_item_function(idx)
   
   def get_labels_map(self):
      class_names = os.path.join(CORE_FOLDER, "class_names.csv")
      output = {}
      assert os.path.exists(class_names), "Error. The program was not able to locate class_names.csv in {}".format(CORE_FOLDER)
      with open(class_names, mode = "r", newline="") as csvfile:
         reader = csv.reader(csvfile)
         idx = 0
         for row in reader:
            row = row[0]
            if row == "class":
               continue
            output[idx] = row
            idx += 1
      
      return output