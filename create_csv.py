import os
import traceback
import csv
import configparser
from utils import CONFIG_PATH

def make_csv(folder_path, file_name):
   with open("{}.csv".format(file_name), mode = "w", newline = "") as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(("image", "label"))
      for directory in os.listdir(os.path.join(folder_path)):
         for file in os.listdir(os.path.join(folder_path, directory)):
            writer.writerow((os.path.join(folder_path, directory, file), directory))


if __name__ == "__main__":
   config = configparser.ConfigParser()
   config.read(CONFIG_PATH)
   assert config["PATHS"]["TRAINING_DATA_PATH"] != "", "Enter the path to train data folder in {}".format(CONFIG_PATH)
   assert config["PATHS"]["TEST_DATA_PATH"] != "", "Enter the path to test data folder in {}".format(CONFIG_PATH)

   config.set("NETWORK_SETTINGS", "N_CLASSES", "{}".format(len(os.listdir(os.path.join(config["PATHS"]["TRAINING_DATA_PATH"])))))
   with open(os.path.join(CONFIG_PATH), "w") as f:
      config.write(f)

   make_csv(config["PATHS"]["TRAINING_DATA_PATH"], "train_data")
   make_csv(config["PATHS"]["TEST_DATA_PATH"], "test_data")