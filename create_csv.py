import os
pathify = lambda x: os.path.join(x)
import csv
import configparser
from utils import CONFIG_PATH, CORE_FOLDER

def make_csv(in_folder_path, out_folder_path, file_name):
   with open(os.path.join(out_folder_path, "{}.csv".format(file_name)), mode = "w", newline = "") as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(("image", "label"))
      for class_idx, directory in enumerate(os.listdir(os.path.join(in_folder_path))):
         for file in os.listdir(os.path.join(in_folder_path, directory)):
            writer.writerow((os.path.join(in_folder_path, directory, file), class_idx))

if __name__ == "__main__":
   if not os.path.exists(CORE_FOLDER):
      os.mkdir(CORE_FOLDER)

   # Read config and check for custom training/testing data path
   config = configparser.ConfigParser()
   config.read(CONFIG_PATH)
   if not os.path.exists(pathify(config["PATHS"]["train_data_path"])):
      assert False, "Enter the path to train data folder (train_data_path field) in {}".format(CONFIG_PATH)

   if not os.path.exists(pathify(config["PATHS"]["test_data_path"])):
      assert False, "Enter the path to test data folder (test_data_path field) in {}".format(CONFIG_PATH)

   # collect and write down the number of classes detected to config.ini
   config.set("NETWORK_SETTINGS", "n_classes", "{}".format(len(os.listdir(pathify(config["PATHS"]["train_data_path"])))))
   with open(pathify(CONFIG_PATH), "w") as f:
      config.write(f)

   # Write down the name of each class to class_names.csv
   with open(os.path.join(CORE_FOLDER, "class_names.csv"), mode = "w", newline = "") as csvfile:
      writer = csv.writer(csvfile)
      class_names = [["class"]]
      for class_name in os.listdir(pathify(config["PATHS"]["train_data_path"])):
         class_names.append([class_name])
      writer.writerows(class_names)

   make_csv(config["PATHS"]["train_data_path"], CORE_FOLDER, "train_data")
   make_csv(config["PATHS"]["test_data_path"], CORE_FOLDER, "test_data")