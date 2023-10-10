# Explor1
To use pretrained weights: download from the [link](https://drive.google.com/file/d/1ISIjILjfet2XyBe-wrFzD0WBavlWkDEV/view?usp=sharing) pretrained weights, and drop them into the same folder where main.py is located<br>
CS2613 Exploration Project 1
To simply run the demo of the model on basic settings - run main.py after downloading pretrained weights<br>
See the wiki page for more details.<br>
Important note on file structure:<br>
```
train_folder
|----------class_name_folder1 <- training pictures of class 1 go here
|----------class_name_folder2 <- training pictures of class 2 go here
...
|----------class_name_foldern <- training pictures of class n go here
test_folder
|----------class_name_folder1 <- testing pictures of class 1 go here
|----------class_name_folder2 <- testing pictures of class 2 go here
...
|----------class_name_foldern <- testing pictures of class 3 go here
```
**NOTE: demo_test_data and demo_train_data DOES NOT CONTAIN PROPER DATA. IT IS AN EXAMPLE OF STRUCTURE THAT MUST BE FOLLOWED**<br>
**ADDITIONALLY, IT SERVES AS A TEST FOR create_csv.py**<br>
Dependencies:
```
matplotlib 3.7.2
torch  2.0.1+cpu
torchvision  0.15.2+cpu
pandas  2.1.1
```