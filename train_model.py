from model import *
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os

N_EPOCHS = 5
MODEL_SAVE_LOCATION = "model_weights"

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

    if batch % 10 == 0:
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


img_transforms = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(), transforms.Resize((64, 64), antialias = False), transforms.Normalize([0.5], [0.5])])


train_dataset = datasets.MNIST(os.path.join("data_train"), train = True, transform = img_transforms, download = True)
test_dataset = datasets.MNIST(os.path.join("data_test"), False, img_transforms, download = True)

train_loader = DataLoader(train_dataset, 128, True, drop_last = True)
test_loader = DataLoader(test_dataset, 128, True, drop_last = True)

model = ConvNetModel(1, config_t, len(train_dataset.classes))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS)

for _ in range(N_EPOCHS):
  print("Epoch: ", _+1)
  train_loop(train_loader, model, loss_fn, optimizer)
  test_loop(test_loader, model, loss_fn)
  scheduler.step()

print("Save model? (y/n)")
user_input = input()
if user_input != "n":
   torch.save(model.parameters(), MODEL_SAVE_LOCATION)
   print("Model saved")