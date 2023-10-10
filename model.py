#https://arxiv.org/pdf/2201.03545v2.pdf
import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DropPath(nn.Module):
  """
    Given features of shape (B, N, H, W).
    The forward method randomly zeros out a whole feature set along N.

    Args:
      keep_p (float): how much features is left.
      inplace (bool): whether it is inplace operation or not. 
  """
  def __init__(self, keep_p: float = 1.0, inplace: bool = False) -> None:
    super().__init__()
    self.keep_p = keep_p
    self.inplace = inplace
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.training:
      mask_shape = (x.shape[0],) + (1,) * (x.ndim-1)
      mask = x.new_empty(mask_shape).bernoulli(self.keep_p)

      mask.div(self.keep_p)

      return x.mul(mask) if self.inplace else (x * mask)
      
    return x


class LayerNorm2d(nn.Module):
  """
    Classic layer normalization, but for images :)

    Args:
      dim (int): number of features/channels in the input.
      epsilon (float): small value to handle the division by zero cases.
  """
  def __init__(self, dim: int, epsilon: float=1e-6) -> None:
    super().__init__()

    self.gamma = nn.Parameter(torch.ones(dim))
    self.bias = nn.Parameter(torch.zeros(dim))
    self.eps = epsilon

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    u = x.mean(1, keepdim=True)
    s = (x-u).pow(2).mean(1, keepdim=True)
    x = (x-u) / torch.sqrt(s+self.eps)
    
    return x * self.gamma[None, :, None, None] + self.bias[None, :, None, None]
  

class ConvNetblock(nn.Module):
  """
    A whole block for the ConvNext architecture.

    Args:
      channels (int): number of features/channels in.
      keep_prob (float): how much features/channels is left.
      init_scale (float): learnable parameter. Initial scale of the output.
  """
  def __init__(self, channels: int, keep_prob: float=1.0, init_scale: float=1e-6) -> None:
    super().__init__()
    self.dconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
    self.conv1 = nn.Conv2d(channels, channels*4, 1)
    self.conv2 = nn.Conv2d(channels*4, channels, 1)

    self.ln = LayerNorm2d(channels)
    self.act = nn.GELU()
    self.drop = DropPath(keep_prob) if keep_prob != 1.0 else nn.Identity()
    self.scale = (nn.Parameter(init_scale*torch.ones(channels), requires_grad=True)
                  if init_scale > 0 else None) 
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.dconv(x)
    out = self.ln(out)
    out = self.act(self.conv1(out))
    out = self.conv2(out)
    if self.scale is not None:
      out *= self.scale[None, :, None, None]
    return x + self.drop(out)


# Define configurations as in the paper
configs = { 
  "config_t" : [(96, 192, 384, 768), (3, 3, 9, 3)],
  "config_s" : [(96, 192, 384, 768), (3, 3, 27, 3)],
  "config_b" : [(128, 256, 512, 1024), (3, 3, 27, 3)],
  "config_l" : [(192, 384, 768, 1536), (3, 3, 27, 3)],
  "config_xl" : [(256, 512, 1024, 2048), (3, 3, 27, 3)]
}

class ConvNetModel(nn.Module):
  """
    Args:
      img_channels - number of channels in a single image.
      configuration - configuration of the neural network. Possible configurations
                      are included in the above cell.
      n_classes - number of classes for classification task.
    """
  def __init__(self, img_channels: int, configuration, n_classes: int) -> None:
    super().__init__()
    channels, blocks = configuration
    self.down, self.blocks = nn.ModuleList([]), nn.ModuleList([])
    dimensions, repeats = configuration

    # "stem"
    self.down.append(nn.Sequential(
        nn.Conv2d(img_channels, dimensions[0], 4, 4),
        LayerNorm2d(dimensions[0]),
    ))
    # downsample layers
    for i in range(len(dimensions)-1):
      self.down.append(nn.Sequential(
          LayerNorm2d(dimensions[i]),
          nn.Conv2d(dimensions[i], dimensions[i+1], 2, 2)
      ))
    
    for i, repeat in enumerate(repeats):
      block = []
      for _ in range(repeat):
        block.append(ConvNetblock(dimensions[i], 0.5))
      self.blocks.append(nn.Sequential(
          *block
      ))
    
    self.norm = nn.LayerNorm(dimensions[-1])
    self.classificator = nn.Linear(dimensions[-1], n_classes) if n_classes != 0 else nn.Identity()
  
  def forward_extractor(self, x: torch.Tensor) -> torch.Tensor:
    for down, block in zip(self.down, self.blocks):
      x = down(x)
      x = block(x)
    return self.norm(x.mean([-2, -1]))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.classificator(self.forward_extractor(x))