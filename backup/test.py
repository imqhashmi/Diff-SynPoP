import torch

# Creating two tensors
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# Concatenating along dimension 0 (rows)
concatenated_tensor = torch.cat((tensor1, tensor2), dim=-1)

# Printing concatenated tensor
print(concatenated_tensor)