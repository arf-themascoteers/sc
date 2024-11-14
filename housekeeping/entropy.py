import torch
import torch.nn.functional as F

def entropy(values):
    probs = F.softmax(values, dim=0)
    #print(probs)
    logs = torch.log(probs + 1e-10)
    #print(logs)
    sums = -torch.sum(probs * logs)
    return sums

arrays = [
    torch.tensor([4, 0, 0, 0])*1.0,
    torch.tensor([3, 1, 0, 0])*1.0,
    torch.tensor([2, 1, 1, 0])*1.0,
    torch.tensor([1, 1, 1, 1])*1.0,
]

for i, arr in enumerate(arrays):
    print(f"H({arr.tolist()}) = {entropy(arr):.4f}")
    print("")
    print("")
