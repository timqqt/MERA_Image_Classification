# MERA_Image_Classification
## Code Contributor: Fanjie Kong
#### Finished Work:
1. Implemented 2D MERA model using PyTorch and TensorFlow. TensorFlow version is more time-efficient.
2. Tested our 2D MERA model on MNIST, NeedleMNIST(64x64, 128x128) and LIDC dataset. 

| ------- | --- | --- | --- | --- |
|         | MNIST | NeedleMNIST(64x64)| NeedleMNIST(128x128)  | LIDC  |
| ------- | --- | --- | --- | --- |
| CNN     | 0.983 | 0.760 | 0.739 | 0.780 |
| ------- | --- | --- | --- | --- |
| Tensor-NN | 0.985 | 0.740 | 0.727 | 0.860 |
| ------- | --- | --- | --- | --- |
| 2D MERA | 0.903 | 0.705 | 0.714 | 0.760 |
| ------- | --- | --- | --- | --- |

