# MERA_Image_Classification
## Code Contributor: Fanjie Kong
### TBD:
- The script for training MNIST
- The script for training NeedleMNIST
- The script for training LIDC
- TensorFlow version implementation. I use the tensorflow version for experiments in paper.
#### Finished Work:
1. Implemented 2D MERA model using PyTorch and TensorFlow. TensorFlow version is more time-efficient.
2. Tested our 2D MERA model on MNIST, NeedleMNIST(64x64, 128x128) and LIDC dataset. 

|           	| MNIST 	| NeedleMNIST(64x64) 	| NeedleMNIST(128x128) 	| LIDC  	|
|-----------	|-------	|--------------------	|----------------------	|-------	|
| CNN       	| 0.983 	| 0.760              	| 0.739                	| 0.780 	|
| Tensor-NN 	| 0.985 	| 0.740              	| 0.727                	| 0.860 	|
| 2D MERA   	| 0.903 	| 0.784              	| 0.714                	| 0.760 	|

3. Summarized our work into a paper submitted to QTNML 2020

#### Future Work:
-

#### Description:
##### PyTorch codes:
* Basic Pytorch dependency
* Tested on Pytorch 1.3, Python 3.6 
* Unzip the data and point the path to --data_path
* How to run tests: python train.py --data_path data_location
##### TensorFlow code:
* TensorFlow 2.1.0 and TensorNetwork
* Experiments are performed on Jupyter Notebook

##### Thanks to the following repositories: 
- https://github.com/raghavian/loTeNet_pytorch
