## Requirements
The code can runs in the environment as below:
```
numpy==1.25.2
torch==2.0.1
h5py==3.9.0
pandas==2.0.3
fvcore==0.1.5
```
Note that environments with some pacakeges of lower versions may also work. 

## Dataset
Here we provide codes using the datasets come from paper "Comprehensive RF Dataset Collection and Release: A Deep Learning-Based Device Fingerprinting Use Case". The dataset should be placed in the folder `data/`.

Reference of the paper:
```
Elmaghbub, A., & Hamdaoui, B. (2021, December). Comprehensive RF dataset collection and release: A deep learning-based device fingerprinting use case. In 2021 IEEE Globecom Workshops (GC Wkshps) (pp. 1-7). IEEE.
```

## DL Network Hyperparameter Settings Examples
GRNet:
```python
min_width_per_group = 8
self.fusion_module = '' # no fusion module
slice_len_idx = 6 
```

GRNet-H:
```python
min_width_per_group = 8
self.fusion_module = 'rc'
slice_len_idx = 6 
```

GRNet-M:
```python
min_width_per_group = 8
self.fusion_module = 'se'
slice_len_idx = 6 
```

GRNet-L:
```python
min_width_per_group = 2
self.fusion_module = 'rc'
slice_len_idx = 6 
```

# Running Examples
## Training
The model will be trained by defalut settings. Train ,val and test 0 datasets are from 'Outdoor' days 1-4. Test 1 dataset is from 'Outdoor' days 5

```bash
python main.py --task train
```

## Test
### Test 'GRNet-L' in 'Outdoor' by Default:

```bash
python main.py --task test --model_path 'model/GRNet-L.pth'
```

Default output should be like:
```bash
[finish] dateset:0, test snr:[-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35],test acc:[3.755, 3.92, 6.628, 36.692, 59.306, 72.009, 80.915, 88.672, 91.75, 93.803, 94.48, 94.911]
[finish] dateset:1, test snr:[-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35],test acc:[4.035, 4.117, 8.005, 37.238, 56.538, 65.49, 71.159, 75.186, 77.971, 79.989, 80.985, 81.63]
```
The first line is the performance in known environments, while the second is in unknown environments by default.



