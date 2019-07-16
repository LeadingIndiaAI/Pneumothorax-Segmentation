# [SIIM-ACR-Pneumothorax-Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)
  Abstract--A pneumothorax is an abnormal
  collection of air in the specific pleural space
  between the lung and the chest wall caused
  during chest injuries. Such regions are
  diagnosed by studying X-Ray images containing
  the affected area. However, the task of
  segmenting out the affected region becomes
  cumbersome due to the complex details and
  multi-dimensional features.
## DATASET
Here is the link of [dataset.](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/data)
```
Note:Dataset is downloaded using an api.External official link for direct download may be provided by kaggle in future. 
```
## Prerequisite
Python version used:  python 3.6
### Commands:
* ```pip install tensorflow-gpu==1.8```
* ```pip install pandas```
* ```pip install opencv-python```
* ```pip install keras```
* ```pip install tqdm```
* ```pip install pillow```
script.sh is included in files for linux installation.

## Models
* Mask Rcnn
* Unet
  * Xnet 

## Achievements
| Model | Learning Rate | Accuracy* |
| ----- | -------- | ------- |
| Unet  | 0.000001 | 0.7886  |
| Maskrcnn| 0.0012 | 0.7929  |

 *accuracy is as per submission using dice-coefficient.. more info [here](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview/evaluation)

## Conclusion
- [x] Unet implimentation
- [x] Mask Rcnn imlplimenation
- [ ] To use other models and architecture
## Common error
if you are facing trouble in installing python opencv then this may be helpful
```bash
apt update && apt install -y libsm6 libxext6 libxrender-dev git unzip 
apt-get install libxrender1
```
