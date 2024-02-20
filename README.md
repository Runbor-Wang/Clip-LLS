## :desktop_computer: Requirements

- Pytorch >= 1.13.1
- CUDA >= 11.3
- Other required packages in `requirements.txt`
```
# git clone this repository
git clone https://github.com/ZhexinLiang/CLIP-LIT.git
cd CLIP-LIT

# create new anaconda env
conda create -n CLIP_LIT python=3.7 -y
conda activate CLIP_LIT

# install python dependencies
pip install -r requirements.txt
```

## :running_woman: Inference

### Prepare Testing Data:
You can put the testing images in the `input` folder. If you want to test the backlit images, you can download the BAID test dataset and the Backlit300 dataset from [[Google Drive](https://drive.google.com/drive/folders/1tnZdCxmWeOXMbzXKf-V4HYI4rBRl90Qk?usp=sharing) | [BaiduPan (key:1234)](https://pan.baidu.com/s/1bdGTpVeaHNLWN4uvYLRXXA)].

### Testing:

```
python test.py
```
The path of input images and output images and checkpoints can be changed. 

Example usage:
```
python test.py -i ./Backlit300 -o ./inference_results/Backlit300 -c ./pretrained_models/enhancement_model.pth
```

## :train: Training

### Prepare Training Data and the initial weights:
You should download the backlit and reference image dataset and put it under the repo. In our experiment, we randomly select 380 backlit images from BAID training dataset and 384 well-lit images from DIV2K dataset as the unpaired training data. We provide the training data we use at [[Google Drive](https://drive.google.com/drive/folders/1X1tawqmUsn69T24VmHSl_qmEFxGLzMf0?usp=sharing) | [BaiduPan (key:1234)](https://pan.baidu.com/s/1a0_mUpoFJszjH1eHfBbJPw)] for your reference.

You should also download the initial prompt pair checkpoint (`init_prompt_pair.pth`) from [[Release](https://github.com/ZhexinLiang/CLIP-LIT/releases/tag/v1.0.0) | [Google Drive](https://drive.google.com/drive/folders/1mImPIUaYbXfZ_CHPvdNK-xKrt94abQO5?usp=sharing) | [BaiduPan (key:1234)](https://pan.baidu.com/s/1H4lOrLaYlS0PYTF4pgfSDw)] and put it into `pretrained_models/init_pretrained_models` folder.

After the data and the initial model weights are prepared, you can use the command to change the training data path, fine-tune the prompt and train the model.

If you don't want to download the initial prompt pair, you can train without the initial checkpoints using the command below. But in this way, the number of the total iterations should be at least $50K$ based on our experiments.
 
### Commands
Example usage:
```
python train.py -b ./train_data/BAID_380/resize_input/ -r ./train_data/DIV2K_384/
```
There are other arguments you may want to change. You can change the hyperparameters using the cmd line.

For example, you can use the following command to **train from scratch**.
```
python train.py \
 -b ./train_data/BAID_380/resize_input/ \
 -r ./train_data/DIV2K_384/             \
 --train_lr 0.00002                     \
 --prompt_lr 0.000005                   \
 --eta_min 5e-6                         \
 --weight_decay 0.001                   \
 --num_epochs 3000                      \
 --num_reconstruction_iters 1000        \
 --num_clip_pretrained_iters 8000       \
 --train_batch_size 8                   \
 --prompt_batch_size 16                 \
 --display_iter 20                      \
 --snapshot_iter 20                     \
 --prompt_display_iter 20               \
 --prompt_snapshot_iter 100             \
 --load_pretrain False                  \
 --load_pretrain_prompt False
```
