# M²: Meshed-Memory Transformer with custom modifications
This repository contains the reference code for the paper _[Meshed-Memory Transformer for Image Captioning](https://arxiv.org/abs/1912.08226)_ (CVPR 2020).

Please cite with the following BibTeX:

```
@inproceedings{cornia2020m2,
  title={{Meshed-Memory Transformer for Image Captioning}},
  author={Cornia, Marcella and Stefanini, Matteo and Baraldi, Lorenzo and Cucchiara, Rita},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
<p align="center">
  <img src="images/m2.png" alt="Meshed-Memory Transformer" width="320"/>
</p>

## Environment setup
Clone the repository and create the `m2release` conda environment using the `environment_custom.yml` file:
```
conda env create -f environment_custom.yml
conda activate m2release
```

Then download spacy data by executing the following command:
```
python -m spacy download en
```

Note: Python 3.6 is required to run the code. 


## Data preparation
To run the code, annotations and detection features for the COCO dataset are needed. Please download the annotations file [annotations.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing) and extract it. 

Detection features are computed with the code provided by [1]. To reproduce authors' result, please download the COCO features file [coco_detections.hdf5](https://drive.google.com/open?id=1MV6dSnqViQfyvgyHrmAT_lLpFbkzp3mx) (~53.5 GB), in which detections of each image are stored under the `<image_id>_features` key. `<image_id>` is the id of each COCO image, without leading zeros (e.g. the `<image_id>` for `COCO_val2014_000000037209.jpg` is `37209`), and each value should be a `(N, 2048)` tensor, where `N` is the number of detections. If you want to use your own detections just build your own `Features.hdf5` file following the same format.


## Evaluation
To reproduce the results reported in the paper, download the pretrained model file [meshed_memory_transformer.pth](https://drive.google.com/file/d/1naUSnVqXSMIdoiNz_fjqKwh9tXF7x8Nx/view?usp=sharing) and place it in the code folder, or use your own saved model to see your results.

Run `python test_custom.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations (default: 'annotations') |
| `--weights` | Path to pretrained or custom weights (default: 'meshed_memory_transformer.pth') |
| `--d_in` | Dimensionality of region features (default: 2048) |
| `--vocab` | Path to model vocabulary (default: 'vocab.pkl') | 

#### Expected output
Under `output_logs/`, you may also find the expected output of the evaluation code for pretrained model with pre-extracted features.


## Training procedure
Please create the folder `saved_models` inside the code folder before start training. Then run `python train_custom.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name (default: 'm2_transformer') |
| `--batch_size` | Batch size (default: 50) |
| `--workers` | Number of workers (default: 0) |
| `--m` | Number of memory vectors (default: 40) |
| `--head` | Number of heads (default: 8) |
| `--warmup` | Warmup value for learning rate scheduling (default: 10000) |
| `--resume_last` | If used, the training will be resumed from the last checkpoint |
| `--resume_best` | If used, the training will be resumed from the best checkpoint |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations (default: 'annotations') |
| `--logs_folder` | Path folder for tensorboard logs (default: 'tensorboard_logs')|
| `--d_in` | Dimensionality of region features (default: 2048) |
| `--random` | If used, training epochs are capped at 15 even if patience is not reached |
| `--scst` | If used, training with scst is enabled, otherwise only xe stage is carried out |
| `--buil_vocab` | If used, a new vocabulary is built, otherwise the pre-built vocabulary is used |

For example, to train the model with the parameters used in authors' experiments, use
```
python train_custom.py --features_path /path/to/features --annotation_folder /path/to/annotations --scst
```

<p align="center">
  <img src="images/results.png" alt="Sample Results" width="850"/>
</p>

#### References
[1] P. Anderson, X. He, C. Buehler, D. Teney, M. Johnson, S. Gould, and L. Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_, 2018.
