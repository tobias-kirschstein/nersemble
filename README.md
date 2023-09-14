# NeRSemble: Multi-view Radiance Field Reconstruction of Human Heads

[Paper](https://arxiv.org/pdf/2305.03027.pdf) | [Video](https://youtu.be/a-OAWqBzldU) | [Project Page](https://tobias-kirschstein.github.io/nersemble/)

![](static/nersemble_teaser.gif)

[Tobias Kirschstein](https://tobias-kirschstein.github.io/), [Shenhan Qian](https://shenhanqian.github.io), [Simon Giebenhain](https://simongiebenhain.github.io/), [Tim Walter](https://www.linkedin.com/in/tim-walter-7203aa20b/?originalSubdomain=de) and [Matthias Nießner](https://niessnerlab.org/)  
**Siggraph 2023**

# 1. Installation
### 1.1. Dependencies
- PyTorch 2.0
- nerfstudio
- tinycudann


 1. Setup environment
    ```
    conda env create -f environment.yml
    conda activate nersemble
    ```
    which creates a new conda environment `nersemble` (Installation may take a while).


 2. Manually install `tinycudann`:
    ```
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    ```
    (Also helpful, if you get an error like `ImportError: DLL load failed while importing _86_C: The specified procedure could not be found.` later on)


 3. Install the `nersemble` package itself by running 
    ```shell
    pip install -e .
    ```
    inside the cloned repository folder.
    
### 1.2. Environment Paths

All paths to data / models / renderings are defined by _environment variables_.  
Please create a file in your home directory in `~/.config/nersemble/.env` with the following content:
```shell
NERSEMBLE_DATA_PATH="..."
NERSEMBLE_MODELS_PATH="..."
NERSEMBLE_RENDERS_PATH="..."
```
Replace the `...` with the locations where data / models / renderings should be located on your machine.
 - `NERSEMBLE_DATA_PATH`:  Location of the multi-view video dataset (See [section 2](#2-dataset) for how to obtain the dataset)
 - `NERSEMBLE_MODELS_PATH`: During training, model checkpoints and configs will be saved here
 - `NERSEMBLE_RENDERS_PATH`: Video renderings of trained models will be stored here

If you do not like creating a config file in your home directory, you can instead hard-code the paths in the [env.py](src/nersemble/env.py).

### 1.3. Troubleshooting

You may run into this error at the beginning of training:
```shell
\lib\site-packages\torch\include\pybind11\cast.h(624): error: too few arguments for template template parameter "Tuple"
          detected during instantiation of class "pybind11::detail::tuple_caster<Tuple, Ts...> [with Tuple=std::pair, Ts=<T1, T2>]"
(721): here

\lib\site-packages\torch\include\pybind11\cast.h(717): error: too few arguments for template template parameter "Tuple"
          detected during instantiation of class "pybind11::detail::tuple_caster<Tuple, Ts...> [with Tuple=std::pair, Ts=<T1, T2>]"
(721): here
```
This occurs during compilation of `torch_efficient_distloss` and can be solved by either training without 
distortion loss or by changing one line in the `torch_efficient_distloss` library (see [https://github.com/sunset1995/torch_efficient_distloss/issues/8](https://github.com/sunset1995/torch_efficient_distloss/issues/8)).

# 2. Dataset

Access to the dataset can be requested [here](https://forms.gle/rYRoGNh2ed51TDWX9).  
To reproduce the experiments from the paper, only download the `nersemble_XXX_YYY.zip` files (There are 10 in total for the 10 different sequences), as well as the `camera_params.zip`.
Extract these .zip files into `NERSEMBLE_DATA_PATH`.  
Also, see [src/nersemble/data_manager/multi_view_data.py](src/nersemble/data_manager/multi_view_data.py) for an explanation of the folder layout.
# 3. Usage

### 3.1. Training

```shell
python scripts/train/train_nersemble.py $ID $SEQUENCE_NAME --name $NAME
```

where `$ID` is the id of the participant in the dataset (e.g., `030`) and `SEQUENCE_NAME` is the name of the expression / emotion / sentence (e.g., `EXP-2-eyes`).
`$NAME` may optionally be used to annotate the checkpoint folder and the wandb experiment with some descriptive experiment name. 

The training script will place model checkpoints and configuration in `${NERSEMBLE_MODELS_PATH}/nersemble/NERS-XXX-${name}/`. The incremental run id `XXX` will be automatically determined.

#### GPU Requirements
Training takes roughly 1 day and requires at least an RTX A6000 GPU (**48GB VRAM**). GPU memory requirements may be lowered by tweaking some of these hyperparameters:
 - `--max_n_samples_per_batch`: restricts How many ray samples are fed through the model at once (default 20 for 2^20 samples)
 - `--n_hash_encodings`: Number of hash encodings in the ensemble (default 32). Using 16 should give comparable quality (`--latent_dim_time` needs to be set to the same value)
 - `--cone_angle`: Use larger steps between ray samples for further away points. The default value of `0` (no step size increase) provides the best quality. Try values up to `0.004`
 - `--n_train_rays`: Number of rays per batch (default 4096). Lower values can affect convergence
 - `--mlp_num_layers` / `--mlp_layer_width`: Making the deformation field smaller should still provide reasonable performance.

#### RAM requirements
Per default, the training script will cache loaded images in RAM which can cause RAM usage up to 200G. RAM usage can be lowered by:
 - `--max_cached_images` (default 10k): Set to `0` to completely disable caching

#### Special config for sequences 97 and 124

We disable the occupancy grid acceleration structure from Instant NGP as well as the use of distortion loss due to complex hair motion in **sequence 97**:
```shell
python scripts/train/train_nersemble.sh 97 HAIR --name $name --disable_occupancy_grid --lambda_dist_loss 0
```

We only train on a subset of **sequence 124** (timesteps 95-570) and slightly prolong the warmup phase due to the complexity of the sequence:
```shell
 python scripts/train/train_nersemble.sh 124 FREE --name $name --start_timestep 95 --n_timesteps 475 --window_hash_encodings_begin 50000 --window_hash_encodings_end 100000
```
### 3.2. Evaluation

In the paper, all experiments are conducted by training on only 12 cameras and evaluating rendered images on 4 hold-out views (cameras `222200040`, `220700191`, `222200043` and `221501007`).

 - For obtaining the reported **PSNR**, **SSIM** and **LPIPS** metrics (evaluated at 15 evenly spaced timesteps):
    ```shell
    python scripts/evaluate/evaluate_nersemble.py NERS-XXX
    ```
    where `NERS-XXX` is the run name obtained from running the training script above.

 - For obtaining the **JOD video metric** (evaluated at 24fps, takes much longer):
    ```shell
    python scripts/evaluate/evaluate_nersemble.py NERS-XXX --skip_timesteps 3 --max_eval_timesteps -1
    ```

The evaluation results will be printed in the terminal and persisted as a `.json` file in the model folder `${NERSEMBLE_MODELS_PATH}/NERS-XXX-${name}/evaluation`. 

### 3.3. Rendering
From a trained model `NERS-XXX`, a circular trajectory (4s) may be rendered via:
```shell
python scripts/render/render_nersemble.py NERS-XXX
```
The resulting `.mp4` file is stored in `NERSEMBLE_RENDERS_PATH`.

# 4. Trained Models

We provide one trained NeRSemble for each of the 10 sequences used in the paper:

| Participant ID | Sequence                  | Model                                                                            |
|----------------|---------------------------|----------------------------------------------------------------------------------|
| 18             | EMO-1-shout+laugh         | [NERS-9018](https://nextcloud.tobias-kirschstein.de/index.php/s/gQoLTHjQkNNHN2j) |
| 30             | EXP-2-eyes                | [NERS-9030](https://nextcloud.tobias-kirschstein.de/index.php/s/gQoLTHjQkNNHN2j) |
| 38             | EXP-1-head                | [NERS-9038](https://nextcloud.tobias-kirschstein.de/index.php/s/gQoLTHjQkNNHN2j) |
| 85             | SEN-01-port_strong_smokey | [NERS-9085](https://nextcloud.tobias-kirschstein.de/index.php/s/gQoLTHjQkNNHN2j) |
| 97             | HAIR                      | [NERS-9097](https://nextcloud.tobias-kirschstein.de/index.php/s/gQoLTHjQkNNHN2j) |
 | 124            | FREE                      | [NERS-9124](https://nextcloud.tobias-kirschstein.de/index.php/s/gQoLTHjQkNNHN2j) |
| 175            | EXP-6-tongue-1            | [NERS-9175](https://nextcloud.tobias-kirschstein.de/index.php/s/gQoLTHjQkNNHN2j) |
| 226            | EXP-3-cheeks+nose         | [NERS-9226](https://nextcloud.tobias-kirschstein.de/index.php/s/gQoLTHjQkNNHN2j) |
| 227            | EXP-5-mouth               | [NERS-9227](https://nextcloud.tobias-kirschstein.de/index.php/s/gQoLTHjQkNNHN2j) |
| 240            | EXP-4-lips                | [NERS-9240](https://nextcloud.tobias-kirschstein.de/index.php/s/gQoLTHjQkNNHN2j) |

Simply put the downloaded model folders into `${NERSEMBLE_MODELS_PATH}/nersemble`.  
You can then use the `evaluate_nersemble.py` and `render_nersemble.py` scripts to obtain renderings or reproduce the official metrics below. 

# 5. Official metrics

Metrics averaged over all 10 sequences from the NVS benchmark (same 10 sequences as in the paper):

| Model     | PSNR  | SSIM  | LPIPS | JOD  |
|-----------|-------|-------|-------|------|
| NeRSemble | 31.48 | 0.872 | 0.217 | 7.85 |

Note the following:
 - The metrics are slightly different from the paper due to the newer version of nerfstudio used in this repository
 - PSNR, SSIM and LPIPS are computed on only 15 evenly spaced timesteps (to make comparisons cheaper)
 - JOD is computed on every 3rd timestep (using ` --skip_timesteps 3 --max_eval_timesteps -1`)
 - Metrics for sequence 97 were computed with `--no_use_occupancy_grid_filtering`

<hr>

If you find our code, dataset or paper useful, please consider citing
```bibtex
@article{kirschstein2023nersemble,
    author = {Kirschstein, Tobias and Qian, Shenhan and Giebenhain, Simon and Walter, Tim and Nie\ss{}ner, Matthias},
    title = {NeRSemble: Multi-View Radiance Field Reconstruction of Human Heads},
    year = {2023},
    issue_date = {August 2023},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {42},
    number = {4},
    issn = {0730-0301},
    url = {https://doi.org/10.1145/3592455},
    doi = {10.1145/3592455},
    journal = {ACM Trans. Graph.},
    month = {jul},
    articleno = {161},
    numpages = {14},
}
```

Contact [Tobias Kirschstein](mailto:tobias.kirschstein@tum.de) for questions, comments and reporting bugs, or open a GitHub issue.