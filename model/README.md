---
license: mit
datasets:
- Somayeh-h/Nordland
- OPR-Project/OxfordRobotCar_OpenPlaceRecognition
language:
- en
metrics:
- recall_at_1
- recall_at_5
pipeline_tag: image-feature-extraction
tags:
- place-recognition
- visual-place-recognition
- computer-vision
- transformer
- 3d-vision
library:
- pytorch
- lightning
---

# Model Card for UniPR-3D

UniPR-3D is a universal visual place recognition (VPR) framework that supports both **single-frame** and **sequence-to-sequence** matching. It leverages **3D visual geometry grounded tokens** within a transformer architecture to produce robust, viewpoint-invariant descriptors for long-term place recognition under challenging environmental variations (e.g., seasonal, weather, lighting, and viewpoint changes).

## Model Details

### Model Description

- **Developed by:** Tianchen Deng, Xun Chen, Ziming Li, Hongming Shen, Danwei Wang, Javier Civera, Hesheng Wang  
- **Shared by:** Tianchen Deng  
- **Model type:** Vision Transformer with 3D-aware token aggregation for visual place recognition  
- **Language(s):** English (dataset metadata); model is vision-only  
- **License:** MIT  

### Model Sources

- **Repository:** [repo](https://github.com/dtc111111/UniPR-3D)  
- **Paper:** [UniPR-3D: Towards Universal Visual Place Recognition with 3D Visual Geometry Grounded Transformer](https://arxiv.org/abs/2512.21078) (arXiv:2512.21078, 2025)  
- **Demo:** No demo available

## Uses

### Direct Use

This model can be used **out-of-the-box** to extract compact, discriminative global descriptors from:
- Single RGB images (for frame-to-frame VPR)
- Sequences of images (for sequence-to-sequence VPR)

These descriptors are suitable for large-scale localization, robot navigation, and SLAM systems requiring robustness to appearance changes.

### Downstream Use

- Integration into **visual SLAM** or **long-term autonomous navigation** pipelines  
- Replacement for traditional VPR backbones (e.g., NetVLAD, MixVPR, EigenPlaces)  
- Fine-tuning on domain-specific datasets (e.g., underground, aerial, or underwater environments)

### Out-of-Scope Use

- **Not intended** for real-time inference on low-power embedded devices without optimization (latency ~8.23 ms on RTX 4090)  
- **Not designed** for non-visual modalities (e.g., LiDAR, audio, text)  
- Performance may degrade in **extreme occlusion**, **textureless scenes**, or **indoor environments not seen during training**

## Bias, Risks, and Limitations

- Trained primarily on **urban street-level imagery** (GSV-Cities, Mapillary MSLS), so generalization to rural, indoor, or non-Western cities may be limited  
- Inherits biases from training data (e.g., geographic overrepresentation of North America/Europe)  
- No explicit fairness or demographic considerations (as it is a geometric vision model)

### Recommendations

- Evaluate on target domain before deployment  
- Monitor recall performance on your specific dataset using standard VPR metrics (R@1, R@5)

## How to Get Started with the Model

The exact inference script is provided in the GitHub repo (`eval_lora.py`, `main_ft.py`). Pretrained weights are available on Hugging Face or via the repo release.

## Training Details

### Training Data

- **Single-frame model**: Trained on [GSV-Cities](https://github.com/amaralibey/gsv-cities)  
- **Multi-frame model**: Trained on [Mapillary Street-Level Sequences (MSLS)](https://www.mapillary.com/dataset/places)  
- Both datasets contain millions of geo-tagged urban street-view images across diverse cities, seasons, and conditions.

### Training Procedure

#### Preprocessing
- Images resized to 518×518  
- Sequences sampled with spatial proximity for multi-frame training

#### Training Hyperparameters
- **Backbone**: DINOv2 (ViT-large)  
- **Optimization**: AdamW, learning rate scheduling  
- **Loss**: Multi-similarity loss with pair weighting  
- **Training regime**: Mixed-precision (fp16) on NVIDIA GPUs

#### Speeds, Sizes, Times
- **Inference latency**: Single frame - 8.23 ms per image (RTX 4090)  
- **Descriptor dimension**: 17152 (for UniPR-3D)  
- Training time: Not disclosed (multi-day runs on multi-GPU setup)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data
- Single frame evaluation:
  - <a href="https://codalab.lisn.upsaclay.fr/competitions/865">MSLS Challenge</a>, where you upload your predictions to their server for evaluation.
  - Single-frame <a href="https://www.mapillary.com/dataset/places">MSLS</a> Validation set
  - Nordland dataset, <a href="https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Pittsburgh250k/">Pittsburgh</a> dataset and SPED dataset, you may download them from <a href="https://surfdrive.surf.nl/index.php/s/sbZRXzYe3l0v67W">here</a>, aligned with DINOv2 SALAD.
- Multi-frame evaluation:
  - Multi-frame <a href="https://www.mapillary.com/dataset/places">MSLS</a> Validation set
  - Two sequence from <a href="https://robotcar-dataset.robots.ox.ac.uk/datasets/">Oxford RobotCar</a>, you may download them <a href="https://entuedu-my.sharepoint.com/personal/heshan001_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fheshan001%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2Fcasevpr%5Fdatasets%2Foxford%5Frobotcar&viewid=e5dcb0e9%2Db23f%2D44cf%2Da843%2D7837d3064c2e&ga=1">here</a>.
    - 2014-12-16-18-44-24 (winter night) query to 2014-11-18-13-20-12 (fall day) db
    - 2014-11-14-16-34-33 (fall night) query to 2015-11-13-10-28-08 (fall day) db
  - <a href="https://github.com/gmberton/VPR-datasets-downloader/blob/main/download_nordland.py">Nordland (filtered) dataset</a>

#### Factors
- Seasonal variation (summer ↔ winter)
- Day vs. night
- Weather (sunny, rainy, snowy)
- Viewpoint change (lateral shift, orientation)

#### Metrics
- **Recall@K (R@1, R@5, R@10)**: Standard metric for VPR – fraction of queries with correct match in top-K retrieved database images

### Results

#### Summary

Our method achieves significantly higher recall than competing approaches, achieving new state-of-the-art performance on both single and multiple frame benchmarks.
##### Single-frame matching results

<style>
  table, th, td {
    border-collapse: collapse;
    text-align: center;
  }
</style>
<table>
  <tr>
    <th colspan="2"></th>
    <th colspan="2">MSLS Challenge</th>
    <th colspan="2">MSLS Val</th>
    <th colspan="2">NordLand</th>
    <th colspan="2">Pitts250k-test</th>
    <th colspan="2">SPED</th>
  </tr>
  <tr>
    <th>Method</th>
    <th>Latency (ms)</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@1</th>
    <th>R@5</th>
  </tr>
  <tr>
    <td>MixVPR</td>
    <td>1.37</td>
    <td>64.0</td>
    <td>75.9</td>
    <td>88.0</td>
    <td>92.7</td>
    <td>58.4</td>
    <td>74.6</td>
    <td>94.6</td>
    <td><u>98.3</u></td>
    <td>85.2</td>
    <td>92.1</td>
  </tr>
  <tr>
    <td>EigenPlaces</td>
    <td>2.65</td>
    <td>67.4</td>
    <td>77.1</td>
    <td>89.3</td>
    <td>93.7</td>
    <td>54.4</td>
    <td>68.8</td>
    <td>94.1</td>
    <td>98.0</td>
    <td>69.9</td>
    <td>82.9</td>
  </tr>
  <tr>
    <td>DINOv2 SALAD</td>
    <td>2.41</td>
    <td><u>73.0</u></td>
    <td><u>86.8</u></td>
    <td><u>91.2</u></td>
    <td><u>95.3</u></td>
    <td><u>69.6</u></td>
    <td><u>84.4</u></td>
    <td><u>94.5</u></td>
    <td><b>98.7</b></td>
    <td><u>89.5</u></td>
    <td><u>94.4</u></td>
  </tr>
  <tr>
    <td>UniPR-3D (ours)</td>
    <td>8.23</td>
    <td><b>74.3</b></td>
    <td><b>87.5</b></td>
    <td><b>91.4</b></td>
    <td><b>96.0</b></td>
    <td><b>76.2</b></td>
    <td><b>87.3</b></td>
    <td><b>94.9</b></td>
    <td>98.1</td>
    <td><b>89.6</b></td>
    <td><b>94.5</b></td>
  </tr>
</table>

##### Sequence matching results

<table>
  <tr>
    <th></th>
    <th colspan="3">MSLS Val</th>
    <th colspan="3">NordLand</th>
    <th colspan="3">Oxford1</th>
    <th colspan="3">Oxford2</th>
  </tr>
  <tr>
    <th>Method</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
  </tr>
  <tr>
    <td>SeqMatchNet</td>
    <td>65.5</td>
    <td>77.5</td>
    <td>80.3</td>
    <td>56.1</td>
    <td>71.4</td>
    <td>76.9</td>
    <td>36.8</td>
    <td>43.3</td>
    <td>48.3</td>
    <td>27.9</td>
    <td>38.5</td>
    <td>45.3</td>
  </tr>
  <tr>
    <td>SeqVLAD</td>
    <td>89.9</td>
    <td>92.4</td>
    <td>94.1</td>
    <td>65.5</td>
    <td>75.2</td>
    <td>80.0</td>
    <td>58.4</td>
    <td>72.8</td>
    <td>80.8</td>
    <td>19.1</td>
    <td>29.9</td>
    <td>37.3</td>
  </tr>
  <tr>
    <td>CaseVPR</td>
    <td><u>91.2</u></td>
    <td><u>94.1</u></td>
    <td><u>95.0</u></td>
    <td><u>84.1</u></td>
    <td><u>89.9</u></td>
    <td><u>92.2</u></td>
    <td><u>90.5</u></td>
    <td><u>95.2</u></td>
    <td><u>96.5</u></td>
    <td><u>72.8</u></td>
    <td><u>85.8</u></td>
    <td><u>89.9</u></td>
  </tr>
  <tr>
    <td>UniPR-3D (ours)</td>
    <td><b>93.7</b></td>
    <td><b>95.7</b></td>
    <td><b>96.9</b></td>
    <td><b>86.8</b></td>
    <td><b>91.7</b></td>
    <td><b>93.8</b></td>
    <td><b>95.4</b></td>
    <td><b>98.1</b></td>
    <td><b>98.7</b></td>
    <td><b>80.6</b></td>
    <td><b>90.3</b></td>
    <td><b>93.9</b></td>
  </tr>
</table>


## Compute Infrastructure

### Hardware
- NVIDIA RTX 4090

### Software
- Python 3.11.10 + CUDA 12.1  
- Based on [SALAD](https://github.com/serizba/salad) and [VGGT](https://github.com/facebookresearch/vggt)

## Citation

**BibTeX:**
```bibtex
@article{deng2025unipr3d,
  title={UniPR-3D: Towards Universal Visual Place Recognition with 3D Visual Geometry Grounded Transformer},
  author={Deng, Tianchen and Chen, Xun and Li, Ziming and Shen, Hongming and Wang, Danwei and Civera, Javier and Wang, Hesheng},
  journal={arXiv preprint arXiv:2512.21078},
  year={2025}
}
```

**APA:**
Deng, T., Chen, X., Li, Z., Shen, H., Wang, D., Civera, J., & Wang, H. (2025). UniPR-3D: Towards Universal Visual Place Recognition with 3D Visual Geometry Grounded Transformer. *arXiv preprint arXiv:2512.21078*.

## Contact

For questions, pretrained model access, or qualitative comparisons, please contact:  
📧 **Tianchen Deng** – [dengtianchen@sjtu.edu.cn](mailto:dengtianchen@sjtu.edu.cn)

---

> 📌 **Acknowledgement**: This implementation builds upon [SALAD](https://github.com/serizba/salad) and [VGGT](https://github.com/facebookresearch/vggt). Please cite those works if you use their components.