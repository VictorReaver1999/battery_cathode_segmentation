# Automated Nanoscale Segmentation and Classification of Battery Material Components Using SAM and Convolutional Neural Networks

**MSc Thesis — Hamburg University of Technology (TUHH), 2025**  
**Author: Fadel Victor Shanaa**

---

## Overview

This repository contains the code developed for my MSc thesis on quantitative microstructure analysis of a composite battery cathode material consisting of **lithium iron phosphate (LiFePO₄)**, **carbon black**, and **NaCl**.

The core challenge: given high-resolution grayscale 2D SEM images of this composite, segment and classify each individual material phase — then use the segmented images to train a neural network capable of doing the same on unseen data, perform material analysis, and reconstruct the 3D microstructure.

---

## Pipeline Overview

```
2D SEM Images
      │
      ▼
Unsupervised Segmentation (DBSCAN / GMM / K-Means)
      +
Segment Anything Model (SAM) + Manual Labelling
      │
      ▼
Labelled / Segmented Images
      │
      ▼
ConvNet Training & Inference on new SEM images
      │
      ▼
2D Material Analysis
      │
      ▼
SliceGAN 3D Reconstruction* → 3D Material Analysis*
      │
      ▼
Cross-comparison of 2D vs 3D results
```

*SliceGAN reconstruction code and 3D analysis were lost due to storage failure and are not included in this repository. All results are documented in the thesis PDF.

---

## Repository Contents

| File / Folder | Description |
|---|---|
| `autoencoder_debugged.py` | Autoencoder-only CNN with bilinear upsampling |
| `gmm_autoencoder.py` | GMM segmentation with autoencoder integration |
| `k_means_with_auto_encoder_masks.py` | K-Means segmentation using autoencoder-derived masks |
| `NeuralNet.py` | ConvNet architecture for training and inference |
| `simple_k_means_with_masks.py` | Baseline K-Means segmentation with mask output |
| `mask_visualizer.py` | Visualisation utility for segmentation masks |
| `mask_visualizer_kmeans.py` | K-Means specific mask visualiser |
| `processed_images_with_aspect_preserved_gmm/` | GMM-processed output images |
| `processed_images_with_aspect_preserved_kmeans/` | K-Means-processed output images |

---

## Methods

### Unsupervised Segmentation
Three classical clustering approaches were applied to the grayscale SEM images to separate material phases based on pixel intensity and texture:
- **K-Means** — baseline clustering
- **Gaussian Mixture Models (GMM)** — probabilistic soft assignment
- **DBSCAN** — density-based, no assumed cluster count

### Segment Anything Model (SAM)
Meta's SAM was used alongside manual labelling to produce higher-quality ground truth segmentations. Given the complexity of the composite (carbon black is a diffuse matrix, making purely intensity-based separation unreliable), manual labelling was required for a portion of the dataset.

### ConvNet
A convolutional neural network was trained on the segmented/labelled images and used for inference on new SEM data, automating the classification pipeline.

---

## Dataset

- **Modality:** Grayscale SEM images (high resolution)
- **Material system:** LiFePO₄ / carbon black / NaCl composite cathode
- **Note:** Raw dataset is not included in this repository due to size and partial data loss. Processed output images are included in the `processed_images_*` folders.

---

## Known Limitations

- Carbon black segmentation is inherently difficult from grayscale SEM alone — elemental contrast (EDS/EDX) would improve results significantly
- Manual labelling at particle scale across large SEM images is time-intensive and introduces labeller variability
- SliceGAN 3D reconstruction code was lost and is not reproducible from this repository
- Results showed overclassification in some material phases — documented and discussed in the thesis

---

## Thesis

Full thesis available on Zenodo: *[10.5281/zenodo.20586116]*

---

## License

MIT License — free to use with attribution.

## Citation

If you use this code or methodology, please cite:

> Shanaa, F.V. (2025). *Automated Nanoscale Segmentation and Classification of Battery Material Components Using SAM and Convolutional Neural Networks (TUHH). [10.5281/zenodo.20586116]