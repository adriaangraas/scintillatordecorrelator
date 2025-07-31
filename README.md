# Scintillator Decorrelator

This repository contains code and Jupyter notebook examples for the publication:

> *Scintillator decorrelation for self-supervised X-ray radiograph denoising*
> [📄 DOI: 10.1088/1361-6501/addc06](https://doi.org/10.1088/1361-6501/addc06)

The article describes how to remove correlations from X-ray radiographs acquired by e.g. Caesium-Iodine scintillator detectors, via PRF estimation and deconvolution. This enables self-supervised radiograph denoising with Noise2Self and Noise2Void, as well as self-supervised Computed Tomography (CT) with a blind-spot losses in the sinogram.

📑 [View slides.pdf](./slides.pdf) — Supplementary presentation material.


---

## 📓 Notebooks

- 📘 [`usage.ipynb`](./usage.ipynb) — Demonstrates how to use the software in this repository.
- 🧪 [`phantoms.ipynb`](./phantoms.ipynb) — Details for the data used in the paper (Teledyne DALSA Xineos-3131 detectors), available at [Zenodo DOI: 10.5281/zenodo.15383254](https://doi.org/10.5281/zenodo.15383254).
- 🧠 [`2detect.ipynb`](./2detect.ipynb) — Contains the kernel estimation procedure for the 2DeTeCT dataset ([Nature Scientific Data DOI: 10.1038/s41597-023-02484-6](https://doi.org/10.1038/s41597-023-02484-6)).

---

## 📚 Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{Graas_2025,
  doi = {10.1088/1361-6501/addc06},
  url = {https://dx.doi.org/10.1088/1361-6501/addc06},
  year = {2025},
  month = {jun},
  publisher = {IOP Publishing},
  volume = {36},
  number = {6},
  pages = {065415},
  author = {Graas, Adriaan and Lucka, Felix},
  title = {Scintillator decorrelation for self-supervised {X}-ray radiograph denoising},
  journal = {Measurement Science and Technology}
}
```
