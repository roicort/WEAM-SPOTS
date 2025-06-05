# Entropic Associative Memory (EAM) for the SPOTS-10 Dataset

This repository contains the procedures to replicate the experiments presented in the paper
Pineda, Luis A. & Rafael Morales _Imagery in the Entropic Associative Memory_ with the SPOTS-10 dataset. The code is written in Python and uses PyTorch for deep learning tasks.

## Dataset 

The SPOTS-10 dataset is a collection of 10 animal categories, each with 10000 images. 
The categories are as follows:

| ID | Category      |
|----|---------------|
| 0  | Cheetah       |
| 1  | Deer          |
| 2  | Giraffe       |
| 3  | Hyena         |
| 4  | Jaguar        |
| 5  | Leopard       |
| 6  | Tapir Calf    |
| 7  | Tiger         |
| 8  | Whale Shark   |
| 9  | Zebra         |

For more details, see [SPOTS/README.md](SPOTS/README.md).

## Requirements 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Run

sh train.sh && choose.py && sh dream.sh

---

## Experimentos

Cada experimento está en una rama, el primero y el segundo fueron con y sin normalizar. El tercero tiene el mejor score obtenido para el clasificador > 80%. En el cuarto se reproduce con las imágenes del autoencoder.
