# Entropic associative memory using a single register

This repository contains the procedures to replicate the experiments presented in the paper

Pineda, Luis A. & Rafael Morales (under review). _Imagery in the Entropic Associative Memory_.

## Dataset SPOTS-10

El dataset SPOTS-10 contiene imágenes clasificadas en 10 categorías animales:

| ID | Categoría      |
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

Para más detalles, consulta [SPOTS/README.md](SPOTS/README.md).

## Uso 

sh train.sh && choose.py && sh dream.sh

## Experimentos

Cada experimento esta en una rama, el primero y el segundo fueron con y sin normalizar. El tercero tiene el mejor score obtenido para el clasificador > 80%. En el cuarto se reproduce con las imagenes del autoencoder.
