# CropAI training images

This folder contains four leaf-condition classes used by `train.py`:

| Class | Images |
| --- | ---: |
| Bacterial spot | 2,134 |
| Early blight | 1,009 |
| Late blight | 1,908 |
| Healthy | 1,593 |
| **Total** | **6,644** |

The added color images are from the [PlantVillage dataset repository](https://github.com/spMohanty/PlantVillage-Dataset), under `raw/color/`. See Mohanty, Hughes and Salathé, [“Using Deep Learning for Image-Based Plant Disease Detection”](https://doi.org/10.3389/fpls.2016.01419), *Frontiers in Plant Science* (2016). The original repository describes 54,306 images spanning 14 crop species and 26 diseases. This project uses the four target classes listed above. Thirty-one pre-existing project images were retained and combined with the PlantVillage images. Fourteen exact duplicate copies in the downloaded subset were removed.

PlantVillage photos were collected in controlled conditions. The training script makes a random image-level 80/20 train/validation split; it does not create an independent field test set. This dataset is useful for an educational classification demo, but it does not represent all field conditions. Keep this source note with the data when presenting or sharing the project.
