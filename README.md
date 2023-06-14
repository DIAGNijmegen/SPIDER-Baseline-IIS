# Vertebra segmentation and identification with iterative FCNs

![Animation](/animation.gif)

## Structure

* Source code for development, training and validation (`devel/`)
* Original code written for the Medical Image Analysis paper (`original/`)
* Processor for applying the method to unseen images (`processor/`)
* Processor submitted to the VerSe19 challenge (`processor_verse2019/`)
* Implementations of a few landmark detection approaches, used in other projects for finding L3 (`detectors/`)

All components are based on docker images. To build these, run `make build` and `make push` to push them to doduo:

* `doduo1.umcn.nl/nikolas/vertebra-segmentation:devel` The image for development use, e.g., network training and evaluation.
* `doduo1.umcn.nl/nikolas/vertebra-segmentation:interactive` Image for interactive development, otherwise identical to devel.
* `doduo1.umcn.nl/nikolas/vertebra-segmentation:processor` Processor image for inference on SOL, implements the DIAG processor API.
* `doduo1.umcn.nl/nikolas/vertebra-segmentation:app` Processor image for inference on grand-challenge.org

## Publications

N. Lessmann et al.,
"Iterative fully convolutional neural networks for automatic vertebra segmentation and identification",
Medical Image Analysis 53 (2019), pp. 142-155, https://doi.org/10.1016/j.media.2019.02.005

## Running scripts on SOL

The `devel/` folder contains the main codebase. Scripts that can be executed are indicated by a name starting with a capital letter.

Overall, the steps are:

1. **Train** - trains the network
2. **AverageStateDicts** - the network often ends up in an unstable state where the loss curve wobbles up and down. The robustness can be increased substantially by averaging the weights over the last few snapshots (15k-25k iterations).
3. **Test** - runs the network on one or more datasets
4. **Evaluate** - computes Dice score and surface distance

To add a new dataset, two helper scripts are available:

1. **GenerateMetadataFile** - each dataset consists of an `images` and a `masks` folder (if available), and a `metadata.json` file that lists each image in the dataset. A first version of this file can be created with this script.
2. **VerifyDataIntegrity** - runs consistency checks on a dataset

### Experiments

The pipeline works with a concept of Experiments. Many of the scripts listed above require as first command line argument the name of an experiment. This name can be chosen freely, but the convention is to use CamelCase.  The experiment name is used to identify the experiment and to create a folder with the same name in the `experiments/` folder. All results related to that particular experiment are stored in that folder, e.g., network weights, learning curves, inference results, etc.

Additionally, the pipeline supports specifying the location of the codebase - this enables multiple people to work on the same codebase and run it from Chansey without interfering with each other. When running scripts on SOL, both the location of the codebase and the "base dir" (which contains the `datasets/` and `experiments/` folders) need to be provided.

### Training the network

This command would start the training process:

```commandline
~/c-submit --require-cpus=4 --require-mem=50g --gpu-count=1 --require-gpu-mem=11g --priority=high \
           --env=VERSEG_BASEDIR=/mnt/netcache/bodyct/experiments/vertebra_segmentation_t8243 \
           --env=VERSEG_EXTDIR=/mnt/netcache/bodyct/experiments/vertebra_segmentation_t8243/code/devel \
           nikolas 1234 168 doduo1.umcn.nl/nikolas/vertebra-segmentation:devel \
           Train ExperimentName --dataset verse2019_nlst_csi_xvertseg_radboud_scoliosis_synthetic
```

Training the network requires around 40 GB of memory when using all available CT datasets.

When the training has finished (after at least 150k iterations), run the following command to average the weights:
```commandline
~/c-submit --require-cpus=4 --require-mem=16g --gpu-count=1 --require-gpu-mem=11g --priority=high \
           --env=VERSEG_BASEDIR=/mnt/netcache/bodyct/experiments/vertebra_segmentation_t8243 \
           --env=VERSEG_EXTDIR=/mnt/netcache/bodyct/experiments/vertebra_segmentation_t8243/code/devel \
           nikolas 1234 4 doduo1.umcn.nl/nikolas/vertebra-segmentation:devel \
           AverageStateDicts ExperimentName --from_epoch 175000 --to_epoch 200000 --make_epoch 999999
```

### Testing the network

The network can be run on the individual datasets and also only on their subsets (like "testing"):

```commandline
~/c-submit --require-cpus=4 --require-mem=16g --gpu-count=1 --require-gpu-mem=11g --priority=high \
           --env=VERSEG_BASEDIR=/mnt/netcache/bodyct/experiments/vertebra_segmentation_t8243 \
           --env=VERSEG_EXTDIR=/mnt/netcache/bodyct/experiments/vertebra_segmentation_t8243/code/devel \
           nikolas 1234 24 doduo1.umcn.nl/nikolas/vertebra-segmentation:devel \
           Test ExperimentName --epoch 999999 --dataset nlst --subset testing --export_orgres
```

To compute metrics like the Dice score, use the following command:

```commandline
~/c-submit --require-cpus=4 --require-mem=24g --priority=high \
           --env=VERSEG_BASEDIR=/mnt/netcache/bodyct/experiments/vertebra_segmentation_t8243 \
           --env=VERSEG_EXTDIR=/mnt/netcache/bodyct/experiments/vertebra_segmentation_t8243/code/devel \
           nikolas 1234 12 doduo1.umcn.nl/nikolas/vertebra-segmentation:devel \
           Evaluate ExperimentName --dataset nlst --subset testing --eval_orgres
```

## Running the model locally

The processor can be run locally if a docker installation with access to a suitable GPU is available. The command below assumes that the docker image was imported from the snapshot stored in `\\radng_diag_msk\projects\VertebraSegmentation\vertebra-segmentation-processor.tar.xz`

To prepare the input data, place all images that you want to process in a single folder. Supported file formats are .mha, .mhd, .nii and .nii.gz

Create another empty folder for the output and run the following docker command:

```commandline
docker run -it --rm \
           -v D:/spine/images:/input:r \
           -v D:/spine/segmentations:/output:rw \
           radboudumc.nl/diag/vertebra-segmentation:processor --imagedir /input --nometadata
```
