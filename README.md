# SPIDER Baseline algorithm: Iterative Instance Segmentation of the spine

## Publications

N. Lessmann et al.,
"Iterative fully convolutional neural networks for automatic vertebra segmentation and identification",
Medical Image Analysis 53 (2019), pp. 142-155, https://doi.org/10.1016/j.media.2019.02.005

## Running scripts

The `devel/` folder contains the main codebase. Scripts that can be executed are indicated by a name starting with a capital letter.

Overall, the steps are:

1. **TrainSpine** - trains the network
2. **AverageStateDicts** - the network often ends up in an unstable state where the loss curve wobbles up and down. The robustness can be increased substantially by averaging the weights over the last few snapshots (15k-25k iterations).
3. **TestSpine*** - runs the network on one or more datasets
4. **EvaluateSpine** - computes Dice score and surface distance

To add a new dataset, two helper scripts are available:

1. **GenerateMetadataFile** - each dataset consists of an `images` and a `masks` folder (if available), and a `metadata.json` file that lists each image in the dataset. A first version of this file can be created with this script.
2. **VerifyDataIntegrity** - runs consistency checks on a dataset

### Experiments

The pipeline works with a concept of Experiments. Many of the scripts listed above require as first command line argument the name of an experiment. This name can be chosen freely, but the convention is to use CamelCase.  The experiment name is used to identify the experiment and to create a folder with the same name in the `experiments/` folder. All results related to that particular experiment are stored in that folder, e.g., network weights, learning curves, inference results, etc.

Additionally, the pipeline supports specifying the location of the codebase - this enables multiple people to work on the same codebase and run it without interfering with each other. 

