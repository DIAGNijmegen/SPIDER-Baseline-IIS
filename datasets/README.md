# SPIDER Baseline algorithm: Iterative Instance Segmentation of the spine

## Use of datasets structure

Datasets used for training the IIS baseline algorithm (e.g. the SPIDER dataset) should be placed in this directory. The folder should be named accordingly and should contain two folders (images and masks) containing all data as well as a metadata.json file. 
The specific dataset should be added to the code if you would want to use it for training (add it to the first part of the execute function). 

 A first version of the metadata file can be created with the **GenerateMetadataFile** script.