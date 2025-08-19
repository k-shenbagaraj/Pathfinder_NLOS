# An Attention Network-Based Approach to Dynamic Non-Line-Of-Sight Human Tracking Using a Mobile Robot


### Environment

By using Anaconda, you can easily setup the environment using `environment.yml`

```shell
conda env create -f environment.yml

conda activate nlospatch
```
### Download Weights

The trained weights for our NLOS-Patch Network and PlaneRecNet is available at the [link](https://drive.google.com/drive/folders/1T9bkrxMQRUi9WudZ9kMm8Ek02gzfwx__?usp=sharing)

Please download the weights and unzip the folder to this parent folder
The output of this should be 
```
+project_root
|   Readme.md
|   environment.yml
|   test.py
+---utils
+---configs
+---DynamicNLOSweights 
    +---NLOSmodel
    |       best.pth
    |       configs.yaml  
    +---PlanRecNetmodel
    |       PlaneRecNet.pth
+---testdata_real   
   +---0
   |        0.mat
   |        1.mat
   |        2.mat
   |      
   |        ...
   |        31.mat
   
            
```


### Sample Real Test Data Sequence

The folder `testdata_real` contains sample pre-processed inputs to the NLOS-Patch Network for convenience.
This is the output of the Plane Processing Pipeline discussed in the paper Sec. 4.1.

Here each subfolder contains a trajectory of length 32.
Each mat file corresponds to a timestamp and contains a dictionary of `[raw_input_planes, diff_planes, PlaneID, NLOS_GT, v_GT, map_sizes]`  
### Test Data

The inference on a sample test sequence is in `test.py`

This code sequence loads a sample test sequence and outputs the network predicted trajectory along with ground truth which is saved as `predicted_trajectory.png`


### Plane Processing Pipeline

The code for the plane processing pipeline is in folder `PlaneProcessing`.
This is shared here for the reviewers to get better clarity of the pipeline.