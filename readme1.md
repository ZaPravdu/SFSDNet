# Source Free SDNet
This project is forked from SDNet: https://github.com/fyw1999/MovingDroneCrowd

The corresponding paper of SDNet: https://arxiv.org/abs/2503.10701

# Usage
## 1. Scripts
There are several different script in this modified project as below:

- **train_script.py:** This script is the main script to finetune the SDNet and train the Cross Frame Autoencoder.
- **SDNet_inference.py & VGGAE_inference.py:** These are 2 scripts used to generate the pseudo density map and the error map of both reconstruction and density map.
- **analyze_error.py:** This script can calculate the correlation coefficient between the reconstruction and density error map on patch level.
- **data_visualization.py:** This script is used to visualize the results from analyze_error.py. 
- **SFSDNet_test.py:** This script is used to test the finetuned model.

## 2. Modules
- **model_assembler:** This is a model module layered on the VIC module from the original project. Its function is to call the class from the original project correctly, and define a training scheduler for pytorch lighting workflow.
- **dataset_assemble:** The function of this module is to call the dataset class in the original project correctly.