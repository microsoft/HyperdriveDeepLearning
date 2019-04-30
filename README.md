# Out of Stock Detection Demo

To get started with the tutorial, please proceed with following steps **in sequential order**.

 * [Prerequisites](#prerequisites)
 * [Steps](#steps)
 * [Cleaning up](#cleanup)

<a id='prerequisites'></a>
## Prerequisites
1. Linux (x64) with GPU enabled.
2. [Anaconda Python](https://www.anaconda.com/download)
3. [Docker](https://docs.docker.com/v17.12/install/linux/docker-ee/ubuntu) installed.
4. [Azure account](https://azure.microsoft.com).

The demo was developed on an [Azure Ubuntu
DSVM](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro),
which addresses the first three prerequisites.

<a id='steps'></a>
## Steps
Please follow these steps to set up your environment and run notebooks.  They setup the notebooks to use Docker and Azure seamlessly.

1. Create the Python virtual environment using the environment.yml:
   ```
   conda env create -f environment.yml
   ```
2. Activate the virtual environment:
   ```
   conda activate outofstock_env
   ```

3. Clone the following repo and follow all the [instructions for installing Tensorflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) and install all necessary dependencies in the environment created in the first step.

    ```
   git clone https://github.com/tensorflow/models.git
   ```

4. Login to Azure:
   ```
   az login
   ```
5. If you have more than one Azure subscription, select it:
   ```
   az account set --subscription <Your Azure Subscription>
   ```
6. Start the Jupyter notebook server in the virtual environment:
   ```
   jupyter notebook
   ```
7. Select correct kernel: set the kernel to be `Python [conda env: outofstock_aml]`(or `Python 3` if that option does not show).

8. After following the setup instructions above, run the Jupyter notebooks in order starting with the first notebook.

<a id='cleanup'></a>
## Cleaning up
To remove the conda environment created see [here](https://conda.io/projects/continuumio-conda/en/latest/commands/remove.html). The last Jupyter notebook also gives details on deleting Azure resources associated with this repository.

# Contributing
This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
