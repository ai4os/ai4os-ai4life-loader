# ai4life
[![Build Status](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS%2Fai4os-ai4life-loader%2Fmain)](https://jenkins.services.ai4os.eu/job/AI4OS/job/ai4os-ai4life-loader/job/main/)

The [BioImage Model Zoo](https://bioimage.io/#/) is a community-driven platform that 
  provides standardized deep learning models for bioimage analysis.
  
  This module integrates models from the **BioImage.IO** package into the **AI4EOSC** Marketplace
  dashboard, specifically those using PyTorch weights and following the v0.5 format.
  The module allows users to seamlessly explore, deploy, and utilize these models within 
  the AI4EOSC ecosystem, providing a user-friendly interface for advanced bioimage analysis.
  
**Key Features** 
  - Model Discovery: Automatically fetch and list models available in BioImage.IO that meet the criteria (PyTorch weights, v0.5 format).

  - Metadata Visualization: Display essential information about each model,
   such as input/output specifications, authors, license details, and documentation.

  - Seamless Deployment: Enable one-click deployment of models to AI4EOSC compute resources.

  - Model Preview: Provide an interactive preview to test models on sample data directly in the dashboard.

**Supported Models**
- PyTorch Models: Only models with PyTorch weights are supported to ensure compatibility with our deployment backend.
- BioImage.IO v0.5 Specification: Models must adhere to the v0.5 specification, ensuring a standardized format for inputs, outputs, and metadata.


To launch it, first install the package then run [deepaas](https://github.com/ai4os/DEEPaaS):

> ![warning](https://img.shields.io/badge/Warning-red.svg) **Warning**: If you are using a virtual environment, make sure you are working with the last version of pip before installing the package. Use `pip install --upgrade pip` to upgrade pip.

```bash
git clone https://github.com/ai4os/ai4os-ai4life-loader
cd ai4life
pip install -e .
deepaas-run --listen-ip 0.0.0.0
```
The associated Docker image(s) for this module can be found in:
https://hub.docker.com/r/ai4oshub/ai4os-ai4life-loader/tags 

##  AI4OS-AI4Life-Loader Deployment Guide
**Deployment Steps**
- Navigate to the [AI4EOSC dashboard](https://dashboard.cloud.ai4eosc.eu/marketplace) marketplace
- Locate and select the ai4os-ai4life-loader tool
- Click on the "Deploy" button to start the deployment process
**Configuration Form**
- Select your desired model from the dropdown menu
- Complete all required fields in the deployment form
**Post-Deployment**
- The system will initialize your selected model
- Wait for the deployment process to complete and the status change to running 
- Access your deployed model through the provided endpoint 

 ## Project structure

```
├── Jenkinsfile             <- Describes basic Jenkins CI/CD pipeline
├── Dockerfile              <- Steps to build a DEEPaaS API Docker image
├── LICENSE                 <- License file
├── README.md               <- The top-level README for developers using this project.
├── VERSION                 <- Version file indicating the version of the model
│
├── ai4life
│   ├── README.md           <- Instructions on how to integrate your model with DEEPaaS.
│   ├── __init__.py         <- Makes <your-model-source> a Python module
│   ├── ...                 <- Other source code files
│   └── config.py           <- Module to define CONSTANTS used across the AI-model python package
│
├── api                     <- API subpackage for the integration with DEEP API
│   ├── __init__.py         <- Makes api a Python module, includes API interface methods
│   ├── config.py           <- API module for loading configuration from environment
│   ├── responses.py        <- API module with parsers for method responses
│   ├── schemas.py          <- API module with definition of method arguments
│   └── utils.py            <- API module with utility functions
│
├── data                    <- Data subpackage for the integration with DEEP API
│
├── docs                    <- A default Sphinx project; see sphinx-doc.org for details
│
├── models                  <- Folder to store your models
│
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for ordering),
│                              the creator's initials (if many user development),
│                              and a short `_` delimited description, e.g.
│                              `1.0-jqp-initial_data_exploration.ipynb`.
│
├── references              <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated graphics and figures to be used in reporting
│
├── requirements-dev.txt    <- Requirements file to install development tools
├── requirements-test.txt   <- Requirements file to install testing tools
├── requirements.txt        <- Requirements file to run the API and models
│
├── pyproject.toml          <- Makes project pip installable (pip install -e .)
│
├── tests                   <- Scripts to perform code testing
│   ├── configurations      <- Folder to store the configuration files for DEEPaaS server
│   ├── conftest.py         <- Pytest configuration file (Not to be modified in principle)
│   ├── data                <- Folder to store the data for testing
│   ├── models              <- Folder to store the models for testing
│   ├── test_deepaas.py     <- Test file for DEEPaaS API server requirements (Start, etc.)
│   ├── test_metadata       <- Tests folder for model metadata requirements
│   ├── test_predictions    <- Tests folder for model predictions requirements
│   └── test_training       <- Tests folder for model training requirements
│
└── tox.ini                 <- tox file with settings for running tox; see tox.testrun.org
```



## Documentation

TODO: Add instructions on how to build documentation

 