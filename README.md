# Seek Commonality but Preserve Differences: Dissected Dynamics Modeling for Multi-modal Visual RL

## Our code is modified from "Learning Invariant Representations for Reinforcement Learning without Reconstruction".

## Requirements

We assume you have access to a gpu that can run CUDA 9.2. Then, the simplest way to install all required dependencies is to create an anaconda environment 




## Instructions


```
conda env create -f conda_env.yml
```
After the installation ends you can activate your environment with:
```
source activate carla_0.9.13
```

Download the appropriate package for version 0.9.13 of CARLA. Then, follow the official instructions for installation.

Terminal 1:
```
cd CARLA_0.9.13
bash CarlaUE4.sh -fps 20
```



## CARLA
Download CARLA from https://github.com/carla-simulator/carla/releases, e.g.:
1. https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.13.tar.gz
2. https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.13.tar.gz

Add to your python path:
```
export PYTHONPATH=$PYTHONPATH:/home/XXXX-1/code/bisim_metric/CARLA_0.9.13/PythonAPI
export PYTHONPATH=$PYTHONPATH:/home/XXXX-1/code/bisim_metric/CARLA_0.9.13/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/XXXX-1/code/bisim_metric/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.5-linux-x86_64.egg
```
and merge the directories.

Then pull altered carla branch files:
```
git fetch
git checkout carla
```




Terminal 2:
```
bash run.sh
```

## License
This project is CC-BY-NC 4.0 licensed, as found in the LICENSE file.
