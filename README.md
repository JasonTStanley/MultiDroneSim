## MultiDroneSim
This repository uses the gym-pybullet-drones library to simulate multiple drones in a 3D environment.

To setup this project, we recommend creating a new conda environment from the environment.yaml file as follows:
```bash
conda env create -n <ENV_NAME> -f environment.yaml
```
```bash
conda activate <ENV_NAME>
```
The `environment.yaml` file includes all dependencies for the project except for [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones),
which needs to be built separately. The following should be sufficient to install the latest version, 
but see the repository for more information if needed.
```bash
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones

pip install -e .
```
To check that the project is setup properly, run the following command:
```bash
python MultiDroneExample.py
```
This should open a window with a 3D environment with two drones hovering off the ground.
