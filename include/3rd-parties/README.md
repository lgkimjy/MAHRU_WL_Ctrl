# third-parties

Put external dependencies here if you want this template to vendor them.

For MuJoCo, either clone it here:

```bash
git clone --branch 3.2.6 --depth 1 https://github.com/google-deepmind/mujoco.git mujoco
```

or configure CMake with:

```bash
cmake -S . -B build -DTEMPLATE_MUJOCO_SOURCE_DIR=/path/to/mujoco
```

> [!note]
Clone all the external sources (e.g. MuJoCo, unitree_sdk2, osqp, qpOASES, etc.)

For this repository, MuJoCo version 3.2.6 works, if wanted to use different version of MuJoCo, you might have to modify ```SimulationInterface.cpp``` and ```SimulationInterface.hpp```. 
```shell
$ git clone -b 3.2.6 https://github.com/google-deepmind/mujoco.git
```

others
```shell
$ git clone https://github.com/coin-or/qpOASES.git
$ git clone https://github.com/andreacasalino/csvcpp.git
```
