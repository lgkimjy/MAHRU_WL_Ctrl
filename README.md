# mujoco_template

Pure MuJoCo controller template.

This root folder intentionally does not contain a robot-specific definition.
After cloning or copying this template, fill these files first:

- `include/RobotDefinition.hpp`
- `config/fsm_JPosCtrl_config.yaml`
- `model/template/scene.xml`
- `model/template/robot.xml` or `model/template/robot.urdf`

`robot_name` is a placeholder namespace. Rename it or add `using namespace` in
your project as needed.
