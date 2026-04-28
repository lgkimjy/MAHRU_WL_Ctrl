#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


def normalize_task_name(task: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", task)
    if not tokens:
        raise ValueError("Task name must contain at least one alphanumeric character.")
    return "".join(token[:1].upper() + token[1:] for token in tokens)


def write_text(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Re-run with --force to overwrite it.")
    path.write_text(content, encoding="utf-8")


def insert_before(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    if block in text:
        return
    if marker not in text:
        raise RuntimeError(f"Failed to find marker in {path}: {marker}")
    updated = text.replace(marker, f"{block}{marker}", 1)
    path.write_text(updated, encoding="utf-8")


def insert_after_last_regex(path: Path, pattern: str, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    if block in text:
        return
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        raise RuntimeError(f"Failed to find insertion point in {path}: {pattern}")
    pos = matches[-1].end()
    updated = text[:pos] + block + text[pos:]
    path.write_text(updated, encoding="utf-8")


def replace_regex(path: Path, pattern: str, repl: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Failed to update {path} with pattern: {pattern}")
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new sim2sim FSM task scaffold.")
    parser.add_argument("--task", required=True, help="Task name, for example Balance-Ctrl")
    parser.add_argument("--force", action="store_true", help="Overwrite generated files if they already exist")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    task_name = normalize_task_name(args.task)
    class_name = f"FSM_{task_name}State"
    enum_name = f"FSM_{task_name}"

    header_path = repo_root / "include" / "StateMachine" / f"FSM_{task_name}.hpp"
    source_path = repo_root / "include" / "StateMachine" / f"FSM_{task_name}.cpp"
    config_path = repo_root / "config" / f"fsm_{task_name}_config.yaml"
    ctrl_hpp_path = repo_root / "include" / "StateMachine" / "StateMachineCtrl.hpp"
    ctrl_cpp_path = repo_root / "include" / "StateMachine" / "StateMachineCtrl.cpp"

    header_guard = f"__FSM_{task_name.upper()}_HPP__"

    header_text = f"""#ifndef {header_guard}
#define {header_guard}

#include <string>

#include "RobotDefinition.hpp"
#include "States.hpp"
#include "RobotStates.hpp"

#include "Interface/MuJoCo/traj_viz_util.hpp"

using namespace mahru;

template <typename T>
class {class_name} : public States {{
public:
    explicit {class_name}(RobotData& robot);
    ~{class_name}() {{}};

    void onEnter() override;
    void runNominal() override;
    void checkTransition() override {{}};
    void runTransition() override {{}};
    void setVisualizer(mujoco::TrajVizUtil* visualizer) override;

private:
    RobotData* robot_data_;
    mujoco::TrajVizUtil* viz_ = nullptr;

    void updateModel();
    void updateCommand();
    void updateVisualization();
    void readConfig(std::string config_file);
}};

#endif // {header_guard}
"""

    source_text = f"""#include "FSM_{task_name}.hpp"

#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>

template <typename T>
{class_name}<T>::{class_name}(RobotData& robot) :
    robot_data_(&robot)
{{
    std::cout << "[ {class_name} ] Constructed" << std::endl;
}}

template <typename T>
void {class_name}<T>::onEnter()
{{
    std::cout << "[ {class_name} ] OnEnter" << std::endl;

    if (viz_) {{
        // viz_->clearPrefix("{task_name}/");
        viz_->clear();
    }}

    readConfig(CMAKE_SOURCE_DIR "/config/fsm_{task_name}_config.yaml");
    this->state_time = 0.0;
}}

template <typename T>
void {class_name}<T>::runNominal()
{{
    updateModel();
    updateCommand();
    updateVisualization();

    this->state_time += 0.001;
}}

template <typename T>
void {class_name}<T>::setVisualizer(mujoco::TrajVizUtil* visualizer)
{{
    viz_ = visualizer;
}}

template <typename T>
void {class_name}<T>::updateModel()
{{
}}

template <typename T>
void {class_name}<T>::updateCommand()
{{
    robot_data_->ctrl.torq_d.setZero();
}}

template <typename T>
void {class_name}<T>::updateVisualization()
{{
    if (!viz_) return;

    viz_->sphere("{task_name}/base",
        robot_data_->fbk.p_B,
        0.035, {{0.3f, 0.0f, 0.3f, 0.8f}}
    );
}}

template <typename T>
void {class_name}<T>::readConfig(std::string config_file)
{{
    std::cout << "[ {class_name} ] readConfig: " << config_file << std::endl;
    const YAML::Node config = YAML::LoadFile(config_file);
    (void)config;
}}

// template class {class_name}<float>;
template class {class_name}<double>;
"""

    config_text = f"""notes:
  description: "Fill in task-specific gains, commands, and policy settings here."
"""

    write_text(header_path, header_text, args.force)
    write_text(source_path, source_text, args.force)
    write_text(config_path, config_text, args.force)

    insert_before(
        ctrl_hpp_path,
        "    NUM_STATE\n",
        f"    {enum_name},\n",
    )
    insert_after_last_regex(
        ctrl_cpp_path,
        r'^#include "StateMachine/FSM_[^"]+\.hpp"\n',
        f'#include "StateMachine/FSM_{task_name}.hpp"\n',
    )
    insert_after_last_regex(
        ctrl_cpp_path,
        r"^\s*state_list_\[StateList::[A-Za-z0-9_]+\] = new [A-Za-z0-9_]+(?:<[^>]+>)?\(robot\);\n",
        f"    state_list_[StateList::{enum_name}] = new {class_name}<double>(robot);\n",
    )
    replace_regex(
        ctrl_cpp_path,
        r"current_state_ = state_list_\[StateList::[A-Za-z0-9_]+\];",
        f"current_state_ = state_list_[StateList::{enum_name}];",
    )

    created = [header_path, source_path, config_path, ctrl_hpp_path, ctrl_cpp_path]
    for path in created:
        print(path.relative_to(repo_root))


if __name__ == "__main__":
    main()
