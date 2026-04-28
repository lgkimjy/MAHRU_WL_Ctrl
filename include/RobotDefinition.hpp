#pragma once

#include <array>
#include <string_view>

constexpr size_t DOF2 = 2;
constexpr size_t DOF3 = 3;
constexpr size_t DOF4 = 4; 
constexpr size_t DOF6 = 6;

#define _FLOATING_BASE

namespace mahru
{
    constexpr size_t nDoF_base = 6;
    constexpr size_t nDoFQuat_base = 7;

    constexpr size_t num_leg_joint = 5;
    constexpr size_t num_arm_joint = 4;
    constexpr size_t num_waist_joint = 1;
    constexpr size_t num_act_joint = 2 * num_leg_joint + 2 * num_arm_joint + num_waist_joint;

    constexpr size_t nDoF = num_act_joint + nDoF_base;
    constexpr size_t nDoFQuat = num_act_joint + nDoFQuat_base;

    constexpr size_t NO_OF_BODY = num_act_joint + 1;

    constexpr std::array<int, num_act_joint> robot_joint_idx = {};

    constexpr std::array<int, num_act_joint> jnt_mapping_idx = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20
    };

    inline constexpr std::string_view scene = "/model/scene.xml";
    inline constexpr std::string_view model_xml = "/model/MAHRU-WL_w_Battery.xml";
    inline constexpr std::string_view model_urdf = "/model/MAHRU-WL_w_Battery.urdf";

    inline constexpr std::array<std::string_view, num_act_joint> joint_names = {};
}
