cmake_minimum_required(VERSION 3.16)

project(mahru_ctrl LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_definitions(CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

set(TEMPLATE_MUJOCO_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/include/3rd-parties/mujoco"
  CACHE PATH "Path to the MuJoCo source tree")

if(NOT EXISTS "${TEMPLATE_MUJOCO_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "MuJoCo source tree not found.\n"
    "Clone MuJoCo into include/3rd-parties/mujoco, for example:\n"
    "  git clone --branch 3.2.6 --depth 1 https://github.com/google-deepmind/mujoco.git include/3rd-parties/mujoco\n"
    "or configure with -DTEMPLATE_MUJOCO_SOURCE_DIR=/path/to/mujoco")
endif()

set(MUJOCO_BUILD_EXAMPLES OFF CACHE BOOL "Build MuJoCo examples" FORCE)
set(MUJOCO_BUILD_SIMULATE ON CACHE BOOL "Build MuJoCo simulate library" FORCE)
set(MUJOCO_BUILD_TESTS OFF CACHE BOOL "Build MuJoCo tests" FORCE)
add_subdirectory("${TEMPLATE_MUJOCO_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/mujoco")

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL CXX)
find_package(yaml-cpp CONFIG REQUIRED)

file(GLOB FSM_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/StateMachine/FSM_*.cpp"
)

set(ARBML_SRC
  include/3rd-parties/ARBMLlib/ARBML.cpp
  include/3rd-parties/ARBMLlib/RigidBody.cpp
)

add_executable(mahru_ctrl
  src/main.cpp
  src/MJSimulationBridge.cpp
  include/StateMachine/StateMachineCtrl.cpp
  include/Interface/MuJoCo/SimulationInterface.cpp
  include/Interface/MuJoCo/simulate.cc
  ${FSM_SOURCES}
  ${ARBML_SRC}
)

target_include_directories(mahru_ctrl PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${TEMPLATE_MUJOCO_SOURCE_DIR}/simulate"
  "${TEMPLATE_MUJOCO_SOURCE_DIR}/sample"
  ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(mahru_ctrl PRIVATE
  Eigen3::Eigen
  yaml-cpp::yaml-cpp
  mujoco::libsimulate
  mujoco::mujoco
  glfw
  lodepng
  Threads::Threads
  ${HDF5_LIBRARIES}
)

if(DEFINED MUJOCO_SIMULATE_COMPILE_OPTIONS)
  target_compile_options(mahru_ctrl PUBLIC ${MUJOCO_SIMULATE_COMPILE_OPTIONS})
endif()

set_target_properties(mahru_ctrl PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
