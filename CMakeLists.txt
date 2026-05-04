cmake_minimum_required(VERSION 3.16)

project(mahru_ctrl LANGUAGES C CXX)

if(APPLE)
  add_compile_options(-Wno-error)
endif()

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

# MuJoCo 포함 코드 직전에 추가
set(CMAKE_REQUIRED_FLAGS "-Wno-error")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")
set(MUJOCO_BUILD_EXAMPLES OFF CACHE BOOL "Build MuJoCo examples" FORCE)
set(MUJOCO_BUILD_SIMULATE ON CACHE BOOL "Build MuJoCo simulate library" FORCE)
set(MUJOCO_BUILD_TESTS OFF CACHE BOOL "Build MuJoCo tests" FORCE)
add_subdirectory("${TEMPLATE_MUJOCO_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/mujoco")
set(QPOASES_BUILD_EXAMPLES OFF CACHE BOOL "Build qpOASES examples" FORCE)
set(QPOASES_AVOID_LA_NAMING_CONFLICTS OFF CACHE BOOL "" FORCE)
add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/3rd-parties/qpOASES"
  "${CMAKE_BINARY_DIR}/qpOASES")

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL CXX)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(eigen-quadprog REQUIRED)
include(FetchContent)
find_package(OsqpEigen QUIET)
if(NOT OsqpEigen_FOUND)
  message(STATUS "OsqpEigen not found. Fetching OSQP and OsqpEigen via FetchContent.")
  set(OSQP_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
  set(OSQP_BUILD_DEMO_EXE OFF CACHE BOOL "" FORCE)
  set(OSQP_BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
  set(OSQP_EIGEN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(OSQP_EIGEN_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(
    osqp
    GIT_REPOSITORY https://github.com/osqp/osqp.git
    GIT_TAG v0.6.3
  )
  FetchContent_MakeAvailable(osqp)
  foreach(_osqp_target IN ITEMS osqp osqpstatic)
    if(TARGET ${_osqp_target})
      target_compile_definitions(${_osqp_target} PRIVATE _POSIX_C_SOURCE=199309L)
    endif()
  endforeach()

  FetchContent_Declare(
    osqp_eigen
    GIT_REPOSITORY https://github.com/robotology/osqp-eigen.git
    GIT_TAG v0.8.0
  )
  set(_BUILD_TESTING_SAVED ${BUILD_TESTING})
  set(_BUILD_SHARED_LIBS_SAVED ${BUILD_SHARED_LIBS})
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(osqp_eigen)
  set(BUILD_TESTING ${_BUILD_TESTING_SAVED} CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS ${_BUILD_SHARED_LIBS_SAVED} CACHE BOOL "" FORCE)
endif()

if(NOT TARGET OsqpEigen::OsqpEigen AND TARGET OsqpEigen)
  add_library(OsqpEigen::OsqpEigen ALIAS OsqpEigen)
endif()

if(TARGET mujoco::libmujoco_simulate)
  set(MUJOCO_SIMULATE_TARGET mujoco::libmujoco_simulate)
elseif(TARGET mujoco::libsimulate)
  set(MUJOCO_SIMULATE_TARGET mujoco::libsimulate)
else()
  message(FATAL_ERROR "MuJoCo simulate library target was not found.")
endif()

file(GLOB FSM_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/StateMachine/FSM_*.cpp"
)

set(ARBML_SRC
  include/3rd-parties/ARBMLlib/ARBML.cpp
  include/3rd-parties/ARBMLlib/RigidBody.cpp
)

set(MPC_SRC
  include/Controller/ConvexMPC/ConvexMpc.cpp
  include/Controller/ConvexMPC/GaitGenerator.cpp
  include/Controller/ConvexMPC/LocoCtrl.cpp
  include/Trajectory/OnlineOptTraj.cpp
)

set(WBC_SRC
  include/WBC/WeightedWBC.cpp
)

add_executable(mahru_ctrl
  src/main.cpp
  src/MJSimulationBridge.cpp
  include/StateMachine/StateMachineCtrl.cpp
  include/Interface/MuJoCo/SimulationInterface.cpp
  include/Interface/MuJoCo/simulate.cc
  ${FSM_SOURCES}
  ${ARBML_SRC}
  ${MPC_SRC}
  ${WBC_SRC}
)

target_include_directories(mahru_ctrl PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/3rd-parties/qpOASES/include"
  "${TEMPLATE_MUJOCO_SOURCE_DIR}/simulate"
  "${TEMPLATE_MUJOCO_SOURCE_DIR}/sample"
  ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(mahru_ctrl PRIVATE
  Eigen3::Eigen
  OsqpEigen::OsqpEigen
  eigen-quadprog::eigen-quadprog
  yaml-cpp::yaml-cpp
  qpOASES
  ${MUJOCO_SIMULATE_TARGET}
  mujoco::mujoco
  glfw
  lodepng
  Threads::Threads
  ${HDF5_LIBRARIES}
)

if(APPLE)
  target_link_libraries(mahru_ctrl PRIVATE "-framework CoreVideo")
endif()

if(DEFINED MUJOCO_SIMULATE_COMPILE_OPTIONS)
  target_compile_options(mahru_ctrl PUBLIC ${MUJOCO_SIMULATE_COMPILE_OPTIONS})
endif()

set_target_properties(mahru_ctrl PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
