#ifndef ROBOT_CPP_PARAMS_H
#define ROBOT_CPP_PARAMS_H

// control time related
//#define CTRL_FREQUENCY 2.5  // ms
// #define GRF_UPDATE_FREQUENCY 2.5 // ms
#define MAIN_UPDATE_FREQUENCY 3.0 // ms
// #define HARDWARE_FEEDBACK_FREQUENCY 2.0  // ms

// mpc
#define PLAN_HORIZON 9
#define MPC_STATE_DIM 13
#define MPC_CONSTRAINT_DIM 20

// robot constant
#define NUM_LEG 4
#define NUM_DOF_PER_LEG 3
#define DIM_GRF 12
#define NUM_DOF 12

#define FOOT_SWING_CLEARANCE1 0.0f
#define FOOT_SWING_CLEARANCE2 0.4f

#endif //ROBOT_CPP_PARAMS_H
