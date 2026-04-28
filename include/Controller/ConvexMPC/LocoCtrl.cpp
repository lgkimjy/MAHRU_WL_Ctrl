#include "ConvexMPC/LocoCtrl.hpp"

LocoCtrl::LocoCtrl() :
    mpc_solver(Eigen::VectorXd(13).setZero(), Eigen::VectorXd(12).setZero()),
    stand(12, Eigen::Vector4i(0, 0, 0, 0), Eigen::Vector4i(12, 12, 12, 12), "Stand"),
    linewalk(12, Eigen::Vector4i(0, 0, 6, 6), Eigen::Vector4i(6, 6, 6, 6), "LineWalk"),   // single stance
    pointwalk(12, Eigen::Vector4i(0, 0, 0, 6), Eigen::Vector4i(0, 6, 0, 6), "PointWalk"),   // single stance
    linewalk2(16, Eigen::Vector4i(3, 3, 11, 11), Eigen::Vector4i(10, 10, 10, 10), "LineWalkDSP"),  // double stance
    pointwalk2(16, Eigen::Vector4i(0, 3, 0, 11), Eigen::Vector4i(0, 10, 0, 10), "PointWalkDSP"), // double stance
    run(12, Eigen::Vector4i(0, 0, 0, 0), Eigen::Vector4i(12, 12, 12, 12), "Run"), 
    // slide(20, Eigen::Vector4i(3, 3, 13, 13), Eigen::Vector4i(4, 14, 4, 14), "Sliding") // temporal for sliding, currently same as double stance included linewalking
    // slide(20, Eigen::Vector4i(7, 3, 17, 13), Eigen::Vector4i(10, 14, 10, 14), "Sliding") // for sliding, currently same as double stance included linewalking
    slide(20, Eigen::Vector4i(3, 3, 13, 13), Eigen::Vector4i(14, 14, 14, 14), "Sliding") // for sliding, currently same as double stance included linewalking
    // slide(28, Eigen::Vector4i(3, 3, 17, 17), Eigen::Vector4i(22, 22, 22, 22), "Sliding") // for sliding, currently same as double stance included linewalking, longer DSP
{
    std::cout << "LocoCtrl constructor" << std::endl;
    
    mpc_states.setZero();
    mpc_states_d.setZero();
    mpc_solver.reset();

    iterationCounter = 0;

    // opt. trajectory initialization
    optTraj.set_mid_air_height(0.079);
    optTraj.set_costs(1e1, 1e1, 1e0, 1e-6);

    // Initialize foot trajectory
    left_foot_position.setZero();
    left_foot_velocity.setZero();
    left_foot_acceleration.setZero();
    right_foot_position.setZero();
    right_foot_velocity.setZero();
    right_foot_acceleration.setZero();

    p_whl_slide_ref.setZero();
    p_whl_slide.setZero();
    p_whl_slide_d.setZero();
    pdot_whl_slide_d.setZero();
    pddot_whl_slide_d.setZero();
}

void LocoCtrl::compute_contactSequence(RobotState &state, gaitTypeDef gaitType, double dt, stateMachineTypeDef &FSM)
{
    Gait *gait;
    if(gaitType == STAND) {
        gait = &stand;
    } else if(gaitType == LINE_WALK) {
        gait = &linewalk;
    } else if(gaitType == POINT_WALK) {
        gait = &pointwalk;
    } else if(gaitType == LINE_WALK2) {
        gait = &linewalk2;
    } else if(gaitType == POINT_WALK2) {
        gait = &pointwalk2;
    } else if(gaitType == SLIDE) {
        gait = &slide;
    } else {
        std::cout << "Invalid gait type" << std::endl;
    }
    gait_cur = gait;

    // Eigen::Vector2d contact_state;
    // contact_state = gait->getContactSubPhase();
    int iterationsBetweenMPC = 50;
    gait->setIterations(iterationsBetweenMPC, iterationCounter);

    Eigen::Vector4d contactStates = gait->getContactSubPhase();
    Eigen::Vector4d swingStates = gait->getSwingSubPhase();

    mpcTable = gait->mpc_gait();
			
    for(int foot=0; foot<4; foot++)
    {
        std::string footName = foot == 0 ? "rfoot" : "lfoot";
        double contactState = contactStates(foot);
        double swingState = swingStates(foot); 
        // std::cout << footName << "swing "  << ": " << swingState << std::endl;
        // std::cout << footName << "Contact "  << ": " << contactState << std::endl;
    }

    // Print the entire mpcTable_ array
    // std::cout << "mpcTable_['ltoe'] = ";
    // for (int i = 0; i < gait->get_nMPC_segments(); i++)
    // {
    //     std::cout << mpcTable[i*4] << " ";
    // }std::cout << std::endl;
    // std::cout << "mpcTable_['lwhl'] = ";
    // for (int i = 0; i < gait->get_nMPC_segments(); i++)
    // {
    //     std::cout << mpcTable[i*4+1] << " ";
    // }std::cout << std::endl;
    // std::cout << "mpcTable_['rtoe'] = ";
    // for (int i = 0; i < gait->get_nMPC_segments(); i++)
    // {
    //     std::cout << mpcTable[i*4+2] << " ";
    // }std::cout << std::endl;
    // std::cout << "mpcTable_['rwhl'] = ";
    // for (int i = 0; i < gait->get_nMPC_segments(); i++)
    // {
    //     std::cout << mpcTable[i*4+3] << " ";
    // }std::cout << std::endl << std::endl;

    // if(mpcTable[0] == 1 && mpcTable[1] == 1) {
    //     FSM = DOUBLE_STANCE;
    // } else if(mpcTable[0] == 0 && mpcTable[1] == 1) {
    //     FSM = LEFT_CONTACT;
    // } else if(mpcTable[0] == 1 && mpcTable[1] == 0) {
    //     FSM = RIGHT_CONTACT;
    // } else {
    //     std::cout << "Invalid contact state" << std::endl;
    // }


    /// @test option 2
    // Update contact window with sliding window approach
    for (int i = 0; i < 4; i++) {
        if (state.ctrl.contact_window[i].size() >= PLAN_HORIZON) {
            state.ctrl.contact_window[i].erase(state.ctrl.contact_window[i].begin());
        }
    }

    state.ctrl.contact_window[0].push_back(mpcTable[0]);
    state.ctrl.contact_window[1].push_back(mpcTable[1]);
    state.ctrl.contact_window[2].push_back(mpcTable[2]);
    state.ctrl.contact_window[3].push_back(mpcTable[3]);

    // Ensure size is exactly PLAN_HORIZON
    for (int i = 0; i < 4; i++) {
        while (state.ctrl.contact_window[i].size() < PLAN_HORIZON) {
            state.ctrl.contact_window[i].push_back(state.ctrl.contact_window[i].back());
        }
    }

    // Print the entire contact_window array
    // std::vector<string> foot_names = {"lwhl", "ltoe", "rwhl", "rtoe"};
    // std::cout << std::endl;
    // for (int i = 0; i < 4; i++) {
    //     std::cout << "contact_window[" << foot_names[i] << "] = ";
    //     for (int j = 0; j < PLAN_HORIZON; j++) {
    //         std::cout << state.ctrl.contact_window[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }


    if(state.ctrl.contact_window[1][0] == 1 && state.ctrl.contact_window[3][0] == 1) {
        if(state.ctrl.prevStateMachine != state.ctrl.StateMachine) {
            // state.ctrl.prevStateMachine = state.ctrl.StateMachine;
            // state.ctrl.StateMachine = DOUBLE_STANCE;
        }
        // state.ctrl.prevStateMachine = state.ctrl.StateMachine;
        // state.ctrl.StateMachine = DOUBLE_STANCE;
        FSM = DOUBLE_STANCE;
        // std::cout << "DOUBLE_STANCE" << std::endl;
    } else if(state.ctrl.contact_window[1][0] == 0 && state.ctrl.contact_window[3][0] == 1) {
        if(state.ctrl.prevStateMachine != state.ctrl.StateMachine) {
            state.ctrl.prevStateMachine = state.ctrl.StateMachine;
            // state.ctrl.StateMachine = LEFT_CONTACT;
        }
        // state.ctrl.prevStateMachine = state.ctrl.StateMachine;
        // state.ctrl.StateMachine = LEFT_CONTACT;
        FSM = LEFT_CONTACT;
        // std::cout << "LEFT_CONTACT" << std::endl;
    } else if(state.ctrl.contact_window[1][0] == 1 && state.ctrl.contact_window[3][0] == 0) {
        if(state.ctrl.prevStateMachine != state.ctrl.StateMachine) {
            state.ctrl.prevStateMachine = state.ctrl.StateMachine;
            // state.ctrl.StateMachine = RIGHT_CONTACT;
        }
        // state.ctrl.prevStateMachine = state.ctrl.StateMachine;
        // state.ctrl.StateMachine = RIGHT_CONTACT;
        FSM = RIGHT_CONTACT;
        // std::cout << "RIGHT_CONTACT" << std::endl;
    } else {
        std::cout << "Invalid contact state" << std::endl;
    }
    // std::cout << "FSM = " << FSM << std::endl;
}

void LocoCtrl::calcReference(RobotState &state, double mpc_dt)
{
    // Initial state - update the gradient and first row of the constraint matrix    
    mpc_states << state.fbk.euler_B(0), state.fbk.euler_B(1), state.fbk.euler_B(2),          // Base orientation
                    state.fbk.p_CoM(0), state.fbk.p_CoM(1), state.fbk.p_CoM(2),             // CoM?Base? position
                    state.fbk.omega_B(0), state.fbk.omega_B(1), state.fbk.omega_B(2),       // angular velocity
                    state.fbk.pdot_CoM(0), state.fbk.pdot_CoM(1), state.fbk.pdot_CoM(2),
                    -9.8;    // linear velocity
    
    state.ctrl.lin_vel_d_wrld = state.fbk.R_B * state.ctrl.lin_vel_d;
    state.ctrl.ang_vel_d_wrld = state.fbk.R_B * state.ctrl.ang_vel_d;   // not sure for now
    state.ctrl.euler_B_d(0) = 0.0;
    state.ctrl.euler_B_d(1) = 0.0;
    state.ctrl.euler_B_d(2) += state.ctrl.ang_vel_d(2) * 0.05;

    // Future reference state 
    mpc_states_d.resize(13 * PLAN_HORIZON);
    for(int i = 0; i < PLAN_HORIZON; ++i) {
        // Calculate predicted state 
        mpc_states_d.segment<13>(i * 13) <<
                state.ctrl.euler_B_d(0),
                state.ctrl.euler_B_d(1),
                state.ctrl.euler_B_d(2),
                // state.fbk.euler_B(2) + state.ctrl.ang_vel_d[2] * mpc_dt * (i),
                state.fbk.p_CoM(0) + state.ctrl.lin_vel_d_wrld[0] * mpc_dt * (i),
                state.fbk.p_CoM(1) + state.ctrl.lin_vel_d_wrld[1] * mpc_dt * (i),
                state.ctrl.p_CoM_d(2), // 0.7?
                state.ctrl.ang_vel_d_wrld(0),   // not sure for now, prev: ang_vel_d(0);
                state.ctrl.ang_vel_d_wrld(1),   // not sure for now, prev: ang_vel_d(1);
                state.ctrl.ang_vel_d_wrld(2),   // not sure for now, prev: ang_vel_d(2);
                state.ctrl.lin_vel_d_wrld(0),
                state.ctrl.lin_vel_d_wrld(1),
                state.ctrl.lin_vel_d_wrld(2),
                -9.8;
        
        // Linearize about reference (calcualte A and B matrices)
        mpc_solver.calculate_A_mat_c(mpc_states_d.segment<3>(0));  // it should be roll pitch yaw order
        mpc_solver.calculate_B_mat_c(state.param.robot_mass,
                                    state.param.trunk_inertia,
                                    state.fbk.R_B,
                                    state.fbk.foot_pos_abs);    
        // state.fbk.foot_pos_abs.block<3, 1>(0, 0) = state.fbk.foot_pos_abs.block<3, 1>(0, 0) + state.ctrl.lin_vel_d * mpc_dt;
        // state.fbk.foot_pos_abs.block<3, 1>(0, 1) = state.fbk.foot_pos_abs.block<3, 1>(0, 1) + state.ctrl.lin_vel_d * mpc_dt;
        // state.fbk.foot_pos_abs.block<3, 1>(0, 2) = state.fbk.foot_pos_abs.block<3, 1>(0, 2) + state.ctrl.lin_vel_d * mpc_dt;
        // state.fbk.foot_pos_abs.block<3, 1>(0, 3) = state.fbk.foot_pos_abs.block<3, 1>(0, 3) + state.ctrl.lin_vel_d * mpc_dt;
        mpc_solver.state_space_discretization(mpc_dt);

        mpc_solver.B_mat_d_list.block<13, 12>(i * 13, 0) = mpc_solver.B_mat_d;
    }
}

auto LocoCtrl::compute_grf(RobotState &state) -> Eigen::Vector<double, MPC_STATE_DIM * PLAN_HORIZON>
{
    // set Q and R weights
    state.param.q_weights = Eigen::VectorXd(13).setZero();
    state.param.r_weights = Eigen::VectorXd(12).setZero();

    state.param.q_weights << 100.0, 100.0, 1000.0,
                            10.0, 10.0, 2700.0,
                            20.0, 20.0, 200.0,
                            20.0, 20.0, 20.0,
                            0.0;
    state.param.r_weights << 1e-5, 1e-5, 1e-6,
                                1e-5, 1e-5, 1e-6,
                                1e-5, 1e-5, 1e-6,
                                1e-5, 1e-5, 1e-6;

    // state.param.q_weights << 1.0, 1.0, 1.0,
    //                         0.0, 0.0, 10.0,
    //                         0.0, 0.0, 1.0,
    //                         1.0, 1.0, 1.0,
    //                         0.0;
    // state.param.r_weights << 1e-5, 1e-5, 1e-6,
    //                          1e-5, 1e-5, 1e-6,
    //                          1e-5, 1e-5, 1e-6,
    //                          1e-5, 1e-5, 1e-6;

    /// @test option 1
    // state.ctrl.contact_window.resize(4);
    // state.ctrl.contact_window[0].clear();
    // state.ctrl.contact_window[1].clear();
    // state.ctrl.contact_window[2].clear();
    // state.ctrl.contact_window[3].clear();
    // for(int i=0; i<PLAN_HORIZON; i++)
    // {
    //     state.ctrl.contact_window[0].push_back(mpcTable[i*2]);
    //     // state.ctrl.contact_window[0].push_back(false);
    //     state.ctrl.contact_window[1].push_back(mpcTable[i*2]);
    //     state.ctrl.contact_window[2].push_back(mpcTable[i*2+1]);
    //     // state.ctrl.contact_window[2].push_back(false);
    //     state.ctrl.contact_window[3].push_back(mpcTable[i*2+1]);

    //     // state.ctrl.contact_window[0].push_back(true);
    //     // state.ctrl.contact_window[1].push_back(true);
    //     // state.ctrl.contact_window[2].push_back(true);
    //     // state.ctrl.contact_window[3].push_back(true);
    // }
    
    // /// @test option 2
    // // Update contact window with sliding window approach
    // for (int i = 0; i < 4; i++) {
    //     if (state.ctrl.contact_window[i].size() >= PLAN_HORIZON) {
    //         state.ctrl.contact_window[i].erase(state.ctrl.contact_window[i].begin());
    //     }
    // }

    // state.ctrl.contact_window[0].push_back(mpcTable[0]);
    // state.ctrl.contact_window[1].push_back(mpcTable[1]);
    // state.ctrl.contact_window[2].push_back(mpcTable[2]);
    // state.ctrl.contact_window[3].push_back(mpcTable[3]);

    // // Ensure size is exactly PLAN_HORIZON
    // for (int i = 0; i < 4; i++) {
    //     while (state.ctrl.contact_window[i].size() < PLAN_HORIZON) {
    //         state.ctrl.contact_window[i].push_back(state.ctrl.contact_window[i].back());
    //     }
    // }

    // // Print the entire contact_window array
    // std::vector<string> foot_names = {"lwhl", "ltoe", "rwhl", "rtoe"};
    // std::cout << std::endl;
    // for (int i = 0; i < 4; i++) {
    //     std::cout << "contact_window[" << foot_names[i] << "] = ";
    //     for (int j = 0; j < PLAN_HORIZON; j++) {
    //         std::cout << state.ctrl.contact_window[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }


    // calculate QP matrices
    mpc_solver.calculate_qp_mats(state, mpc_states, mpc_states_d);

    // solve QP
    if (!solver.isInitialized()) {
        std::cout << "[initSolver]" << std::endl;
        solver.settings()->setVerbosity(false);
        solver.settings()->setWarmStart(true);	// false? or true?
        solver.data()->setNumberOfVariables(NUM_DOF * PLAN_HORIZON);
        solver.data()->setNumberOfConstraints(MPC_CONSTRAINT_DIM * PLAN_HORIZON);
        solver.data()->setLinearConstraintsMatrix(mpc_solver.linear_constraints);
        solver.data()->setHessianMatrix(mpc_solver.hessian);
        solver.data()->setGradient(mpc_solver.gradient);
        solver.data()->setLowerBound(mpc_solver.lb);
        solver.data()->setUpperBound(mpc_solver.ub);
        solver.initSolver();
    } else {
        solver.updateHessianMatrix(mpc_solver.hessian);
        solver.updateGradient(mpc_solver.gradient);
        solver.updateLowerBound(mpc_solver.lb);
        solver.updateUpperBound(mpc_solver.ub);
    }
    solver.solve();

    return solver.getSolution();
}

void LocoCtrl::compute_nextfoot(RobotState &state, stateMachineTypeDef &FSM, double cur_sw_time)
{
    Eigen::Matrix3d K_raibert;
    double K_cp = 1.0;

    K_raibert.setZero();
    K_raibert(0, 0) = 0.15;
    K_raibert(1, 1) = 0.15;
    // K_raibert(2, 2) = 0.15;
    Eigen::Vector3d offset;
    double pelvis_width = 0.257;
    // double pelvis_width = 0.37;

    if(gait_cur == &linewalk || gait_cur == &linewalk2 || gait_cur == &slide) {
        // std::cout << "linewalk" << std::endl;
        offset << (FSM == LEFT_CONTACT ? 0.1 : -0.1), pelvis_width/2, 0;
        prev_gait = gait_cur;
    } else if(gait_cur == &pointwalk || gait_cur == &pointwalk2) {
        // std::cout << "pointwalk" << std::endl;
        offset << 0, pelvis_width/2, 0;
        prev_gait = gait_cur;
    } else if(gait_cur == &stand) {
        // std::cout << "stand" << std::endl;
        if(prev_gait == &linewalk || prev_gait == &linewalk2 || prev_gait == &slide) {
            // std::cout << "linewalk" << std::endl;
            offset << (FSM == LEFT_CONTACT ? 0.1 : -0.1), pelvis_width/2, 0;
        } else if(prev_gait == &pointwalk || prev_gait == &pointwalk2) {
            // std::cout << "pointwalk" << std::endl;
            offset << 0, pelvis_width/2, 0;
        }
        // prev_gait = gait_cur;
    }

    // offset << 0, pelvis_width/2, 0;
    // offset << (FSM == LEFT_CONTACT ? 0.1 : -0.1), pelvis_width/2, 0;
    // offset << (FSM == LEFT_CONTACT ? 0.065 : -0.065), pelvis_width/2, 0;

    offset = state.fbk.R_B * offset;
    if(FSM == LEFT_CONTACT)
    {
        state.ctrl.p_footplacement_target = 
            -offset + state.fbk.p_CoM 
            + 0.3 / 2 * state.fbk.pdot_CoM + K_raibert * (state.fbk.pdot_CoM - state.fbk.R_B * state.ctrl.lin_vel_d)
            // + (K_cp * sqrt(0.7/9.8)) * state.fbk.pdot_CoM.cross(state.ctrl.ang_vel_d);
            + (K_cp * sqrt(0.7/9.8)) * state.ctrl.ang_vel_d.cross(state.fbk.pdot_CoM);
        // state.ctrl.p_footplacement_target = state.fbk.p_CoM + 0.3 / 2 * state.fbk.pdot_CoM + K_raibert * (state.fbk.pdot_CoM - state.ctrl.lin_vel_d);
        // state.ctrl.p_footplacement_target = -offset + state.fbk.p_CoM + 0.3 / 2 * state.fbk.pdot_CoM + K_raibert * (state.fbk.pdot_CoM - state.ctrl.lin_vel_d) 
        //                                             + 1/2 * sqrt(0.6/9.8) * state.fbk.pdot_CoM;
        state.ctrl.p_footplacement_target(2) = 0.0;
    }
    else if(FSM == RIGHT_CONTACT)
    {
        state.ctrl.p_footplacement_target = 
            offset + state.fbk.p_CoM 
            + 0.3 / 2 * state.fbk.pdot_CoM + K_raibert * (state.fbk.pdot_CoM - state.fbk.R_B * state.ctrl.lin_vel_d)
            // + (K_cp * sqrt(0.7/9.8)) * state.fbk.pdot_CoM.cross(state.ctrl.ang_vel_d);
            + (K_cp * sqrt(0.7/9.8)) * state.ctrl.ang_vel_d.cross(state.fbk.pdot_CoM);
        // state.ctrl.p_footplacement_target = state.fbk.p_CoM + 0.3 / 2 * state.fbk.pdot_CoM + K_raibert * (state.fbk.pdot_CoM - state.ctrl.lin_vel_d);
        // state.ctrl.p_footplacement_target = offset + state.fbk.p_CoM + 0.3 / 2 * state.fbk.pdot_CoM + K_raibert * (state.fbk.pdot_CoM - state.ctrl.lin_vel_d) 
        //                                             + 1/2 * sqrt(0.6/9.8) * state.fbk.pdot_CoM;
        state.ctrl.p_footplacement_target(2) = 0.0;
    }
    else
    {
        // std::cout << "Invalid FSM" << std::endl;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // input: step duration, initial step time, current step time, step duration
    // output: swing trajectory
    ////////////////////////////////////////////////////////////////////////////////
    Eigen::Vector3d lprevious_support_foot_position_;
    Eigen::Vector3d rprevious_support_foot_position_;
    if(cur_sw_time < 0.002)
    {
        if(FSM == LEFT_CONTACT)
        {
            state.fbk.p_C[1](2) = 0.0;
            rprevious_support_foot_position_ = state.fbk.p_C[1]; // 0: right toe, 1: right wheel, 2: left toe, 3: left wheel
            right_foot_position = state.fbk.p_C[1];
            // std::cout << "rprevious_support_foot_position_: " << rprevious_support_foot_position_.transpose()<< std::endl;
        }
        else if(FSM == RIGHT_CONTACT)
        {
            state.fbk.p_C[3](2) = 0.0;
            lprevious_support_foot_position_ = state.fbk.p_C[3];
            left_foot_position = state.fbk.p_C[3];
            // std::cout << "lprevious_support_foot_position_: " << lprevious_support_foot_position_.transpose() << std::endl;
        }
    }

    double init_step_time = 0.0;
    double curr_step_time = cur_sw_time;
    double step_duration = 0.3;      // step duration
    if(FSM == LEFT_CONTACT) 
    {
    	if(curr_step_time < step_duration - 0.01)
    	{
    		optTraj.compute(rprevious_support_foot_position_, 
    							right_foot_position, right_foot_velocity, right_foot_acceleration, 
    							state.ctrl.p_footplacement_target, 
    							init_step_time, curr_step_time, step_duration);
    	}
    	optTraj.get_next_state(curr_step_time + 0.001, 
    						right_foot_position, right_foot_velocity, right_foot_acceleration);
        
        // std::cout << "curr_step_time: " << curr_step_time << std::endl;

        state.fbk.p_C[3](2) = 0.0;
        left_foot_position = state.fbk.p_C[3];
        left_foot_velocity.setZero();
        left_foot_acceleration.setZero();
    }
    else if(FSM == RIGHT_CONTACT)
    {
        if(curr_step_time < step_duration - 0.01)
    	{
            optTraj.compute(lprevious_support_foot_position_, 
                                left_foot_position, left_foot_velocity, left_foot_acceleration, 
                                state.ctrl.p_footplacement_target, 
                                init_step_time, curr_step_time, step_duration);
    	}
    	optTraj.get_next_state(curr_step_time + 0.001, 
    						left_foot_position, left_foot_velocity, left_foot_acceleration);
        
        // std::cout << "curr_step_time: " << curr_step_time << std::endl;

        state.fbk.p_C[1](2) = 0.0;
        right_foot_position = state.fbk.p_C[1];
        right_foot_velocity.setZero();
        right_foot_acceleration.setZero();
    }
    // std::cout << cur_sw_time << std::endl;

    int n_sliding_segments = 8;
    if(cur_sw_time > 0.299) {
        if(FSM == LEFT_CONTACT) {
            p_whl_slide = right_foot_position;
            p_whl_slide_ref = right_foot_position + n_sliding_segments * (state.fbk.R_B * state.ctrl.lin_vel_d * 0.05);
            sliding_traj.is_moving_ = false;
        } else if(FSM == RIGHT_CONTACT) {
            p_whl_slide = left_foot_position;
            p_whl_slide_ref = left_foot_position + n_sliding_segments * (state.fbk.R_B * state.ctrl.lin_vel_d * 0.05);
            sliding_traj.is_moving_ = false;
        }
        // std::cout << RED << "p_whl_slide_ref: " << p_whl_slide_ref.transpose() << RESET << std::endl;
    }

    double whl_rad = 0.079;
    if(gait_cur == &slide && FSM == DOUBLE_STANCE) {
        if(sliding_traj.is_moving_ == false) {
		    sliding_traj.setTargetPosition(p_whl_slide, p_whl_slide_ref, 0.2, 1 / 1000.0, QUINTIC);
        }
        sliding_traj.computeTraj(state.ctrl.p_sw_d, state.ctrl.pdot_sw_d, state.ctrl.pddot_sw_d);
        state.ctrl.whl_ang_pos_d = (1 / whl_rad) * state.ctrl.p_sw_d(0);
        state.ctrl.whl_ang_vel_d = (1 / whl_rad) * state.ctrl.pdot_sw_d(0);
        state.ctrl.whl_ang_acc_d = (1 / whl_rad) * state.ctrl.pddot_sw_d(0);
        // std::cout << BLUE << "state.ctrl.p_sw_d: " << state.ctrl.p_sw_d.transpose() << RESET << std::endl;

        sliding_traj.computeTraj(p_whl_slide_d, pdot_whl_slide_d, pddot_whl_slide_d);
        state.ctrl.whl_ang_acc_d = 1 / whl_rad * pddot_whl_slide_d(0);
        // std::cout << "whl_ang_acc_d: " << state.ctrl.whl_ang_acc_d << std::endl;
    }
}