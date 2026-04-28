#pragma once

#include <iostream>
#include <Eigen/Dense>

class Gait
{
private:
    int* mpcTable_;
    int nMPC_segments_;
    
    Eigen::Array4i offsets_;           // offset in mpc segments
    Eigen::Array4i durations_;         // duration of step in mpc segments
    Eigen::Array4d offsetsPhase_;      // offsets in phase (0 to 1)
    Eigen::Array4d durationsPhase_;    // durations in phase (0 to 1)

    int iteration_;
    int nIterations_;
    int currentIteration_;
    double phase_;

public:
    Gait(int nMPC_segments, Eigen::Vector4i offsets, Eigen::Vector4i durations, const std::string &name);
    ~Gait();

    int stance_;
    int swing_;

    Eigen::Vector4d getContactSubPhase();
    Eigen::Vector4d getSwingSubPhase();
    int* mpc_gait();
    void setIterations(int iterationsPerMPC, int currentIteration);

    inline int get_nMPC_segments() { return nMPC_segments_; }
};