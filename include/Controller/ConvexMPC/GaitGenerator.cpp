#include "GaitGenerator.hpp"

// Constructor: Initializes gait parameters using provided values.
Gait::Gait(int nMPC_segments, Eigen::Vector4i offsets, Eigen::Vector4i durations, const std::string &name) : offsets_(offsets.array()), 
                                                                                                 durations_(durations.array()),
                                                                                                 nIterations_(nMPC_segments)
{
    nMPC_segments_ = nMPC_segments;
    mpcTable_ = new int[nMPC_segments * 4];
    offsetsPhase_ = offsets.cast<double>() / (double)nMPC_segments;
    durationsPhase_ = durations.cast<double>() / (double)nMPC_segments;
    stance_ = durations[0];
    swing_ = nMPC_segments - durations[0];
}

Gait::~Gait()
{
    delete[] mpcTable_;
}

Eigen::Vector4d Gait::getContactSubPhase()
{
    Eigen::Array4d progress = phase_ - offsetsPhase_;

    for (int i = 0; i < 4; i++)
    {
        if (progress[i] < 0)
            progress[i] += 1.;
        if (progress[i] > durationsPhase_[i])
        {
            progress[i] = 0.;
        }
        else
        {
            progress[i] = progress[i] / durationsPhase_[i];
        }
    }

    return progress.matrix();
}

Eigen::Vector4d Gait::getSwingSubPhase()
{
  Eigen::Array4d swing_offset = offsetsPhase_ + durationsPhase_; 
  for (int i = 0; i < 4; i++)
    if (swing_offset[i] > 1)
      swing_offset[i] -= 1.;

  Eigen::Array4d swing_duration = 1. - durationsPhase_;

  Eigen::Array4d progress = phase_ - swing_offset;

  for (int i = 0; i < 4; i++)
  {
    if (progress[i] < 0)
      progress[i] += 1.;
    if (progress[i] > swing_duration[i])
    {
      progress[i] = 0.;
    }
    else
    {
      progress[i] = progress[i] / swing_duration[i];
    }
  }

  return progress.matrix();
}

// Generate and return the MPC gait table.
int *Gait::mpc_gait()
{
  for (int i = 0; i < nIterations_; i++)
  {
    int iter = (i + iteration_) % nIterations_;
    Eigen::Array4i progress = iter - offsets_; // 0 5
    for (int j = 0; j < 4; j++)
    {
      if (progress[j] < 0)
        progress[j] += nIterations_;
      if (progress[j] < durations_[j])
        mpcTable_[i * 4 + j] = 1;
      else
        mpcTable_[i * 4 + j] = 0;
    }
  }

  // // Print the entire mpcTable_ array
  // std::cout << "mpcTable_['lfoot'] = ";
  // for (int i = 0; i < 12; i++)
  // {
  //   std::cout << mpcTable_[i+1] << " ";
  // }std::cout << std::endl;
  // std::cout << "mpcTable_['rfoot'] = ";
  // for (int i = 0; i < 12; i++)
  // {
  //   std::cout << mpcTable_[i*2+1] << " ";
  // }std::cout << std::endl;

  return mpcTable_;
}

// Update iteration and phase based on the given values.
void Gait::setIterations(int iterationsPerMPC, int currentIteration)
{
  iteration_ = (currentIteration / iterationsPerMPC) % nIterations_;
  phase_ = (double)(currentIteration % (iterationsPerMPC * nIterations_)) / (double)(iterationsPerMPC * nIterations_);
}