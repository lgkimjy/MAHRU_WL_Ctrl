#pragma once

#include <eigen-quadprog/QuadProg.h>
#include <Eigen/Eigen>

#include <cmath>
#include <string>

class OnlineOptTraj
{
public:
    OnlineOptTraj();

    ~OnlineOptTraj();

    bool compute(const Eigen::Ref<const Eigen::Vector3d> &start_pose,
                 const Eigen::Ref<const Eigen::Vector3d> &current_pose,
                 const Eigen::Ref<const Eigen::Vector3d> &current_velocity,
                 const Eigen::Ref<const Eigen::Vector3d> &current_acceleration,
                 const Eigen::Ref<const Eigen::Vector3d> &target_pose,
                 const double &start_time,
                 const double &current_time,
                 const double &end_time);

    void get_next_state(const double &next_time,
                        Eigen::Ref<Eigen::Vector3d> next_pose,
                        Eigen::Ref<Eigen::Vector3d> next_velocity,
                        Eigen::Ref<Eigen::Vector3d> next_acceleration);

    void print_solver() const;

    std::string to_string() const;

    double get_mid_air_height() {
        return mid_air_height_;
    }
    double get_last_end_time_taken_into_account() {
        return last_end_time_seen_;
    }

    void set_mid_air_height(double mid_air_height) {
        mid_air_height_ = mid_air_height;
    }

    void set_costs(double cost_x,
                   double cost_y,
                   double cost_z,
                   double hess_regularization)
    {
        cost_x_ = cost_x;
        cost_y_ = cost_y;
        cost_z_ = cost_z;
        Q_regul_ = Eigen::MatrixXd::Identity(nb_var_, nb_var_) * hess_regularization;
    }
private:

    double mid_air_height_;
    int nb_var_x_, nb_var_y_, nb_var_z_;

    Eigen::VectorXd time_vec_x_;
    Eigen::VectorXd time_vec_y_;
    Eigen::VectorXd time_vec_z_;
    Eigen::Vector3d start_pose_;
    Eigen::Vector3d current_pose_;
    Eigen::Vector3d previous_solution_pose_;
    Eigen::Vector3d current_velocity_;
    Eigen::Vector3d current_acceleration_;
    Eigen::Vector3d target_pose_;

    double start_time_;
    double current_time_;
    double end_time_;
    double last_end_time_seen_;
    int nb_var_;
    int nb_eq_;
    int nb_ineq_;

    Eigen::QuadProgDense qp_solver_;

    Eigen::VectorXd x_opt_;
    Eigen::VectorXd x_opt_lb_;
    Eigen::VectorXd x_opt_ub_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd Q_regul_;

    double cost_x_, cost_y_, cost_z_;
    Eigen::VectorXd q_;
    Eigen::MatrixXd A_eq_;
    Eigen::VectorXd B_eq_;

    Eigen::MatrixXd A_ineq_;
    Eigen::VectorXd B_ineq_;

    void t_vec(const double &time, Eigen::VectorXd &time_vec)
    {
        time_vec(0) = 1.0;
        for (int i = 1; i < time_vec.size(); ++i)
        {
            time_vec(i) = std::pow(time, i);
        }
    }

    void dt_vec(const double &time, Eigen::VectorXd &time_vec)
    {
        time_vec(0) = 0.0;
        time_vec(1) = 1.0;
        for (int i = 2; i < time_vec.size(); ++i)
        {
            double id = i;
            time_vec(i) = id * std::pow(time, i - 1);
        }
    }

    void ddt_vec(const double &time, Eigen::VectorXd &time_vec)
    {
        time_vec(0) = 0.0;
        time_vec(1) = 0.0;
        time_vec(2) = 2.0;
        for (int i = 3; i < time_vec.size(); ++i)
        {
            double id = i;
            time_vec(i) = id * (id - 1.0) * std::pow(time, i - 2);
        }
    }
};
