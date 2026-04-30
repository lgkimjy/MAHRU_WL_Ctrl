#pragma once

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <hdf5.h>

#include "RobotStates.hpp"

class HDF5Logger {
public:
    explicit HDF5Logger(const std::string& filename) : filename_(filename)
    {
        running_ = true;
        writer_thread_ = std::thread(&HDF5Logger::writerLoop, this);
    }

    ~HDF5Logger()
    {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            running_ = false;
        }
        condition_.notify_one();

        if (writer_thread_.joinable()) {
            writer_thread_.join();
        }
    }

    HDF5Logger(const HDF5Logger&) = delete;
    HDF5Logger& operator=(const HDF5Logger&) = delete;

    void log(double time, const RobotData& state)
    {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            log_queue_.push({time, state});
        }
        condition_.notify_one();
    }

private:
    struct LogEntry {
        double time;
        RobotData state;
    };

    struct Dataset {
        hid_t id = -1;
        size_t offset = 0;
        size_t dim = 0;
    };

    std::string filename_;

    std::thread writer_thread_;
    std::queue<LogEntry> log_queue_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool running_ = false;

    hid_t file_ = -1;
    std::unordered_map<std::string, Dataset> datasets_;
    std::unordered_map<std::string, std::vector<std::vector<double>>> buffers_;
    static constexpr size_t CHUNK_SIZE = 50;

    void writerLoop()
    {
        file_ = H5Fcreate(filename_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_ < 0) {
            std::cerr << "[ HDF5Logger ] Failed to create " << filename_ << std::endl;
            return;
        }
        std::cout << "[ HDF5Logger ] logging to " << filename_ << std::endl;

        while (true) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return !log_queue_.empty() || !running_; });

            if (!running_ && log_queue_.empty()) break;

            LogEntry entry = std::move(log_queue_.front());
            log_queue_.pop();
            lock.unlock();

            bufferData(entry);
        }

        flushAllBuffers();
        closeFile();
    }

    void bufferData(const LogEntry& entry)
    {
        addToBuffer("time", entry.time);

        addToBuffer("fbk/qpos", entry.state.fbk.qpos);
        addToBuffer("fbk/qvel", entry.state.fbk.qvel);
        addToBuffer("fbk/p_B", entry.state.fbk.p_B);
        addToBuffer("fbk/pdot_B", entry.state.fbk.pdot_B);
        addToBuffer("fbk/quat_B", entry.state.fbk.quat_B.coeffs());
        addToBuffer("fbk/R_B", entry.state.fbk.R_B);
        addToBuffer("fbk/varphi_B", entry.state.fbk.varphi_B);
        addToBuffer("fbk/omega_B", entry.state.fbk.omega_B);
        addToBuffer("fbk/p_CoM", entry.state.fbk.p_CoM);
        addToBuffer("fbk/pdot_CoM", entry.state.fbk.pdot_CoM);
        addContactArrayToBuffer("fbk/p_C", entry.state.fbk.p_C);
        addContactArrayToBuffer("fbk/pdot_C", entry.state.fbk.pdot_C);
        addToBuffer("fbk/jpos", entry.state.fbk.jpos);
        addToBuffer("fbk/jvel", entry.state.fbk.jvel);

        addToBuffer("ctrl/lin_vel_d", entry.state.ctrl.lin_vel_d);
        addToBuffer("ctrl/ang_vel_d", entry.state.ctrl.ang_vel_d);
        addToBuffer("ctrl/jpos_d", entry.state.ctrl.jpos_d);
        addToBuffer("ctrl/jvel_d", entry.state.ctrl.jvel_d);
        addToBuffer("ctrl/torq_d", entry.state.ctrl.torq_d);
        addToBuffer("ctrl/p_CoM_d", entry.state.ctrl.p_CoM_d);
        addToBuffer("ctrl/pdot_CoM_d", entry.state.ctrl.pdot_CoM_d);
        addToBuffer("ctrl/roll_momentum_rate_d", entry.state.ctrl.roll_momentum_rate_d);
        addToBuffer("ctrl/roll_momentum_rate_actual", entry.state.ctrl.roll_momentum_rate_actual);
        addToBuffer("ctrl/roll_momentum_rate_wbc", entry.state.ctrl.roll_momentum_rate_wbc);
        addToBuffer("ctrl/roll_momentum_rate_wbc_error", entry.state.ctrl.roll_momentum_rate_wbc_error);
        addToBuffer("ctrl/swing_leg_roll_momentum_rate_d",
                    entry.state.ctrl.swing_leg_roll_momentum_rate_d);
        addToBuffer("ctrl/swing_leg_roll_momentum_rate_wbc",
                    entry.state.ctrl.swing_leg_roll_momentum_rate_wbc);
        addToBuffer("ctrl/roll_momentum_y_err", entry.state.ctrl.roll_momentum_y_err);
        addToBuffer("ctrl/roll_momentum_ydot_err", entry.state.ctrl.roll_momentum_ydot_err);
        addToBuffer("ctrl/roll_momentum_height", entry.state.ctrl.roll_momentum_height);
        addToBuffer("ctrl/unicycle_state_time", entry.state.ctrl.unicycle_state_time);
        addToBuffer("ctrl/right_foot_lift_phase", entry.state.ctrl.right_foot_lift_phase);
        addToBuffer("ctrl/swing_leg_reaction_offset", entry.state.ctrl.swing_leg_reaction_offset);
        addToBuffer("ctrl/swing_leg_reaction_vel", entry.state.ctrl.swing_leg_reaction_vel);
        addToBuffer("ctrl/swing_lateral_acceleration_d",
                    entry.state.ctrl.swing_lateral_acceleration_d);
        addToBuffer("ctrl/single_wheel_pitch", entry.state.ctrl.single_wheel_pitch);
        addToBuffer("ctrl/single_wheel_pitch_rate", entry.state.ctrl.single_wheel_pitch_rate);
        addToBuffer("ctrl/single_wheel_lin_vel_d", entry.state.ctrl.single_wheel_lin_vel_d);
        addToBuffer("ctrl/single_wheel_lin_acc_d", entry.state.ctrl.single_wheel_lin_acc_d);
        addToBuffer("ctrl/single_wheel_com_offset_d", entry.state.ctrl.single_wheel_com_offset_d);
        addToBuffer("ctrl/single_wheel_phase", entry.state.ctrl.single_wheel_phase);
        addToBuffer("ctrl/single_wheel_stance_qdot_d", entry.state.ctrl.single_wheel_stance_qdot_d);
        addToBuffer("ctrl/single_wheel_stance_qdot", entry.state.ctrl.single_wheel_stance_qdot);
        addToBuffer("ctrl/single_wheel_lateral_vel", entry.state.ctrl.single_wheel_lateral_vel);

        addToBuffer("param/Kp", entry.state.param.Kp);
        addToBuffer("param/Kd", entry.state.param.Kd);
    }

    void addContactArrayToBuffer(
        const std::string& name,
        const std::array<Eigen::Vector3d, 4>& value)
    {
        Eigen::Matrix<double, 4, 3> data;
        for (int i = 0; i < 4; ++i) {
            data.row(i) = value[i].transpose();
        }
        addToBuffer(name, data);
    }

    template <typename Derived>
    void addToBuffer(const std::string& name, const Eigen::MatrixBase<Derived>& value)
    {
        std::vector<double> data(value.size());
        Eigen::Map<Eigen::VectorXd>(data.data(), value.size()) =
            Eigen::Map<const Eigen::VectorXd>(value.derived().data(), value.size());

        buffers_[name].push_back(std::move(data));
        if (buffers_[name].size() >= CHUNK_SIZE) {
            flushDataset(name);
        }
    }

    void addToBuffer(const std::string& name, double value)
    {
        buffers_[name].push_back(std::vector<double>{value});
        if (buffers_[name].size() >= CHUNK_SIZE) {
            flushDataset(name);
        }
    }

    void flushDataset(const std::string& name)
    {
        auto& buffer = buffers_[name];
        if (buffer.empty() || file_ < 0) return;

        const size_t num_rows = buffer.size();
        const size_t dim = buffer.front().size();
        Dataset& dataset = getOrCreateDataset(name, dim);

        std::vector<double> flat;
        flat.reserve(num_rows * dim);
        for (const auto& row : buffer) {
            flat.insert(flat.end(), row.begin(), row.end());
        }

        const hsize_t new_dims[2] = {
            static_cast<hsize_t>(dataset.offset + num_rows),
            static_cast<hsize_t>(dim)
        };
        H5Dset_extent(dataset.id, new_dims);

        hid_t file_space = H5Dget_space(dataset.id);
        const hsize_t start[2] = {static_cast<hsize_t>(dataset.offset), 0};
        const hsize_t count[2] = {
            static_cast<hsize_t>(num_rows),
            static_cast<hsize_t>(dim)
        };
        H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count, nullptr);

        const hsize_t mem_dims[2] = {
            static_cast<hsize_t>(num_rows),
            static_cast<hsize_t>(dim)
        };
        hid_t mem_space = H5Screate_simple(2, mem_dims, nullptr);
        H5Dwrite(dataset.id, H5T_NATIVE_DOUBLE, mem_space, file_space, H5P_DEFAULT, flat.data());

        H5Sclose(mem_space);
        H5Sclose(file_space);

        dataset.offset += num_rows;
        buffer.clear();
    }

    Dataset& getOrCreateDataset(const std::string& name, size_t dim)
    {
        auto it = datasets_.find(name);
        if (it != datasets_.end()) {
            if (it->second.dim != dim) {
                throw std::runtime_error("[HDF5Logger] Dataset width changed: " + name);
            }
            return it->second;
        }

        createParentGroup(name);

        const hsize_t dims[2] = {0, static_cast<hsize_t>(dim)};
        const hsize_t max_dims[2] = {H5S_UNLIMITED, static_cast<hsize_t>(dim)};
        hid_t dataspace = H5Screate_simple(2, dims, max_dims);

        hid_t props = H5Pcreate(H5P_DATASET_CREATE);
        const hsize_t chunk[2] = {CHUNK_SIZE, static_cast<hsize_t>(dim)};
        H5Pset_chunk(props, 2, chunk);
        if (H5Zfilter_avail(H5Z_FILTER_DEFLATE) > 0) {
            H5Pset_deflate(props, 4);
        }

        hid_t id = H5Dcreate2(file_, name.c_str(), H5T_NATIVE_DOUBLE, dataspace,
                              H5P_DEFAULT, props, H5P_DEFAULT);
        H5Pclose(props);
        H5Sclose(dataspace);

        if (id < 0) {
            throw std::runtime_error("[HDF5Logger] Failed to create dataset: " + name);
        }

        auto [inserted, _] = datasets_.emplace(name, Dataset{id, 0, dim});
        return inserted->second;
    }

    void createParentGroup(const std::string& dataset_name)
    {
        const size_t pos = dataset_name.find_last_of('/');
        if (pos == std::string::npos || pos == 0) return;

        const std::string group_name = dataset_name.substr(0, pos);
        if (H5Lexists(file_, group_name.c_str(), H5P_DEFAULT) > 0) return;

        hid_t group = H5Gcreate2(file_, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (group >= 0) {
            H5Gclose(group);
        }
    }

    void flushAllBuffers()
    {
        for (auto& [name, _] : buffers_) {
            flushDataset(name);
        }
        if (file_ >= 0) {
            H5Fflush(file_, H5F_SCOPE_GLOBAL);
        }
    }

    void closeFile()
    {
        for (auto& [_, dataset] : datasets_) {
            if (dataset.id >= 0) {
                H5Dclose(dataset.id);
            }
        }
        datasets_.clear();

        if (file_ >= 0) {
            H5Fclose(file_);
            file_ = -1;
        }
    }
};
