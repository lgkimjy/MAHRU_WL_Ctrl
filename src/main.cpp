#include "MJSimulationBridge.hpp"

#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    bool sim = true;
    bool headless = false;
    double duration_s = 8.0;
    double stop_com_z = -1.0;
    double stop_roll_deg = -1.0;
    double stop_pitch_deg = -1.0;
    double max_wall_time_s = -1.0;
    double progress_interval_s = -1.0;
    int log_stride = 1;
    std::string log_dir_override;
    std::string scene_path = std::string(CMAKE_SOURCE_DIR) + std::string(mahru::scene);

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--headless") {
            headless = true;
        } else if (arg == "--duration" && i + 1 < argc) {
            duration_s = std::stod(argv[++i]);
        } else if (arg == "--log-stride" && i + 1 < argc) {
            log_stride = std::stoi(argv[++i]);
        } else if (arg == "--stop-com-z" && i + 1 < argc) {
            stop_com_z = std::stod(argv[++i]);
        } else if (arg == "--stop-roll-deg" && i + 1 < argc) {
            stop_roll_deg = std::stod(argv[++i]);
        } else if (arg == "--stop-pitch-deg" && i + 1 < argc) {
            stop_pitch_deg = std::stod(argv[++i]);
        } else if (arg == "--max-wall-time" && i + 1 < argc) {
            max_wall_time_s = std::stod(argv[++i]);
        } else if (arg == "--progress-interval" && i + 1 < argc) {
            progress_interval_s = std::stod(argv[++i]);
        } else if (arg == "--log-dir" && i + 1 < argc) {
            log_dir_override = argv[++i];
        } else if (arg == "--scene" && i + 1 < argc) {
            scene_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: mahru_ctrl [--headless] [--duration seconds] "
                         "[--log-stride n] [--stop-com-z z] [--stop-roll-deg deg] "
                         "[--stop-pitch-deg deg] [--max-wall-time seconds] "
                         "[--progress-interval seconds] [--log-dir path] [--scene xml]\n";
            return 0;
        }
    }

    if (sim) {
        std::cout << "Simulation mode" << std::endl;
        std::cout << "Using scene: " << scene_path << std::endl;
        SimulationBridge simulationBridge(scene_path, log_dir_override, headless);
        if (headless) {
            simulationBridge.RunHeadless(
                duration_s,
                log_stride,
                stop_com_z,
                stop_roll_deg,
                stop_pitch_deg,
                max_wall_time_s,
                progress_interval_s
            );
        } else {
            simulationBridge.run();
        }
    }
    else {
        std::cout << "Hardware mode" << std::endl;
    }

    return 0;
}
