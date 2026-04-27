#include "MJSimulationBridge.hpp"

#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    bool sim = true;

    if (sim) {
        std::cout << "Simulation mode" << std::endl;
        std::cout << "Using scene: " << robot_name::scene << std::endl;
        SimulationBridge simulationBridge(std::string(CMAKE_SOURCE_DIR) + "/" + std::string(robot_name::scene));
        simulationBridge.run();
    }
    else {
        std::cout << "Hardware mode" << std::endl;
    }

    return 0;
}
