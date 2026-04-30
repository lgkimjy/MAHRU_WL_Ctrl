#ifndef SIMULATION_INTERFACE_HPP
#define SIMULATION_INTERFACE_HPP

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>

#include <mujoco/mujoco.h>
#include "glfw_adapter.h"
#include "simulate.h"
#include "array_safety.h"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

extern "C"
{
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <unistd.h>
#endif
}

namespace mj = ::mujoco;
namespace mju = ::mujoco::sample_util;

// constants
const double syncMisalign = 0.1;       // maximum mis-alignment before re-sync (simulation seconds)
const double simRefreshFraction = 0.7; // fraction of refresh available for simulation
const int kErrorLength = 1024;         // load error string length

using Seconds = std::chrono::duration<double>;

class SimulationInterface
{
private:
protected:
    // Initialize System
    virtual void Initialize() = 0;
    // Update Sensor Data 
    virtual void UpdateSystemObserver() = 0;
    // Write Control Command to Sim
    virtual void UpdateControlCommand() = 0;
    // Update System's Visualization Data
    virtual void UpdateSystemVisualInfo() = 0;
    // Update User Input
    virtual void UpdateUserInput() = 0;
    // double _ctrl_time = 0.0;
public:
    explicit SimulationInterface(const std::string& scene_file, bool use_gui = true);
    ~SimulationInterface();

    std::string filename_storage_;
    const char* filename_;

    mjvCamera cam_;
    mjvOption opt_;
    mjvPerturb pert_;

    // model and data
    // std::unique_ptr<mj::Simulate> mjSim_ = nullptr;
    mj::Simulate *mjSim_ = nullptr;
    mjModel *mjModel_ = nullptr;
    mjData *mjData_ = nullptr;

    void run();
    void PhysicsThread(mj::Simulate *sim, const char *filename);
    void PhysicsLoop(mj::Simulate &sim);

    const char *Diverged(int disableflags, const mjData *d)
    {
        if (disableflags & mjDSBL_AUTORESET) {
            for (mjtWarning w : {mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS}) {
                if (d->warning[w].number > 0) {
                    return mju_warningText(w, d->warning[w].lastinfo);
                }
            }
        }
        return nullptr;
    }

    mjModel *LoadModel(const char *file, mj::Simulate &sim)
    {
        // this copy is needed so that the mju::strlen call below compiles
        char filename[mj::Simulate::kMaxFilenameLength];
        mju::strcpy_arr(filename, file);

        // make sure filename is not empty
        if (!filename[0]) {
            return nullptr;
        }

        // load and compile
        char loadError[kErrorLength] = "";
        mjModel *mnew = 0;
        auto load_start = mj::Simulate::Clock::now();
        if (mju::strlen_arr(filename) > 4 &&
            !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                          mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4)) {
            mnew = mj_loadModel(filename, nullptr);
            if (!mnew) {
                mju::strcpy_arr(loadError, "could not load binary model");
            }
        } else {
            mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);

            // remove trailing newline character from loadError
            if (loadError[0]) {
                int error_length = mju::strlen_arr(loadError);
                if (loadError[error_length - 1] == '\n') {
                    loadError[error_length - 1] = '\0';
                }
            }
        }
        auto load_interval = mj::Simulate::Clock::now() - load_start;
        double load_seconds = Seconds(load_interval).count();

        if (!mnew) {
            std::printf("%s\n", loadError);
            mju::strcpy_arr(sim.load_error, loadError);
            return nullptr;
        }

        // compiler warning: print and pause
        if (loadError[0]) {
            // mj_forward() below will print the warning message
            std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
            sim.run = 0;
        }

        // if no error and load took more than 1/4 seconds, report load time
        else if (load_seconds > 0.25) {
            mju::sprintf_arr(loadError, "Model loaded in %.2g seconds", load_seconds);
        }

        mju::strcpy_arr(sim.load_error, loadError);

        return mnew;
    }

    //---------------------------------------- plugin handling -----------------------------------------

    // return the path to the directory containing the current executable
    // used to determine the location of auto-loaded plugin libraries
    std::string getExecutableDir()
    {
#if defined(_WIN32) || defined(__CYGWIN__)
        constexpr char kPathSep = '\\';
        std::string realpath = [&]() -> std::string {
            std::unique_ptr<char[]> realpath(nullptr);
            DWORD buf_size = 128;
            bool success = false;
            while (!success) {
                realpath.reset(new (std::nothrow) char[buf_size]);
                if (!realpath) {
                    std::cerr << "cannot allocate memory to store executable path\n";
                    return "";
                }

                DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
                if (written < buf_size) {
                    success = true;
                } else if (written == buf_size) {
                    // realpath is too small, grow and retry
                    buf_size *= 2;
                } else {
                    std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
                    return "";
                }
            }
            return realpath.get();
        }();
#else
        constexpr char kPathSep = '/';
#if defined(__APPLE__)
        std::unique_ptr<char[]> buf(nullptr);
        {
            std::uint32_t buf_size = 0;
            _NSGetExecutablePath(nullptr, &buf_size);
            buf.reset(new char[buf_size]);
            if (!buf) {
                std::cerr << "cannot allocate memory to store executable path\n";
                return "";
            }
            if (_NSGetExecutablePath(buf.get(), &buf_size)) {
                std::cerr << "unexpected error from _NSGetExecutablePath\n";
            }
        }
        const char *path = buf.get();
#else
        const char *path = "/proc/self/exe";
#endif
        std::string realpath = [&]() -> std::string {
            std::unique_ptr<char[]> realpath(nullptr);
            std::uint32_t buf_size = 128;
            bool success = false;
            while (!success) {
                realpath.reset(new (std::nothrow) char[buf_size]);
                if (!realpath) {
                    std::cerr << "cannot allocate memory to store executable path\n";
                    return "";
                }

                std::size_t written = readlink(path, realpath.get(), buf_size);
                if (written < buf_size) {
                    realpath.get()[written] = '\0';
                    success = true;
                } else if (written == -1) {
                    if (errno == EINVAL)
                    {
                        // path is already not a symlink, just use it
                        return path;
                    }

                    std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
                    return "";
                } else {
                    // realpath is too small, grow and retry
                    buf_size *= 2;
                }
            }
            return realpath.get();
        }();
#endif

        if (realpath.empty()) {
            return "";
        }

        for (std::size_t i = realpath.size() - 1; i > 0; --i) {
            if (realpath.c_str()[i] == kPathSep) {
                return realpath.substr(0, i);
            }
        }

        // don't scan through the entire file system's root
        return "";
    }

    // scan for libraries in the plugin directory to load additional plugins
    void scanPluginLibraries()
    {
        // check and print plugins that are linked directly into the executable
        int nplugin = mjp_pluginCount();
        if (nplugin) {
            std::printf("Built-in plugins:\n");
            for (int i = 0; i < nplugin; ++i) {
                std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
            }
        }

// define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
        const std::string sep = "\\";
#else
        const std::string sep = "/";
#endif

        // try to open the ${EXECDIR}/MUJOCO_PLUGIN_DIR directory
        // ${EXECDIR} is the directory containing the simulate binary itself
        // MUJOCO_PLUGIN_DIR is the MUJOCO_PLUGIN_DIR preprocessor macro
        const std::string executable_dir = getExecutableDir();
        if (executable_dir.empty()) {
            return;
        }

        const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
        mj_loadAllPluginLibraries(
            plugin_dir.c_str(), +[](const char *filename, int first, int count) {
                std::printf("Plugins registered by library '%s':\n", filename);
                for (int i = first; i < first + count; ++i) {
                    std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
                } 
            }
        );
    }
};

#endif // SIMULATION_INTERFACE_HPP
