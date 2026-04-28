#include "SimulationInterface.hpp"

SimulationInterface::SimulationInterface(const std::string& scene_file):
  filename_storage_(scene_file),
  filename_(filename_storage_.c_str())
{
    // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
    if (rosetta_error_msg) {
        DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
        std::exit(1);
    }
#endif

    // print version, check compatibility
    std::printf("[ MuJoCo version ] %s\n", mj_versionString());
    if (mjVERSION_HEADER!=mj_version()) {
        mju_error("Headers and library have different versions");
    }

    // Initialize MuJoCo
    scanPluginLibraries();
    mjv_defaultCamera(&cam_);
    mjv_defaultOption(&opt_);
    mjv_defaultPerturb(&pert_);

    auto platform_ui = std::make_unique<mj::GlfwAdapter>();

    // uncomment this line, if wanted to change the spawn window size
    // if (GLFWwindow* window = glfwGetCurrentContext()) {
    //     glfwSetWindowSize(window, 4000, 2000);  // window size
    // }
    
    mjSim_ = new mj::Simulate(
        std::move(platform_ui),
        &cam_, &opt_, &pert_, /* is_passive = */ false
    );
}

SimulationInterface::~SimulationInterface() 
{
    // delete everything we allocated
    mj_deleteData(mjData_);
    mj_deleteModel(mjModel_);
}

void SimulationInterface::run() 
{
    std::thread physicsthreadhandle(
        &SimulationInterface::PhysicsThread, this, mjSim_, filename_
    );
    mjSim_->RenderLoop();
    // @todo: check if i can add custom visualization option here, what happen if i stop the simulation?
    physicsthreadhandle.join();
}

void SimulationInterface::PhysicsThread(mj::Simulate *sim, const char* filename) 
{
    // request loadmodel if file given (otherwise drag-and-drop)
    if (filename != nullptr) {
        mjSim_->LoadMessage(filename);
        mjModel_ = LoadModel(filename, *sim);
        if (mjModel_) {
            // lock the sim mutex
            const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

            mjData_ = mj_makeData(mjModel_);
        }
        if (mjData_) {
            sim->Load(mjModel_, mjData_, filename);

            // lock the sim mutex
            const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

            mj_forward(mjModel_, mjData_);

        } else {
            sim->LoadMessageClear();
        }
    }
    PhysicsLoop(*mjSim_);
}

// simulate in background thread (while rendering in main thread)
void SimulationInterface::PhysicsLoop(mj::Simulate& sim) {
  // cpu-sim syncronization point
  std::chrono::time_point<mj::Simulate::Clock> syncCPU;
  mjtNum syncSim = 0;

  // run until asked to exit
  while (!sim.exitrequest.load()) {
    if (sim.droploadrequest.load()) {
      sim.LoadMessage(sim.dropfilename);
      mjModel* mnew = LoadModel(sim.dropfilename, sim);
      sim.droploadrequest.store(false);

      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.dropfilename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(mjData_);
        mj_deleteModel(mjModel_);

        mjModel_ = mnew;
        mjData_ = dnew;
        mj_forward(mjModel_, mjData_);

      } else {
        sim.LoadMessageClear();
      }
    }

    if (sim.uiloadrequest.load()) {
      sim.uiloadrequest.fetch_sub(1);
      sim.LoadMessage(sim.filename);
      mjModel* mnew = LoadModel(sim.filename, sim);
      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.filename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(mjData_);
        mj_deleteModel(mjModel_);

        mjModel_ = mnew;
        mjData_ = dnew;
        mj_forward(mjModel_, mjData_);

      } else {
        sim.LoadMessageClear();
      }
    }

    // sleep for 1 ms or yield, to let main thread run
    //  yield results in busy wait - which has better timing but kills battery life
    if (sim.run && sim.busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

      // run only if model is present
      if (mjModel_) {
        // running
        if (sim.run) {
          bool stepped = false;

          // record cpu time at start of iteration
          const auto startCPU = mj::Simulate::Clock::now();

          // elapsed CPU and simulation time since last sync
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = mjData_->time - syncSim;

          // requested slow-down factor
          double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

          // misalignment condition: distance from target sim time is bigger than syncmisalign
          bool misaligned =
              std::abs(Seconds(elapsedCPU).count()/slowdown - elapsedSim) > syncMisalign;

          // out-of-sync (for any reason): reset sync times, step
          if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
              misaligned || sim.speed_changed) {
            // re-sync
            syncCPU = startCPU;
            syncSim = mjData_->time;
            sim.speed_changed = false;

            //  when backspace or spacebar is pressed, it comes here for restart from beginning or from where it was paused
            //  update feedback, controller, and command to robot
            //  @todo: call system->runCtrl()
            //  printf("%.4f\n", mjData_->time);
            if(mjData_->time == 0.0){
              std::cout << "--------------------------------" << std::endl;
              std::cout << "[ SimulationInterface ][ PhysicsLoop ] Start Simulation" << std::endl;
              std::cout << "--------------------------------" << std::endl;
              Initialize();
            }
            UpdateSystemObserver();
            UpdateUserInput();
            UpdateControlCommand();
            UpdateSystemVisualInfo();

            // run single step, let next iteration deal with timing
            mj_step(mjModel_, mjData_);
            const char* message = Diverged(mjModel_->opt.disableflags, mjData_);
            if (message) {
              sim.run = 0;
              mju::strcpy_arr(sim.load_error, message);
            } else {
              stepped = true;
            }
          }

          // in-sync: step until ahead of cpu
          else {
            bool measured = false;
            mjtNum prevSim = mjData_->time;

            double refreshTime = simRefreshFraction/sim.refresh_rate;

            // step while sim lags behind cpu and within refreshTime
            while (Seconds((mjData_->time - syncSim)*slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                   mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime)) {
              // measure slowdown before first step
              if (!measured && elapsedSim) {
                sim.measured_slowdown =
                    std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                measured = true;
              }

              // inject noise
              sim.InjectNoise();

              //  update feedback, controller, and command to robot
              //  @todo: call system->runCtrl()
              UpdateSystemObserver();
              UpdateUserInput();
              UpdateControlCommand();
              UpdateSystemVisualInfo();

              // call mj_step
              mj_step(mjModel_, mjData_);
              const char* message = Diverged(mjModel_->opt.disableflags, mjData_);
              if (message) {
                sim.run = 0;
                mju::strcpy_arr(sim.load_error, message);
              } else {
                stepped = true;
              }

              // break if reset
              if (mjData_->time < prevSim) {
                break;
              }
            }
          }

          // save current state to history buffer
          if (stepped) {
            sim.AddToHistory();
          }
        }

        // paused
        else {
          // run mj_forward, to update rendering and joint sliders
          mj_forward(mjModel_, mjData_);
          sim.speed_changed = true;
        }
      }
    }  // release std::lock_guard<std::mutex>
  }
}
