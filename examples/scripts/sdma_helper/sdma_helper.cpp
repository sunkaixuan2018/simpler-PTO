/**
 * SDMA workspace helper — thin C API around SdmaWorkspaceManager.
 * Python loads this .so and calls the C API for SDMA workspace init/finalize.
 *
 * Build: from examples/scripts/sdma_helper with CANN env set:
 *   mkdir build && cd build && cmake .. && make
 */

#include <cstdint>
#include <iostream>

#include "pto/comm/async/sdma/sdma_workspace_manager.hpp"

static pto::comm::sdma::SdmaWorkspaceManager g_sdmaMgr;

extern "C" {

int sdma_helper_init(void** out_workspace) {
    if (out_workspace == nullptr)
        return -1;
    if (!g_sdmaMgr.Init()) {
        std::cerr << "[sdma_helper] SdmaWorkspaceManager::Init() failed" << std::endl;
        return -1;
    }
    *out_workspace = g_sdmaMgr.GetWorkspaceAddr();
    return 0;
}

void sdma_helper_finalize(void) {
    g_sdmaMgr.Finalize();
}

}  // extern "C"
