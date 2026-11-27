// Based on Project Maya (https://github.com/mw00/project-maya)
// Copyright (c) 2026 Peasant Smith, OscarSheng, alesha-pro, cobra91 and the Project Maya contributors
// Licensed under the MIT License.

// src/ik2/ik2_main.cpp - strata-ik2: the Inkling 2 engine path on its own (the same entry point `strata` switches
// to for an inkling2 model).
#include "strata/ik2/engine.hpp"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    // Strata's `components add-build` runs `<exe> --version` to detect the build version;
    // it needs a \d+\.\d+\.\d+ substring, which `strata-inkling 0.1.64` still has. When this
    // binary is installed as the Inkling-2-Mini shape engine (component:strata-inkling),
    // Control's shape_engine._test_start() requires the first line to start with `strata-inkling`.
    // Must be before any GPU/model work.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0 || std::strcmp(argv[i], "-V") == 0) {
            std::printf("strata-inkling 0.1.64\n");
            return 0;
        }
    }
    return strata::ik2::run(argc, argv);
}
