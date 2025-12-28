// pch.h
#ifndef PCH_H
#define PCH_H

// Standard library headers (must come before CUDA headers for Thrust compatibility)
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#include <limits.h>
#include <climits>
#include <limits>
#include <cstddef>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <optional>
#include "cudatimer.h"

// Standard library headers
#include <iostream>
#include <cmath>
#include <math.h>
#include <vector>
#include <string>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include <time.h>

enum DistanceType {
    L2_DISTANCE = 0,
    COSINE_DISTANCE = 1
};

#endif //PCH_H