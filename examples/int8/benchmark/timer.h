/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once
#include <chrono>

namespace timers {
class TimerBase {
 public:
  virtual void start() {}
  virtual void stop() {}
  float microseconds() const noexcept {
    return mMs * 1000.f;
  }
  float milliseconds() const noexcept {
    return mMs;
  }
  float seconds() const noexcept {
    return mMs / 1000.f;
  }
  void reset() noexcept {
    mMs = 0.f;
  }

 protected:
  float mMs{0.0f};
};

template <typename Clock>
class CPUTimer : public TimerBase {
 public:
  using clock_type = Clock;

  void start() {
    mStart = Clock::now();
  }
  void stop() {
    mStop = Clock::now();
    mMs += std::chrono::duration<float, std::milli>{mStop - mStart}.count();
  }

 private:
  std::chrono::time_point<Clock> mStart, mStop;
}; // class CPUTimer

using PreciseCPUTimer = CPUTimer<std::chrono::high_resolution_clock>;
} // namespace timers
