# LLMJob Earn's RandomX CPU worker (see README.md).
#
# Built against a checkout of tevador/RandomX at a pinned commit, which
# scripts/build-xmr-worker.sh fetches and passes in as RANDOMX_SOURCE_DIR.
cmake_minimum_required(VERSION 3.16)

# Static MSVC runtime everywhere, RandomX included: the worker then needs no
# Visual C++ redistributable beside it. RandomX's own cmake_minimum_required
# predates this policy, so the default is what reaches its directory.
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
cmake_policy(SET CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

project(llmjob_xmr_worker C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(RANDOMX_SOURCE_DIR "" CACHE PATH "A tevador/RandomX checkout")
set(RANDOMX_COMMIT "unknown" CACHE STRING "The RandomX commit RANDOMX_SOURCE_DIR holds")
set(XMR_WORKER_VERSION "1.0.0" CACHE STRING "The worker's version")
if(NOT RANDOMX_SOURCE_DIR OR NOT EXISTS "${RANDOMX_SOURCE_DIR}/src/randomx.h")
  message(FATAL_ERROR "Set RANDOMX_SOURCE_DIR to a tevador/RandomX checkout (scripts/build-xmr-worker.sh does this).")
endif()

# EXCLUDE_FROM_ALL: build the library only, not RandomX's tests and benchmark.
add_subdirectory("${RANDOMX_SOURCE_DIR}" randomx EXCLUDE_FROM_ALL)

find_package(Threads REQUIRED)

add_executable(llmjob-xmr-worker src/xmr_worker.cpp)
set_property(TARGET llmjob-xmr-worker PROPERTY CXX_STANDARD 17)
set_property(TARGET llmjob-xmr-worker PROPERTY CXX_STANDARD_REQUIRED ON)
target_include_directories(llmjob-xmr-worker PRIVATE "${RANDOMX_SOURCE_DIR}/src")
target_compile_definitions(llmjob-xmr-worker PRIVATE
  XMR_WORKER_VERSION="${XMR_WORKER_VERSION}" RANDOMX_COMMIT="${RANDOMX_COMMIT}")
target_link_libraries(llmjob-xmr-worker PRIVATE randomx Threads::Threads)

# Linux: fully static, so one binary runs on any x86-64 distribution, HiveOS's
# included, whatever its glibc.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  option(XMR_WORKER_STATIC "Link the Linux worker statically" ON)
  if(XMR_WORKER_STATIC)
    target_link_options(llmjob-xmr-worker PRIVATE -static -s)
  endif()
endif()
