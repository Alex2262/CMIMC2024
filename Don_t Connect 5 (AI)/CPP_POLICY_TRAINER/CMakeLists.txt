cmake_minimum_required(VERSION 3.21)
project(DC5)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "-Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto")

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

add_executable(DC5 main.cpp util.cpp util.h mcts.cpp mcts.h datagen.cpp datagen.cpp datagen.h policy_trainer.cpp policy_trainer.h)

target_link_libraries(DC5 Threads::Threads)