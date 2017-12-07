# C++ compiler and flags
CXX=/usr/local/Cellar/gcc/7.2.0/bin/g++-7
CXXFLAGS=-std=c++14 -O2 -fopenmp

INCLUDE = -I/Users/kd383/Documents/class/cs5220/final/include -I/Users/kd383/Documents/armadillo-8.200.2/include
LIBS = -L/Users/kd383/Documents/armadillo-8.200.2 -larmadillo -framework Accelerate
