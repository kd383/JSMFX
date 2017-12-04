include ../make.inc

SRCS = ex1.cpp ex2.cpp

DEPS = ${SRCS:.cpp=.d}
EXES = ${SRCS:.cpp=}

.PHONY: all
all:	ex1 ex2

ex1: ex1.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LIBS) -o $@ $<

ex2: ex2.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LIBS) -o $@ $<


.PHONY: clean

clean:
	rm -f $(EXES)
