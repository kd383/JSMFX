include ../make.inc

SRCS = $(wildcard *.cpp)

EXES = ${SRCS:.cpp=}

.PHONY: all
all:	$(EXES)

%: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LIBS) -o $@ $<

.PHONY: clean

clean:
	rm -f $(EXES) *.csv
