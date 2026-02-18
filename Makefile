CXX=clang++
CXXFLAGS=-std=c++20 -O0 -g -Wall -Wextra

engine: src/main.cpp src/tokenizer.cpp src/ops.cpp
	$(CXX) $(CXXFLAGS) src/main.cpp src/tokenizer.cpp src/ops.cpp -o $@

test_ops: src/test_ops.cpp src/ops.cpp
	$(CXX) $(CXXFLAGS) src/test_ops.cpp src/ops.cpp -o $@

.PHONY: clean
clean:
	rm -f engine test_ops
	rm -rf engine.dSYM