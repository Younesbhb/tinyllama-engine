CXX=clang++
CXXFLAGS=-std=c++20 -O2 -g -Wall -Wextra -pthread

SRCS=src/main.cpp src/gguf_model.cpp src/forward.cpp src/generate.cpp \
     src/tokenizer.cpp src/ops.cpp

engine: $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@

test: src/test_ops.cpp src/ops.cpp
	$(CXX) $(CXXFLAGS) src/test_ops.cpp src/ops.cpp -o test_ops
	./test_ops

.PHONY: clean test
clean:
	rm -f engine test_ops
	rm -rf engine.dSYM test_ops.dSYM