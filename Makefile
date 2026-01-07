CXX=clang++
CXXFLAGS=-std=c++20 -O0 -g -Wall -Wextra

engine: src/main.cpp src/tokenizer.cpp
	$(CXX) $(CXXFLAGS) src/main.cpp src/tokenizer.cpp -o $@

.PHONY: clean
clean:
	rm -f engine
	rm -rf engine.dSYM
