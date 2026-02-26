#pragma once

#include <string>

class GGUFModel;
struct RunState;

void generate(GGUFModel& model, RunState& state,
              const std::string& prompt,
              int max_tokens = 2048,
              float temperature = 0.7f,
              float top_p = 0.9f);