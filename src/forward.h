#pragma once


/*
The forward() declaration uses GGUFModel& and RunState& as parameter types. 
The compiler needs to know these names exist — but since they're references (not actual objects), 
it doesn't need to know their size or contents.
A forward declaration just says "this name exists, trust me, you'll get the details later.
*/
class GGUFModel;      // forward declaration (no need to #include)
struct RunState;       // forward declaration




void forward(GGUFModel& model, RunState& state, int token, int pos);
