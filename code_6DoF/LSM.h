#pragma once
#include "common.h"
#include <vector> 

const int NUM_CRATERS = 119;
const int NUM_COMBINATIONS = NUM_CRATERS * (NUM_CRATERS - 1) / 2;

struct point{double x; double y;};
struct DataSet{std::vector<point> model; std::vector<std::vector<double>> line_segment;};
struct MatchPairSet{point world_pt; point model_pt;};

void CraterMatching(std::vector<Ellipse>& ellipses, std::vector<point> model, std::vector<std::vector<double>> line_segment, std::vector<MatchPairSet>& MatchPair);