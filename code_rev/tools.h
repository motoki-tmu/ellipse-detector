// 幾何計算、画像表示、ファイル操作等

#pragma once
#include "common.h"
#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

#define V2SP cv::Point2f p3, cv::Point2f p2, cv::Point2f p1, cv::Point2f p4, cv::Point2f p5, cv::Point2f p6

// .value4SixPoints
cv::Point2f lineCrossPoint(cv::Point2f l1p1, cv::Point2f l1p2, cv::Point2f l2p1, cv::Point2f l2p2);

// .value4SixPoints
void point2Mat(cv::Point2f p1, cv::Point2f p2, float mat[2][2]);

// CNED.Detect.Triplets
float value4SixPoints(V2SP);

// CNED.Detect
void showEdge(VVP points_, cv::Mat& picture);
void showEdge_r(VVP points_, cv::Mat& picture);
void showEdge_b(VVP points_, cv::Mat& picture);
void showEdge_y(VVP points_, cv::Mat& picture);
void showEdge_g(VVP points_, cv::Mat& picture);

// .SaveEllipses
int writeFile(std::string fileName_cpp, std::vector<std::string> vsContent);

// Main
void SaveEllipses(const std::string& fileName, const std::vector<Ellipse>& ellipses);

// Main
void Trim(std::string &str);