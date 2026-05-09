#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using VP = std::vector<cv::Point>;
using VVP = std::vector<VP>;
typedef unsigned int uint;

// CNED.Detect.Triplets.GetFastCenter
inline int sgn(float val)
{
    return (0.f < val) - (val < 0.f);
}

// CNED.Detect.Triplets
inline float ed2(const cv::Point& A, const cv::Point& B)
{
	return float(((B.x - A.x)*(B.x - A.x) + (B.y - A.y)*(B.y - A.y)));
}

// CNED.Detect.PreProcessing
void Canny_v2(	cv::InputArray image, cv::OutputArray edges,
                cv::OutputArray sobel_x, cv::OutputArray sobel_y,
                double threshold1, double threshold2,
                int apertureSize = 3, bool L2gradient = false);

void Canny_v3(	cv::InputArray image, cv::OutputArray edges,
                cv::OutputArray sobel_x, cv::OutputArray sobel_y,
                int apertureSize = 3, bool L2gradient = false);


// CNED.Detect.DetectEdge
void Labeling(cv::Mat1b& image, std::vector<std::vector<cv::Point> >& segments, int iMinLength);

// CNED.Detect.DetectEdge
bool SortTopLeft2BottomRight(const cv::Point& lhs, const cv::Point& rhs);
bool SortBottomLeft2TopRight(const cv::Point& lhs, const cv::Point& rhs);

// CNED.Detect.ClusterEllipses(unused)
float GetMinAnglePI(float alpha, float beta);


// Ellipse構造体の定義
struct Ellipse
{
	float _xc; 	  // 中心x座標
	float _yc; 	  // 中心y座標
	float _a; 	  // 長半径
	float _b; 	  // 短半径
	float _rad;   // 回転角
	float _score; // スコア

	Ellipse() : _xc(0.f), _yc(0.f), _a(0.f), _b(0.f), _rad(0.f), _score(0.f) {};
	Ellipse(float xc, float yc, float a, float b, float rad, float score = 0.f) : _xc(xc), _yc(yc), _a(a), _b(b), _rad(rad), _score(score) {};
	Ellipse(const Ellipse& other) : _xc(other._xc), _yc(other._yc), _a(other._a), _b(other._b), _rad(other._rad), _score(other._score) {};

	// 楕円の描画（指定色）※cvRoundはそのままでok
	void Draw(const cv::Mat& img, const cv::Scalar& color, const int thickness)
	{
		cv::ellipse(img, cv::Point(cvRound(_xc),cvRound(_yc)), cv::Size(cvRound(_a),cvRound(_b)), _rad * 180.0 / CV_PI, 0.0, 360.0, color, thickness);
	};

	// 楕円の描画（スコアに応じた色）
	void Draw(const cv::Mat3b& img, const int thickness)
	{
		cv::Scalar color(0, cvFloor(255.f * _score), 0);
		cv::ellipse(img, cv::Point(cvRound(_xc),cvRound(_yc)), cv::Size(cvRound(_a),cvRound(_b)), _rad * 180.0 / CV_PI, 0.0, 360.0, color, thickness);
	};

	// 楕円の比較（スコア順、スコアが同じなら扁平率の大きい方を優先）
	bool operator<(const Ellipse& other) const
	{
		if(_score == other._score)
		{
			float lhs_e = _b / _a;
			float rhs_e = other._b / other._a;
			if(lhs_e == rhs_e)
			{
				return false;
			}
			return lhs_e > rhs_e;
		}
		return _score > other._score;
	};
};
