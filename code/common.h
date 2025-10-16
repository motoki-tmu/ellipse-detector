// 画像処理

#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;
//using namespace cv;

typedef vector<cv::Point> VP;
typedef vector<VP> VVP;
typedef unsigned int uint;

#define _INFINITY 1024 //無限大の値、これじゃ小さい？

// 符号関数を自作、valが正の値なら(0.f < val)がTrue → 1となる
int inline sgn(float val)
{
    return (0.f < val) - (val < 0.f);
}

// 浮動小数点数が無限大かどうかを判定
bool inline isInf(float x)
{
	union // 異なる型を同じメモリ領域に格納
	{
		float f;
		int	  i;
	} u; // unionの変数uを定義

	u.f = x;
	u.i &= 0x7fffffff;
	return !(u.i ^ 0x7f800000);
}

// 2点間の傾きを計算
float inline Slope(float x1, float y1, float x2, float y2)
{
		float den = float(x2 - x1);
		float num = float(y2 - y1);
		if(den != 0)
		{
			return (num / den);
		}
		else
		{
			return ((num > 0) ? float(_INFINITY) : float(-_INFINITY)); // 分母が0のときは無限大を返す
		}
};


/* 元の自作canny関数、openCVのcanny関数を模倣しつつ拡張している（2:Sobelフィルタのxとy方向の勾配画像出力、3:low,high_threshをヒストグラムから自動で決定）
void cvCanny2(	const void* srcarr, void* dstarr,
				double low_thresh, double high_thresh,
			   	void* dxarr, void* dyarr,
                int aperture_size );

void cvCanny3(	const void* srcarr, void* dstarr,
				void* dxarr, void* dyarr,
                int aperture_size );

void Canny2(	cv::InputArray image, cv::OutputArray _edges,
				cv::OutputArray _sobel_x, cv::OutputArray _sobel_y,
            	double threshold1, double threshold2,
            	int apertureSize, bool L2gradient );

void Canny3(	cv::InputArray image, cv::OutputArray _edges,
            	cv::OutputArray _sobel_x, cv::OutputArray _sobel_y,
            	int apertureSize, bool L2gradient );
*/

// 以下にcanny関数を改良
void Canny_v2(	cv::InputArray image, cv::OutputArray edges,
                cv::OutputArray sobel_x, cv::OutputArray sobel_y,
                double threshold1, double threshold2,
                int apertureSize = 3, bool L2gradient = false);

void Canny_v3(	cv::InputArray image, cv::OutputArray edges,
                cv::OutputArray sobel_x, cv::OutputArray sobel_y,
                int apertureSize = 3, bool L2gradient = false);

// 2点間の距離の2乗を計算
float inline ed2(const cv::Point& A, const cv::Point& B) // Pointはint型
{
	return float(((B.x - A.x)*(B.x - A.x) + (B.y - A.y)*(B.y - A.y)));
}

float inline ed2f(const cv::Point2f& A, const cv::Point2f& B) // Point2fはfloat型
{
	return (B.x - A.x)*(B.x - A.x) + (B.y - A.y)*(B.y - A.y);
}

// セグメント化、矩形計算（不要？cv::connectedComponentsWithStats関数の方がいいかも）
void Labeling(cv::Mat1b& image, vector<vector<cv::Point> >& segments, int iMinLength);
void LabelingRect(cv::Mat1b& image, VVP& segments, int iMinLength, vector<cv::Rect>& bboxes);

// 細線化（不要？cv::ximgproc::thinning関数の方がいいかも → opencv/contribのビルドが必要？）
void Thinning(cv::Mat1b& imgMask, uchar byF=255, uchar byB=0);

// 点のソート
bool SortBottomLeft2TopRight(const cv::Point& lhs, const cv::Point& rhs);
bool SortTopLeft2BottomRight(const cv::Point& lhs, const cv::Point& rhs);
bool SortBottomLeft2TopRight2f(const cv::Point2f& lhs, const cv::Point2f& rhs);

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

	// 楕円の描画（指定色）
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

// 角度差の最小値を取得
float GetMinAnglePI(float alpha, float beta);