#include "common.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// 改良したCanny関数（Sobelフィルタのxとy方向の勾配画像出力）
void Canny_v2(	cv::InputArray image, cv::OutputArray edges,
            	cv::OutputArray sobel_x, cv::OutputArray sobel_y,
                double threshold1, double threshold2,
                int apertureSize, bool L2gradient)
{
    cv::Mat src = image.getMat();

    // SobelフィルタでX方向とY方向の勾配を計算、出力画像の型(depth)にCV_16Sを指定するのが一般的
    cv::Sobel(src, sobel_x, CV_16S, 1, 0, apertureSize);
    cv::Sobel(src, sobel_y, CV_16S, 0, 1, apertureSize);

    cv::Canny(src, edges, threshold1, threshold2, apertureSize, L2gradient);
}

// 改良したCanny関数（low,high_threshをヒストグラムから自動で決定）
void Canny_v3(	cv::InputArray image, cv::OutputArray edges,
            	cv::OutputArray sobel_x, cv::OutputArray sobel_y,
                int apertureSize, bool L2gradient)
{
    cv::Mat src = image.getMat();

    // Sobelフィルタで勾配を計算（apertureSizeはSobelフィルタのカーネルサイズ、3）
    cv::Mat dx, dy;
    cv::Sobel(src, dx, CV_16S, 1, 0, apertureSize);
    cv::Sobel(src, dy, CV_16S, 0, 1, apertureSize);

    // 出力用にコピー
    dx.copyTo(sobel_x);
    dy.copyTo(sobel_y);

    // 勾配強度を計算（L1ノルム: |dx| + |dy|）
    cv::Mat mag;
    cv::Mat abs_dx, abs_dy;
    cv::convertScaleAbs(dx, abs_dx);
    cv::convertScaleAbs(dy, abs_dy);
    cv::addWeighted(abs_dx, 1, abs_dy, 1, 0, mag);

    // ヒストグラムを計算して閾値を決定
    // 元のコードの魔法の数字（マジックナンバー）をそのまま使用
    const double percent_of_pixels_not_edges = 0.9;
    const double threshold_ratio = 0.3;

    // ヒストグラムの計算
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat hist;
    cv::calcHist(&mag, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    // 累積ヒストグラムから上限閾値(high_thresh)を決定
    double total = src.rows * src.cols * percent_of_pixels_not_edges;
    double current_sum = 0;
    int high_thresh = 0;
    for (int i = 0; i < histSize; i++)
    {
        current_sum += hist.at<float>(i);
        if (current_sum >= total)
        {
            high_thresh = i;
            break;
        }
    }

    // 下限閾値(low_thresh)を決定
    int low_thresh = static_cast<int>(high_thresh * threshold_ratio);

	std::cout << high_thresh << std::endl << low_thresh << std::endl;

    // 計算した閾値を使って標準Canny関数を呼び出す
    cv::Canny(src, edges, low_thresh, high_thresh, apertureSize, L2gradient);

	// 勾配の大きさがhigh以上 → エッジ、high未満low以上 → エッジと繋がっていればエッジ、low未満 → エッジではない
	// 勾配の大きさの計算方法 → L2gradientがTrueならユークリッド、Falseならマンハッタン（前者の方が正確で低速）
}

// セグメント化（中身はcv::connectedComponentsWithStats関数を用いた自作関数に変更した）
// segments（参照渡し）に独立したエッジを構成するピクセルの座標が格納される
void Labeling(cv::Mat1b& image, VVP& segments, int iMinLength)
{
    cv::Mat labels, stats, centroids; // bboxの座標や重心も計算してくれるが使わない？
    int num_labels = cv::connectedComponentsWithStats(image, labels, stats, centroids); // num_labelsはラベル（独立したエッジ）の総数

    VVP temp_segments(num_labels - 1); // 背景ラベル0を除いた総数の配列を用意

    for (int y = 0; y < labels.rows; ++y)
	{
        for (int x = 0; x < labels.cols; ++x)
		{
            int label = labels.at<int>(y, x);
            if (label > 0)
			{
                temp_segments[label - 1].push_back(cv::Point(x, y));
            }
        }
    }

    segments.clear();
    for (int i = 1; i < num_labels; ++i)
	{
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= iMinLength) // iMinLengthよりも小さいピクセル数で構成されるエッジは削除（main.cppでThLength = 16として定義、名前変わってる）
		{
            segments.push_back(temp_segments[i - 1]);
        }
    }
}

bool SortBottomLeft2TopRight(const cv::Point& lhs, const cv::Point& rhs)
{
	if(lhs.x == rhs.x)
	{
		return lhs.y > rhs.y;
	}
	return lhs.x < rhs.x;
};

bool SortTopLeft2BottomRight(const cv::Point& lhs, const cv::Point& rhs)
{
	if(lhs.x == rhs.x)
	{
		return lhs.y < rhs.y;
	}
	return lhs.x < rhs.x;
};

float GetMinAnglePI(float alpha, float beta)
{
	float pi = float(CV_PI);
	float pi2 = float(2.0 * CV_PI);

	//normalize data in [0, 2*pi]
	float a = fmod(alpha + pi2, pi2);
	float b = fmod(beta + pi2, pi2);

	//normalize data in [0, pi]
	if (a > pi)
		a -= pi;
	if (b > pi)
		b -= pi;

	if (a > b)
	{
		std::swap(a, b);
	}

	float diff1 = b - a;
	float diff2 = pi - diff1;
	return std::min(diff1, diff2);
}
