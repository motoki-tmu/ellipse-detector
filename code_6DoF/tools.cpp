#include "tools.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <math.h>
#include <time.h>
#include <dirent.h>

// 2つの直線の交点を計算
cv::Point2f lineCrossPoint(cv::Point2f l1p1, cv::Point2f l1p2, cv::Point2f l2p1, cv::Point2f l2p2 )
{
	cv::Point2f crossPoint;
	float k1, k2, b1, b2; // k：傾き、b：y切片

	// 0割り防止で直線が垂直な場合を先に処理
	if (l1p1.x == l1p2.x && l2p1.x == l2p2.x)
	{
		crossPoint = cv::Point2f(0, 0);
		return crossPoint;
	}
	if (l1p1.x == l1p2.x)
	{
		crossPoint.x = l1p1.x;
		k2 = (l2p2.y - l2p1.y) / (l2p2.x - l2p1.x);
		b2 = l2p1.y - k2 * l2p1.x;
		crossPoint.y = k2 * crossPoint.x + b2;
		return crossPoint;
	}
	if (l2p1.x == l2p2.x)
	{
		crossPoint.x = l2p1.x;
		k2 = (l1p2.y - l1p1.y) / (l1p2.x - l1p1.x);
		b2 = l1p1.y - k2 * l1p1.x;
		crossPoint.y = k2 * crossPoint.x + b2;
		return crossPoint;
	}

	k1 = (l1p2.y - l1p1.y) / (l1p2.x - l1p1.x);
	k2 = (l2p2.y - l2p1.y) / (l2p2.x - l2p1.x);
	b1 = l1p1.y - k1 * l1p1.x;
	b2 = l2p1.y - k2 * l2p1.x;
	if (k1 == k2)
	{
		crossPoint = cv::Point2f(0, 0);
	}
	else
	{
		crossPoint.x = (b2 - b1) / (k1 - k2);
		crossPoint.y = k1 * crossPoint.x + b1;
	}
	return crossPoint;
}

// 2つの2次元点を行列に変換
void point2Mat(cv::Point2f p1, cv::Point2f p2, float mat[2][2])
{
	mat[0][0] = p1.x;
	mat[0][1] = p1.y;
	mat[1][0] = p2.x;
	mat[1][1] = p2.y;
}

// 6点から楕円の不変量を計算
float value4SixPoints( V2SP ) // V2SPはヘッダファイルで定義されている
{
	float result = 1;
	cv::Mat A, B, C;
	float matB[2][2], matC[2][2];
	cv::Point2f v, w, u;
	v = lineCrossPoint(p1, p2, p3, p4);
	w = lineCrossPoint(p5, p6, p3, p4);
	u = lineCrossPoint(p5, p6, p1, p2);

	point2Mat(u, v, matB);
	point2Mat(p1, p2, matC);
	B = cv::Mat(2, 2, CV_32F, matB);
	C = cv::Mat(2, 2, CV_32F, matC);
	A = C * B.inv();

	result *= A.at<float>(0, 0) * A.at<float>(1, 0) / (A.at<float>(0, 1) * A.at<float>(1, 1));

	point2Mat(p3, p4, matC);
	point2Mat(v, w, matB);
	B = cv::Mat(2, 2, CV_32F, matB);
	C = cv::Mat(2, 2, CV_32F, matC);
	A = C * B.inv();
	result *= A.at<float>(0, 0) * A.at<float>(1, 0) / (A.at<float>(0, 1) * A.at<float>(1, 1));

	point2Mat(p5, p6, matC);
	point2Mat(w, u, matB);
	B = cv::Mat(2, 2, CV_32F, matB);
	C = cv::Mat(2, 2, CV_32F, matC);
	A = C * B.inv();
	result *= A.at<float>(0, 0) * A.at<float>(1, 0) / (A.at<float>(0, 1) * A.at<float>(1, 1));
	return result;
} 

// エッジ描画
void showEdge(VVP points_, cv::Mat& picture)
{
	srand( (unsigned)time( NULL ));
	int radius = 1;
	cv::Point center;

	int sEdge = points_.size();
	cv::Point prev_point;
	cv::Point current_point;
	for (int iEdge = 0; iEdge < sEdge; iEdge++)
	{
		// int r = rand() % 256;
		// int g = rand() % 256;
		// int b = rand() % 256;
		//cv::Scalar color = cv::Scalar(b, g, r);
		cv::Scalar color = cv::Scalar(255, 255, 255); // 白で固定
		VP Edge = points_.at(iEdge);
		int sPoints = Edge.size();
		for (int iPoint = 0; iPoint < sPoints - 1; iPoint++)
		{
			center = Edge.at(iPoint);
			prev_point = Edge.at(iPoint);
			current_point = Edge.at(iPoint + 1);
			cv::line(picture, prev_point, current_point, color, 1, cv::LINE_AA);
		}
	}
}
void showEdge_r(VVP points_, cv::Mat& picture)
{
	srand( (unsigned)time( NULL )); int radius = 1; cv::Point center;
	int sEdge = points_.size(); cv::Point prev_point; cv::Point current_point;
	for (int iEdge = 0; iEdge < sEdge; iEdge++)
	{
		cv::Scalar color = cv::Scalar(0, 0, 255); // 赤
		VP Edge = points_.at(iEdge);
		int sPoints = Edge.size();
		for (int iPoint = 0; iPoint < sPoints - 1; iPoint++)
		{
			center = Edge.at(iPoint); prev_point = Edge.at(iPoint); current_point = Edge.at(iPoint + 1);
			cv::line(picture, prev_point, current_point, color, 1, cv::LINE_AA);
		}
	}
}
void showEdge_b(VVP points_, cv::Mat& picture)
{
	srand( (unsigned)time( NULL )); int radius = 1; cv::Point center;
	int sEdge = points_.size(); cv::Point prev_point; cv::Point current_point;
	for (int iEdge = 0; iEdge < sEdge; iEdge++)
	{
		cv::Scalar color = cv::Scalar(255, 0, 0); // 青
		VP Edge = points_.at(iEdge);
		int sPoints = Edge.size();
		for (int iPoint = 0; iPoint < sPoints - 1; iPoint++)
		{
			center = Edge.at(iPoint); prev_point = Edge.at(iPoint); current_point = Edge.at(iPoint + 1);
			cv::line(picture, prev_point, current_point, color, 1, cv::LINE_AA);
		}
	}
}
void showEdge_y(VVP points_, cv::Mat& picture)
{
	srand( (unsigned)time( NULL )); int radius = 1; cv::Point center;
	int sEdge = points_.size(); cv::Point prev_point; cv::Point current_point;
	for (int iEdge = 0; iEdge < sEdge; iEdge++)
	{
		cv::Scalar color = cv::Scalar(0, 255, 255); // 黄色
		VP Edge = points_.at(iEdge);
		int sPoints = Edge.size();
		for (int iPoint = 0; iPoint < sPoints - 1; iPoint++)
		{
			center = Edge.at(iPoint); prev_point = Edge.at(iPoint); current_point = Edge.at(iPoint + 1);
			cv::line(picture, prev_point, current_point, color, 1, cv::LINE_AA);
		}
	}
}
void showEdge_g(VVP points_, cv::Mat& picture)
{
	srand( (unsigned)time( NULL )); int radius = 1; cv::Point center;
	int sEdge = points_.size(); cv::Point prev_point; cv::Point current_point;
	for (int iEdge = 0; iEdge < sEdge; iEdge++)
	{
		cv::Scalar color = cv::Scalar(0, 255, 0); // 緑
		VP Edge = points_.at(iEdge);
		int sPoints = Edge.size();
		for (int iPoint = 0; iPoint < sPoints - 1; iPoint++)
		{
			center = Edge.at(iPoint); prev_point = Edge.at(iPoint); current_point = Edge.at(iPoint + 1);
			cv::line(picture, prev_point, current_point, color, 1, cv::LINE_AA);
		}
	}
}

// ファイル操作
int writeFile(std::string fileName_cpp, std::vector<std::string> vsContent)
{
	std::string line = "";
	std::vector<std::string> data;
	std::vector<std::string> data_split;
	std::ofstream out(fileName_cpp);
	if (!out)
	{
		std::cout << "Failed to open file for writing" << std::endl;
		return -1;
	}
	for (std::vector<std::string>::iterator i = vsContent.begin(); i < vsContent.end(); i++)
	{
		out << *i << std::endl;
	}
	out.close();
	return 1;
}

// いろんな結果を保存
void SaveEllipses(const std::string& fileName, const std::vector<Ellipse>& ellipses, ResultData& resultdata)
{
	unsigned n = ellipses.size();
	std::vector<std::string> resultString;
	std::stringstream resultsitem;

	resultsitem << "Number of detections: " << n ;
	resultString.push_back(resultsitem.str());

	for (unsigned i = 0; i < n; ++i)
	{
		const Ellipse& e = ellipses[i];
		resultsitem.str("");
		resultsitem << std::fixed << std::setprecision(5) // 小数点以下5桁に固定
                	<< std::setw(10) << e._xc
                	<< std::setw(10) << e._yc
                	<< std::setw(10) << e._a
                	<< std::setw(10) << e._b
                	<< std::setw(10) << e._rad
                	<< std::setw(10) << e._score;
		resultString.push_back(resultsitem.str());
	}

	resultsitem.str(""); resultString.push_back("");
	resultsitem << "Number of matching craters: " << resultdata.matchingpoints;
	resultString.push_back(resultsitem.str()); resultsitem.str("");
	resultsitem << "                deltatheta: " << resultdata.deltatheta;
	resultString.push_back(resultsitem.str()); resultsitem.str("");
	resultsitem << "                     scale: " << resultdata.scale;
	resultString.push_back(resultsitem.str()); resultsitem.str("");

	resultString.push_back("");
	resultsitem << "LSM estimated position: " << std::setw(12) << resultdata.LSMposition[0] << std::setw(12) << resultdata.LSMposition[1];
	resultString.push_back(resultsitem.str()); resultsitem.str("");
	resultsitem << "PnP estimated position: " << std::setw(12) << resultdata.PnPposition[0] << std::setw(12) << resultdata.PnPposition[1] << std::setw(12) << resultdata.PnPposition[2];
	resultString.push_back(resultsitem.str()); resultsitem.str("");
	resultsitem << "PnP estimated attitude: " << std::setw(12) << resultdata.PnPattitude[0] << std::setw(12) << resultdata.PnPattitude[1] << std::setw(12) << resultdata.PnPattitude[2];
	resultString.push_back(resultsitem.str()); resultsitem.str("");

	resultString.push_back("");
	resultsitem << "Crater Detection  Time: " << std::setw(10) << resultdata.times[0];
	resultString.push_back(resultsitem.str()); resultsitem.str("");
	resultsitem << "Crater Matching   Time: " << std::setw(10) << resultdata.times[1];
	resultString.push_back(resultsitem.str()); resultsitem.str("");
	resultsitem << "Pose   Estimation Time: " << std::setw(10) << resultdata.times[2];
	resultString.push_back(resultsitem.str()); resultsitem.str("");
	resultsitem << "       Total      Time: " << std::setw(10) << resultdata.times[3];
	resultString.push_back(resultsitem.str()); resultsitem.str("");

	writeFile(fileName, resultString);

	/*
	for (int i = 0; i < resultString.size(); i++)
	{
		std::cout << resultString[i] << std::endl;
	}
	*/
}