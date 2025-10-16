// 幾何計算、画像表示、ファイル操作、楕円検出の評価等

#pragma once
#include "common.h" // Ellipse型を使うために参照
#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

using namespace std;

#define V2SP cv::Point2f p3, cv::Point2f p2, cv::Point2f p1, cv::Point2f p4, cv::Point2f p5, cv::Point2f p6

// 複数の画像を一つの画面に表示
void MultiImage_OneWin(const string& MultiShow_WinName, const vector<cv::Mat>& SrcImg_V, cv::Size SubPlot, cv::Size ImgMax_Size);

// 2つの直線の交点を計算（cv::Point2f：float型のx,y座標を持つ2次元点）
cv::Point2f lineCrossPoint(cv::Point2f l1p1, cv::Point2f l1p2, cv::Point2f l2p1, cv::Point2f l2p2);

// 2つの2次元点を行列に変換（自作関数、OpenCVには無い）
void point2Mat(cv::Point2f p1, cv::Point2f p2, float mat[2][2]);

// 6点から不変量を計算
float value4SixPoints(V2SP);

// 画像のリサイズ（実際は拡大）
void PyrDown(string picName);

// 画像のリサイズ（scaleで倍率指定）
cv::Mat matResize(cv::Mat src, double scale);

// 検出した楕円のパラメータを保存
void SaveEllipses(const string& fileName, const vector<Ellipse>& ellipses);
void SaveEllipses(const string& workingDir, const string& imgName, const vector<Ellipse>& ellipses /*, const vector<double>& times*/); // こっちはディレクトリも指定できるっぽい

// Ground Truthの読み込み
void LoadGT(vector<Ellipse>& gt, const string& sGtFileName, bool bIsAngleInRadians = true);

// 楕円の重なりを計算（Jaccard係数、閾値はth）
bool TestOverlap(const cv::Mat1b& gt, const cv::Mat1b& test, float th);

int Count(const vector<bool> v);

// 評価スコアの計算
float Evaluate(const vector<Ellipse>& ellGT, const vector<Ellipse>& ellTest, const float th_score, const cv::Mat3b& img);

// ノイズ付加
void salt(cv::Mat& image, int n);

// エッジ描画
void showEdge(vector<vector<cv::Point>> points_, cv::Mat& picture);

// ファイルの書き込み、読み込み、トリム
int writeFile(string fileName_cpp, vector<string> vsContent);
int readFile(string fileName_cpp);
int readFileByChar(string fileName_split);
void Trim(string &str);

vector<string> getStr(string str);

// ディレクトリ内のファイル一覧を取得
void listDir(string real_dir, vector<string>& files, bool r = false);