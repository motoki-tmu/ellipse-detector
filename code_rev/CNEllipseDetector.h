// CNEllipseDetector

#pragma once
#include "common.h"
#include "tools.h"
#include <opencv2/core.hpp>
#include <vector>

#include "LSM.h"

#ifndef GLOBAL // includeガード
#define GLOBAL
	extern bool myselect1; // トリプレットに関するスイッチ
	extern bool myselect2;
	extern bool myselect3;
	extern float tCNC;  // 円弧が同一の楕円のものかの判定
	extern float tTCNl; // 直線を除外するための閾値
#endif

// 制約（cppの方で#ifndefで記述 → ここで#defineされている場合cppの方は無効化）
#define DISCARD_TCN
//#define DISCARD_CONSTRAINT_OBOX
//#define DISCARD_CONSTRAINT_CONVEXITY
//#define DISCARD_CONSTRAINT_POSITION
//#define DISCARD_CONSTRAINT_CNC//our new idea
//#define DISCARD_CONSTRAINT_CENTER

// EllipseData構造体の定義
struct EllipseData 
{
	bool isValid;	  		// 計算結果の判定
	float ta;		  		// 中心線の傾き
	float tb;		  		// （二つの平行弦の中点から傾きが iNs/2 個得られる、それらの中央値）
	float ra;		  		// 参照線の傾き
	float rb;		  		// （円弧の端点と中点を結ぶ線）
	cv::Point2f Ma;	  		// 中央中心点
	cv::Point2f Mb;	  		// （iNs個の参照線の中点の中央値）
	cv::Point2f Cab;  		// 推定中心
	std::vector<float> Sa; 	// 参照線に平行な弦の中点二つを通る線分の傾きの集合
	std::vector<float> Sb; 
};

// 楕円検出アルゴリズム
class CNEllipseDetector
{
	/* パラメータ */

	cv::Size _szPreProcessingGaussKernelSize; // ガウシアンフィルタのカーネルサイズ
	double   _dPreProcessingGaussSigma;       // ガウシアンフィルタのsigma

	int		 _iMinEdgeLength;	   			  // エッジの最小必要長さ（短い円弧を除外）
	float	 _fMinOrientedRectSide;           // 円弧を囲む矩形の最小辺長（直線を除外）
	float	 _fMaxRectAxesRatio;              // 矩形の長辺と短辺の最大比率（直線を除外）

	float    _fThPosition;                    // 2つの円弧の位置関係の閾値（大きく離れたペアを除外）

	unsigned _uNs;                            // 探索する弦の数（unsignedは符号なし整数）

	float	 _fMaxCenterDistance;             // 2つの中心点誤差の閾値
	float	 _fMaxCenterDistance2;            // 上記の2乗値

	float	 _fDistanceToEllipseContour;      // ある点が楕円の輪郭上にあるとみなす閾値

	float	 _fMinScore;				  	  // 楕円の判定スコア
	float	 _fMinReliability;				  // 楕円の信頼度

	float 	 _scoreAlpha;					  // スコア評価の重みα


	/* 補助変数 */

	cv::Size _szImg;		// 入力画像サイズ

	double _myExecTime;		// 実行時間
	std::vector<double> _timesHelper;
	std::vector<double> _times;	// _times is a std::vector containing the execution time of each step.
							// _times[0] : time for edge detection
							// _times[1] : time for pre processing
							// _times[2] : time for grouping
							// _times[3] : time for estimation
							// _times[4] : time for validation
							// _times[5] : time for clustering

	int ACC_N_SIZE;			// size of accumulator N = B/A
	int ACC_R_SIZE;			// size of accumulator R = rho = atan(K)
	int ACC_A_SIZE;			// size of accumulator A

	// std::vector<int> accN; の方が安全？ → cppで accN = new int[ACC_N_SIZE]; を accN.assign(ACC_N_SIZE, 0); に変更
	int* accN;				// pointer to accumulator N
	int* accR;				// pointer to accumulator R
	int* accA;				// pointer to accumulator A


	/* 公開メンバ関数 */

public:
	float countsOfFindEllipse;   // カウント用変数
	float countsOfGetFastCenter; // カウント用変数

	CNEllipseDetector(void);     // コンストラクタ、インスタンス生成時に呼ばれる
	~CNEllipseDetector(void);	 // デストラクタ、インスタンス破棄時に呼ばれる

	// Main
	void SetParameters (cv::Size szPreProcessingGaussKernelSize,
						double	 dPreProcessingGaussSigma,
						float 	 fThPosition,
						float	 fMaxCenterDistance,
						int		 iMinEdgeLength,
						float	 fMinOrientedRectSide,
						float	 fDistanceToEllipseContour,
						float	 fMinScore,
						float	 fMinReliability,
						int      iNs,
						float	 scoreAlpha
						);

	// Main
	void Detect(cv::Mat1b& gray, std::vector<Ellipse>& ellipses);

	// Main
	std::vector<double> GetTimes() { return _times; }

	// Main
	void DrawDetectedEllipses(cv::Mat3b& output, std::vector<Ellipse>& ellipses, int iTopN=0, int thickness=2);



	/* 非公開メンバ関数 */

private:

	static const ushort PAIR_12 = 0x00;
	static const ushort PAIR_23 = 0x01;
	static const ushort PAIR_34 = 0x02;
	static const ushort PAIR_14 = 0x03;
	static const ushort PAIR_13 = 0x04;
	static const ushort PAIR_24 = 0x05;

	void Tic(unsigned idx) //start
	{
		_timesHelper[idx] = 0.0;
		_times[idx] = (double)cv::getTickCount();
	};

	void Tac(unsigned idx) //restart
	{
		_timesHelper[idx] = _times[idx];
		_times[idx] = (double)cv::getTickCount();
	};

	void Toc(unsigned idx) //stop
	{
		_times[idx] = ((double)cv::getTickCount() - _times[idx])*1000. / cv::getTickFrequency();
		_times[idx] += _timesHelper[idx];
	};

	// .Detect
	void PreProcessing(cv::Mat1b& I, cv::Mat1b& DP, cv::Mat1b& DN);

	// .Detect
	void DetectEdges13(cv::Mat1b& DP, VVP& points_1, VVP& points_3);
	void DetectEdges24(cv::Mat1b& DN, VVP& points_2, VVP& points_4);

	// .Detect.Triplets
	inline uint GenerateKey(uchar pair, ushort u, ushort v);

	// .Detect.Triplets.GetFastCenter
	float GetMedianSlope(std::vector<cv::Point2f>& med, cv::Point2f& M, std::vector<float>& slopes);

	// .Detect.Triplets
	void  GetFastCenter	(VP& e1, VP& e2, EllipseData& data);

	// .Detect.Triplets
	cv::Point2f GetCenterCoordinates(EllipseData& data_ij, EllipseData& data_ik);

	// .Detect.Triplets.FindEllipse
	int FindMaxK(const int* v) const;
	int FindMaxN(const int* v) const;
	int FindMaxA(const int* v) const;

	// .Detect.Triplets
	void FindEllipses(cv::Point2f& center, VP& edge_i, VP& edge_j, VP& edge_k, EllipseData& data_ij, EllipseData& data_ik, std::vector<Ellipse>& ellipses);

	// .Detect
	void Triplets124(VVP& pi, VVP& pj, VVP& pk, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses);
	void Triplets231(VVP& pi, VVP& pj, VVP& pk, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses);
	void Triplets342(VVP& pi, VVP& pj, VVP& pk, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses);
	void Triplets413(VVP& pi, VVP& pj, VVP& pk, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses);

	// .Detect.Doublets
	void FindEllipses2(VP& edge_ij, std::vector<Ellipse>& ellipses);
	void FindEllipses3(cv::Point2f& center, VP& edge_i, VP& edge_j, EllipseData& data_ij, std::vector<Ellipse>& ellipses);

	// .Detect
	void Doublets12(VVP& pi, VVP& pj, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses);
	void Doublets23(VVP& pi, VVP& pj, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses);
	void Doublets34(VVP& pi, VVP& pj, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses);
	void Doublets41(VVP& pi, VVP& pj, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses);
	void Doublets13(VVP& pi, VVP& pj, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses);
	void Doublets24(VVP& pi, VVP& pj, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses);

	// .Detect
	void DeleteEllipses(std::vector<Ellipse>& ellipses);

	// .Detect
	void OverlapEllipses(const cv::Size& sz, std::vector<Ellipse>& ellipses);

	// .Detect(unused)
	void ClusterEllipses(std::vector<Ellipse>& ellipses);
};