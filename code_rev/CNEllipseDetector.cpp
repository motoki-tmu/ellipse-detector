// CNEllipseDetector

#include "CNEllipseDetector.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <ctime>


bool myselect1 = true;
bool myselect2 = false;
bool myselect3 = false;
float tCNC = 0.3f;   // 円弧が同一の楕円のものかの判定
float tTCNl = 0.05f; // 直線を除外するための閾値

CNEllipseDetector::CNEllipseDetector(void) : _times(6, 0.0), _timesHelper(6, 0.0) // 6個の要素をもつ配列を0.0で初期化
{
	// Default Parameters Settings
	_szPreProcessingGaussKernelSize = cv::Size(5, 5); // ガウシアンフィルタのカーネルサイズ
	_dPreProcessingGaussSigma = 1.0;                  // ガウシアンフィルタのsigma
	_fThPosition = 1.0f;                              // 2つの円弧の位置関係の閾値（大きく離れたペアを除外）
	_fMaxCenterDistance = 100.0f * 0.05f;             // 2つの中心点誤差の閾値
	_fMaxCenterDistance2 = _fMaxCenterDistance * _fMaxCenterDistance;
	_iMinEdgeLength = 16;                            // エッジの最小必要長さ（短い円弧を除外）
	_fMinOrientedRectSide = 3.0f;                     // 円弧を囲む矩形の最小辺長（直線を除外）
	_fDistanceToEllipseContour = 0.1f;                // ある点が楕円の輪郭上にあるとみなす閾値
	_fMinScore =0.4f;                                 // 楕円の判定スコア
	_fMinReliability =0.5;                            // 楕円の信頼度
	_uNs = 16;                                        // 探索する弦の数

	srand(unsigned(time(NULL)));
}


CNEllipseDetector::~CNEllipseDetector(void)
{
}

void CNEllipseDetector::SetParameters(cv::Size	szPreProcessingGaussKernelSize,
										 double	dPreProcessingGaussSigma,
										 float 	fThPosition,
										 float	fMaxCenterDistance,
										 int		iMinEdgeLength,
										 float	fMinOrientedRectSide,
										 float	fDistanceToEllipseContour,
										 float	fMinScore,
										 float	fMinReliability,
										 int     iNs,
										 float  scoreAlpha
										 )
{
	_szPreProcessingGaussKernelSize = szPreProcessingGaussKernelSize;
	_dPreProcessingGaussSigma = dPreProcessingGaussSigma;
	_fThPosition = fThPosition;
	_fMaxCenterDistance = fMaxCenterDistance;
	_iMinEdgeLength = iMinEdgeLength;
	_fMinOrientedRectSide = fMinOrientedRectSide;
	_fDistanceToEllipseContour = fDistanceToEllipseContour;
	_fMinScore = fMinScore;
	_fMinReliability = fMinReliability;
	_uNs = iNs;

	_fMaxCenterDistance2 = _fMaxCenterDistance * _fMaxCenterDistance;

	_scoreAlpha = scoreAlpha;
}

// 画像の平滑化、エッジ検出、勾配計算
void CNEllipseDetector::PreProcessing(cv::Mat1b& I, cv::Mat1b& DP, cv::Mat1b& DN) // 入力：I（グレースケール画像）、出力：DP（tanθ>0）、DN（tanθ<0）（それぞれサイズは入力と同じ、傾きに応じたエッジのみ描かれた画像に別れる）
{
	Tic(0); //edge detection

	GaussianBlur(I, I, _szPreProcessingGaussKernelSize, _dPreProcessingGaussSigma); // ガウシアンフィルタ

	cv::Mat1b E;				//edge mask
	cv::Mat1s DX, DY;			//sobel derivatives

	// エッジ検出
	Canny_v2(I, E, DX, DY, 60, 20, 3, false); // I:入力画像、E:エッジ画像、DX,DY:勾配
	// Canny_v3(I, E, DX, DY, 3, false);

	cv::imwrite("edges/canny.jpg", E);

	Toc(0); //edge detection

	Tac(1); //preprocessing

	// For each edge points, compute the edge direction
	for (int i = 0; i < _szImg.height; ++i)
	{
		short* _dx = DX.ptr<short>(i); // short:16bitの整数値、int:32bit、long:64bit
		short* _dy = DY.ptr<short>(i);
		uchar* _e = E.ptr<uchar>(i);   // uchar:unsigned char、符号なし文字型
		uchar* _dp = DP.ptr<uchar>(i);
		uchar* _dn = DN.ptr<uchar>(i);

		for (int j = 0; j<_szImg.width; ++j)
		{
			if (!((_e[j] <= 0) || (_dx[j] == 0) || (_dy[j] == 0))) // エッジとして検出され、勾配が0でないことをチェック
			{
				float phi = -(float(_dx[j]) / float(_dy[j]));

				if (phi > 0)	_dp[j] = (uchar)255;
				else if (phi < 0)	_dn[j] = (uchar)255;
			}
		}
	}
	Toc(1); //preprocessing
};

#define	DISCARD_TCN1

// 第1・3象限のエッジ分類（tanθ > 0 のエッジを凸の方向で分離）
// 3種類の方法でエッジの曲率判定を行っている（直線とみなした数をcountedgesで加算しているが特に使っていない）
// 直線判定ならcontinue、円弧判定なら最後に凸判定で分類
void CNEllipseDetector::DetectEdges13(cv::Mat1b& DP, VVP& points_1, VVP& points_3)
{
	VVP contours; // VVP = std::vector<VP>（common.hで定義）
	int countedges=0;

	Labeling(DP, contours, _iMinEdgeLength);  // contoursに独立したエッジを構成するピクセルの座標がラベル別に格納される
	int iContoursSize = int(contours.size());

	for (int i = 0; i < iContoursSize; ++i)
	{
		VP& edgeSegment = contours[i];

#ifndef DISCARD_CONSTRAINT_OBOX // エッジの曲率判定（直線を除外）
		cv::RotatedRect oriented = cv::minAreaRect(edgeSegment); // エッジを囲む最小の長方形を計算（傾きを考慮）
		float o_min = std::min(oriented.size.width, oriented.size.height);

		if (o_min < _fMinOrientedRectSide)
		{
			countedges++;
			continue;
		}
#endif
		
		sort(edgeSegment.begin(), edgeSegment.end(), SortTopLeft2BottomRight);
		int iEdgeSegmentSize = unsigned(edgeSegment.size());

		cv::Point& left = edgeSegment[0]; // 始点
		cv::Point& right = edgeSegment[iEdgeSegmentSize - 1]; // 終点

#ifndef DISCARD_TCN
#ifndef DISCARD_TCN2 // エッジの曲率判定（直線除外）厳しめ
		int flag = 0;
		for(int j = 0; j < iEdgeSegmentSize; j++) // 全ての点の曲率を計算
		{
			cv::Point& mid = edgeSegment[j];
			float data[] = {left.x, left.y, 1, mid.x, mid.y, 1, right.x, right.y, 1};
			cv::Mat threePoints(cv::Size(3, 3), CV_32FC1, data);
			double ans = determinant(threePoints);

			float dx = 1.0f * (left.x - right.x);
			float dy = 1.0f * (left.y - right.y);
			float edgelength2 = dx * dx + dy * dy;
			double TCNl = ans / (2 * sqrt(edgelength2));
			if (abs(TCNl) > tTCNl) // 1点でも曲率の閾値を超えれば終了
			{
				flag = 1;
				break;
			}
		}
		if(0 == flag)
		{
			countedges++;
			continue;
		}
#endif
#ifndef DISCARD_TCN1 // エッジの曲率判定（直線除外）易しめ
		cv::Point& mid = edgeSegment[iEdgeSegmentSize / 2]; // エッジの中点取得
		float data[] = {left.x, left.y, 1, mid.x, mid.y, 1, right.x, right.y, 1};
		cv::Mat threePoints(Size(3, 3), CV_32FC1, data);
		double ans = determinant(threePoints); // 始点・中点・終点からなる三角形の面積を計算

		float dx = 1.0f * (left.x - right.x);
		float dy = 1.0f * (left.y - right.y);
		float edgelength2 = dx * dx + dy * dy;
		double TCNl = ans / edgelength2;

		if (abs(TCNl) < tTCNl) // tTCN = 0.05
		{
			countedges++;
			continue;
		}
#endif
#endif

		// 凸判定
		int iCountTop = 0;
		int xx = left.x;
		for (int k = 1; k < iEdgeSegmentSize; ++k)
		{
			if (edgeSegment[k].x == xx) continue;

			iCountTop += (edgeSegment[k].y - left.y);
			xx = edgeSegment[k].x;
		}

		int width = abs(right.x - left.x) + 1;
		int height = abs(right.y - left.y) + 1;
		int iCountBottom = (width * height) - iEdgeSegmentSize - iCountTop;

		if (iCountBottom > iCountTop)
		{
			points_1.push_back(edgeSegment);
		}
		else if (iCountBottom < iCountTop)
		{
			points_3.push_back(edgeSegment);
		}
	}
};

// 第2・4象限のエッジ分類（tanθ < 0 のエッジを凸の方向で分離）
void CNEllipseDetector::DetectEdges24(cv::Mat1b& DN, VVP& points_2, VVP& points_4 )
{
	VVP contours;
	int countedges=0;

	Labeling(DN, contours, _iMinEdgeLength);

	int iContoursSize = unsigned(contours.size());

	for (int i = 0; i < iContoursSize; ++i)
	{
		VP& edgeSegment = contours[i];

#ifndef DISCARD_CONSTRAINT_OBOX
		cv::RotatedRect oriented = cv::minAreaRect(edgeSegment);
		float o_min = std::min(oriented.size.width, oriented.size.height);

		if (o_min < _fMinOrientedRectSide)
		{
			countedges++;
			continue;
		}
#endif
		
		sort(edgeSegment.begin(), edgeSegment.end(), SortBottomLeft2TopRight);
		int iEdgeSegmentSize = unsigned(edgeSegment.size());

		cv::Point& left = edgeSegment[0];
		cv::Point& right = edgeSegment[iEdgeSegmentSize - 1];

#ifndef DISCARD_TCN
#ifndef DISCARD_TCN2
		int flag = 0;
		for(int j = 0; j < iEdgeSegmentSize; j++){
			cv::Point& mid = edgeSegment[j];
			float data[] = {left.x, left.y, 1, mid.x, mid.y, 1, right.x, right.y, 1};
			cv::Mat threePoints(cv::Size(3, 3), CV_32FC1, data);
			double ans = determinant(threePoints);

			float dx = 1.0f * (left.x - right.x);
			float dy = 1.0f * (left.y - right.y);
			float edgelength2 = dx * dx + dy * dy;
			double TCNl = ans / (2 * sqrt(edgelength2));
			if (abs(TCNl) > tTCNl)
			{
				flag = 1;
				break;
			}
		}
		if (0 == flag)
		{
			countedges++;
			continue;
		}
#endif
#ifndef DISCARD_TCN1
		Point& mid = edgeSegment[iEdgeSegmentSize / 2];
		float data[] = {left.x, left.y, 1, mid.x, mid.y, 1, right.x, right.y, 1};
		Mat threePoints(Size(3, 3), CV_32FC1, data);
		double ans = determinant(threePoints);

		float dx = 1.0f * (left.x - right.x);
		float dy = 1.0f * (left.y - right.y);
		float edgelength2 = dx * dx + dy * dy;
		double TCNl = ans / edgelength2;

		if (abs(TCNl) < tTCNl) 
		{
			countedges++;
			continue;
		}
#endif
#endif

		int iCountBottom = 0;
		int xx = left.x;
		for (int k = 1; k < iEdgeSegmentSize; ++k)
		{
			if (edgeSegment[k].x == xx) continue;

			iCountBottom += (left.y - edgeSegment[k].y);
			xx = edgeSegment[k].x;
		}

		int width = abs(right.x - left.x) + 1;
		int height = abs(right.y - left.y) + 1;
		int iCountTop = (width * height) - iEdgeSegmentSize - iCountBottom;

		if (iCountBottom > iCountTop)
		{
			points_2.push_back(edgeSegment);
		}
		else if (iCountBottom < iCountTop)
		{
			points_4.push_back(edgeSegment);
		}
	}
};

// 円弧のペアとインデックスからキー生成（キーを6つ定義するため、3bitに拡張）
uint inline CNEllipseDetector::GenerateKey(uchar pair, ushort u, ushort v)
{
	return (pair << 29) + (u << 15) + v; // 0~14にv, 15~28にu, 29~31にpairを配置
};

// 楕円の中心を推定（中心候補を入力、中心候補の重心と傾きを出力）Paper.Fig 6-b
float CNEllipseDetector::GetMedianSlope(std::vector<cv::Point2f>& med, cv::Point2f& M, std::vector<float>& slopes)
{
	unsigned iNofPoints = med.size();     // 中心候補点の数を計算
	//cv::Assert(iNofPoints >= 2);        // 点が2個未満ならプログラム停止

	unsigned halfSize = iNofPoints >> 1;  // = 右に1ビットシフト
	unsigned quarterSize = halfSize >> 1; // → 2進数だと半分の値になる

	std::vector<float> xx, yy;
	slopes.reserve(halfSize);             // reserveで必要なメモリ領域を確保
	xx.reserve(iNofPoints);
	yy.reserve(iNofPoints);

	for (unsigned i = 0; i < halfSize; ++i)
	{
		cv::Point2f& p1 = med[i];
		cv::Point2f& p2 = med[halfSize + i]; // iNofPoints = 16なら8から割り当て → med[i]とmed[8+i]の2点の傾きを計算、計8つの傾きを取得

		xx.push_back(p1.x);
		xx.push_back(p2.x);
		yy.push_back(p1.y);
		yy.push_back(p2.y);

		float den = (p2.x - p1.x);
		float num = (p2.y - p1.y);

		if (den == 0) den = 0.00001f;

		slopes.push_back(num / den);
	}

	// 指定したn番目の要素がn番目に来るようにソートする関数
	std::nth_element(slopes.begin(), slopes.begin() + quarterSize, slopes.end());
	std::nth_element(xx.begin(), xx.begin() + halfSize, xx.end());
	std::nth_element(yy.begin(), yy.begin() + halfSize, yy.end());
	M.x = xx[halfSize]; // 中点の中央値
	M.y = yy[halfSize];

	return slopes[quarterSize]; // 傾きの中央値
};

// 楕円の中心を推定（e1, e2はペア円弧）Paper.Fig 6-b
void CNEllipseDetector::GetFastCenter(VP& e1, VP& e2, EllipseData& data)
{
	countsOfGetFastCenter++;
	data.isValid = true;

	unsigned size_1 = unsigned(e1.size());
	unsigned size_2 = unsigned(e2.size());

	unsigned hsize_1 = size_1 >> 1; // ÷2
	unsigned hsize_2 = size_2 >> 1;

	// 円弧上の複数の点ベクトル（ソート済み）の中央番目の座標を円弧の中点としている（幾何学的に厳密な中点ではない）
	cv::Point& med1 = e1[hsize_1];
	cv::Point& med2 = e2[hsize_2];

	cv::Point2f M12, M34;
	float q2, q4;

	{// e1の始点とe2の中点
		float dx_ref = float(e1[0].x - med2.x);
		float dy_ref = float(e1[0].y - med2.y);

		if (dy_ref == 0) dy_ref = 0.00001f;

		float m_ref = dy_ref / dx_ref; // 基準の傾き
		data.ra = m_ref;

		std::vector<cv::Point2f> med; // 平行な弦の中点を格納するベクトル
		med.reserve(hsize_2);

		unsigned minPoints = (_uNs < hsize_2) ? _uNs : hsize_2; // 最大_uNs個の点を探索

		std::vector<uint> indexes(minPoints);
		if (_uNs < hsize_2) // サンプリング処理
		{
			unsigned iSzBin = hsize_2 / unsigned(_uNs);
			unsigned iIdx = hsize_2 + (iSzBin / 2);

			for (unsigned i = 0; i < _uNs; ++i)
			{
				indexes[i] = iIdx;
				iIdx += iSzBin;
			}
		}
		else
		{
			iota(indexes.begin(), indexes.end(), hsize_2);
		}
		for (uint ii = 0; ii < minPoints; ++ii) // 平行な弦の探索
		{
			uint i = indexes[ii];

			float x1 = float(e2[i].x);
			float y1 = float(e2[i].y);

			uint begin = 0;
			uint end = size_1 - 1;

			float xb = float(e1[begin].x);
			float yb = float(e1[begin].y);
			float res_begin = ((xb - x1) * dy_ref) - ((yb - y1) * dx_ref);
			int sign_begin = sgn(res_begin);
			if (sign_begin == 0) // 傾きが等しければmedに追加して次のループへ
			{
				med.push_back(cv::Point2f((xb + x1)* 0.5f, (yb + y1)* 0.5f));
				continue;
			}

			float xe = float(e1[end].x);
			float ye = float(e1[end].y);
			float res_end = ((xe - x1) * dy_ref) - ((ye - y1) * dx_ref);
			int sign_end = sgn(res_end);
			if (sign_end == 0) // 傾きが等しければmedに追加して次のループへ
			{
				med.push_back(cv::Point2f((xe + x1)* 0.5f, (ye + y1)* 0.5f));
				continue;
			}

			if ((sign_begin + sign_end) != 0)
			{
				continue;
			}

			uint j = (begin + end) >> 1;
			while (end - begin > 2)
			{
				float x2 = float(e1[j].x);
				float y2 = float(e1[j].y);
				float res = ((x2 - x1) * dy_ref) - ((y2 - y1) * dx_ref);
				int sign_res = sgn(res);

				if (sign_res == 0)
				{
					med.push_back(cv::Point2f((x2 + x1)* 0.5f, (y2 + y1)* 0.5f));
					break;
				}

				if (sign_res + sign_begin == 0)
				{
					sign_end = sign_res;
					end = j;
				}
				else
				{
					sign_begin = sign_res;
					begin = j;
				}
				j = (begin + end) >> 1;
			}

			med.push_back(cv::Point2f((e1[j].x + x1)* 0.5f, (e1[j].y + y1)* 0.5f));
		}

		if (med.size() < 2) // 平行な弦が2本未満なら無効
		{
			data.isValid = false;
			return;
		}

		q2 = GetMedianSlope(med, M12, data.Sa); // q2に傾き、M12に中央中心点、data.Saに全ての傾きリストが保存される（重心とリストは参照渡し）
	}

	{// e2の始点とe1の中点
		float dx_ref = float(med1.x - e2[0].x);
		float dy_ref = float(med1.y - e2[0].y);

		if (dy_ref == 0) dy_ref = 0.00001f;

		float m_ref = dy_ref / dx_ref;
		data.rb = m_ref;

		std::vector<cv::Point2f> med;
		med.reserve(hsize_1);

		uint minPoints = (_uNs < hsize_1) ? _uNs : hsize_1;

		std::vector<uint> indexes(minPoints);
		if (_uNs < hsize_1)
		{
			unsigned iSzBin = hsize_1 / unsigned(_uNs);
			unsigned iIdx = hsize_1 + (iSzBin / 2);

			for (unsigned i = 0; i < _uNs; ++i)
			{
				indexes[i] = iIdx;
				iIdx += iSzBin;
			}
		}
		else
		{
			iota(indexes.begin(), indexes.end(), hsize_1);
		}

		for (uint ii = 0; ii < minPoints; ++ii)
		{
			uint i = indexes[ii];

			float x1 = float(e1[i].x);
			float y1 = float(e1[i].y);

			uint begin = 0;
			uint end = size_2 - 1;

			float xb = float(e2[begin].x);
			float yb = float(e2[begin].y);
			float res_begin = ((xb - x1) * dy_ref) - ((yb - y1) * dx_ref);
			int sign_begin = sgn(res_begin);
			if (sign_begin == 0)
			{
				med.push_back(cv::Point2f((xb + x1)* 0.5f, (yb + y1)* 0.5f));
				continue;
			}

			float xe = float(e2[end].x);
			float ye = float(e2[end].y);
			float res_end = ((xe - x1) * dy_ref) - ((ye - y1) * dx_ref);
			int sign_end = sgn(res_end);
			if (sign_end == 0)
			{
				med.push_back(cv::Point2f((xe + x1)* 0.5f, (ye + y1)* 0.5f));
				continue;
			}

			if ((sign_begin + sign_end) != 0)
			{
				continue;
			}

			uint j = (begin + end) >> 1;

			while (end - begin > 2)
			{
				float x2 = float(e2[j].x);
				float y2 = float(e2[j].y);
				float res = ((x2 - x1) * dy_ref) - ((y2 - y1) * dx_ref);
				int sign_res = sgn(res);

				if (sign_res == 0)
				{
					med.push_back(cv::Point2f((x2 + x1)* 0.5f, (y2 + y1)* 0.5f));
					break;
				}

				if (sign_res + sign_begin == 0)
				{
					sign_end = sign_res;
					end = j;
				}
				else
				{
					sign_begin = sign_res;
					begin = j;
				}
				j = (begin + end) >> 1;
			}

			med.push_back(cv::Point2f((e2[j].x + x1)* 0.5f, (e2[j].y + y1)* 0.5f));
		}

		if (med.size() < 2)
		{
			data.isValid = false;
			return;
		}
		q4 = GetMedianSlope(med, M34, data.Sb);
	}

	if (q2 == q4)
	{
		data.isValid = false;
		return;
	}

	// 中心座標Cabを計算、Paper.Eq(11), Eq(12)
	float invDen = 1 / (q2 - q4);
	data.Cab.x = (M34.y - q4 * M34.x - M12.y + q2 * M12.x) * invDen;
	data.Cab.y = (q2 * M34.y - q4 * M12.y + q2 * q4 * (M12.x - M34.x)) * invDen;
	data.ta = q2;
	data.tb = q4;
	data.Ma = M12;
	data.Mb = M34;
};

// 中央座標の決定 Paper.Fig 8
cv::Point2f CNEllipseDetector::GetCenterCoordinates(EllipseData& data_ij, EllipseData& data_ik)
{
	float xx[7];
	float yy[7];

	xx[0] = data_ij.Cab.x; // ijペアの推定中心x座標
	xx[1] = data_ik.Cab.x; // ikペアの推定中心x座標
	yy[0] = data_ij.Cab.y; 
	yy[1] = data_ik.Cab.y;

	{
		//1-1（ijペアの中心線1とikペアの中心線1の交点を求める）（中心線1は添え字a）
		float q2 = data_ij.ta; // t：中心線の傾き
		float q4 = data_ik.ta;
		cv::Point2f& M12 = data_ij.Ma; // M：エッジの中点座標
		cv::Point2f& M34 = data_ik.Ma;

		float invDen = 1 / (q2 - q4);
		xx[2] = (M34.y - q4 * M34.x - M12.y + q2 * M12.x) * invDen;
		yy[2] = (q2 * M34.y - q4 * M12.y + q2 * q4 * (M12.x - M34.x)) * invDen;
	}

	{
		//1-2
		float q2 = data_ij.ta;
		float q4 = data_ik.tb;
		cv::Point2f& M12 = data_ij.Ma;
		cv::Point2f& M34 = data_ik.Mb;

		float invDen = 1 / (q2 - q4);
		xx[3] = (M34.y - q4 * M34.x - M12.y + q2 * M12.x) * invDen;
		yy[3] = (q2 * M34.y - q4 * M12.y + q2 * q4 * (M12.x - M34.x)) * invDen;
	}

	{
		//2-2
		float q2 = data_ij.tb;
		float q4 = data_ik.tb;
		cv::Point2f& M12 = data_ij.Mb;
		cv::Point2f& M34 = data_ik.Mb;

		float invDen = 1 / (q2 - q4);
		xx[4] = (M34.y - q4 * M34.x - M12.y + q2 * M12.x) * invDen;
		yy[4] = (q2 * M34.y - q4 * M12.y + q2 * q4 * (M12.x - M34.x)) * invDen;
	}

	{
		//2-1
		float q2 = data_ij.tb;
		float q4 = data_ik.ta;
		cv::Point2f& M12 = data_ij.Mb;
		cv::Point2f& M34 = data_ik.Ma;

		float invDen = 1 / (q2 - q4);
		xx[5] = (M34.y - q4 * M34.x - M12.y + q2 * M12.x) * invDen;
		yy[5] = (q2 * M34.y - q4 * M12.y + q2 * q4 * (M12.x - M34.x)) * invDen;
	}

	xx[6] = (xx[0] + xx[1]) * 0.5f;
	yy[6] = (yy[0] + yy[1]) * 0.5f;


	// 7成分の中央値を楕円の中心座標とする
	std::nth_element(xx, xx + 3, xx + 7);
	std::nth_element(yy, yy + 3, yy + 7);
	float xc = xx[3];
	float yc = yy[3];

	return cv::Point2f(xc, yc);
};

int CNEllipseDetector::FindMaxK(const int* v) const // 楕円の傾きを探索
{
	int max_val = 0;
	int max_idx = 0;
	for (int i = 0; i < ACC_R_SIZE; ++i) // ACC_R_SIZE = 180と後の方で定義
	{
		if (v[i] > max_val) { max_val = v[i], max_idx = i; }
	}

	return max_idx + 90; // オフセット
};

int CNEllipseDetector::FindMaxN(const int* v) const // 短軸/長軸比を探索
{
	int max_val = 0;
	int max_idx = 0;
	for (int i = 0; i<ACC_N_SIZE; ++i) // ACC_N_SIZE = 101
	{
		if (v[i] > max_val) { max_val = v[i], max_idx = i; }
	}

	return max_idx;
};

int CNEllipseDetector::FindMaxA(const int* v) const // 長軸長さを探索
{
	int max_val = 0;
	int max_idx = 0;
	for (int i = 0; i<ACC_A_SIZE; ++i) // ACC_A_SIZE = max(_szImg.height, _szImg.width)
	{
		if (v[i] > max_val) { max_val = v[i], max_idx = i; }
	}

	return max_idx;
};

// 1つの楕円候補の長短軸半径、角度を決定 → 閾値で正常な楕円か判定
void CNEllipseDetector::FindEllipses(cv::Point2f& center, VP& edge_i, VP& edge_j, VP& edge_k, EllipseData& data_ij, EllipseData& data_ik, std::vector<Ellipse>& ellipses)
{
	countsOfFindEllipse++;

	// アキュムレータの初期化
	memset(accN, 0, sizeof(int) * ACC_N_SIZE); // 長短軸比B/A
	memset(accR, 0, sizeof(int) * ACC_R_SIZE); // 傾き
	memset(accA, 0, sizeof(int) * ACC_A_SIZE); // 長軸長さ

	Tac(3); // 実行時間（estimation）の計算

	// 弦の傾きを格納したベクトルのサイズ計算
	int sz_ij1 = int(data_ij.Sa.size());
	int sz_ij2 = int(data_ij.Sb.size());
	int sz_ik1 = int(data_ik.Sa.size());
	int sz_ik2 = int(data_ik.Sb.size());

	// 円弧のサイズ計算
	size_t sz_ei = edge_i.size();
	size_t sz_ej = edge_j.size();
	size_t sz_ek = edge_k.size();

	// Center of the estimated ellipse
	float a0 = center.x;
	float b0 = center.y;

	// Uses 4 combinations of parameters. See Table 1 and Sect [3.2.3] of the paper.
	// ij1 and ik
	{
		float q1 = data_ij.ra;
		float q3 = data_ik.ra;
		float q5 = data_ik.rb;

		for (int ij1 = 0; ij1 < sz_ij1; ++ij1)
		{
			float q2 = data_ij.Sa[ij1];
			float q1xq2 = q1 * q2; // パフォーマンス最適化のために先に計算

			// ij1 and ik1
			for (int ik1 = 0; ik1 < sz_ik1; ++ik1)
			{
				float q4 = data_ik.Sa[ik1];
				float q3xq4 = q3 * q4;

				// See Eq. [13-18] in the paper
				float a = (q1xq2 - q3xq4); // gamma
				float b = (q3xq4 + 1) * (q1 + q2) - (q1xq2 + 1) * (q3 + q4); // beta
				float Kp = (- b + sqrt(b * b + 4 * a * a)) / (2 * a); // K+
				float zplus = ((q1 - Kp) * (q2 - Kp)) / ((1 + q1 * Kp) * (1 + q2 * Kp)); // N+の2乗
				
				if (zplus >= 0.0f) continue;

				float Np = sqrt(-zplus); // N+
				float rho = atan(Kp); // rho tmp
				int rhoDeg;
				if (Np > 1.f)
				{
					Np = 1.f / Np; // Eq. 13
					rhoDeg = cvRound((rho * 180 / CV_PI) + 180) % 180;				
				}
				else
				{
					rhoDeg = cvRound((rho * 180 / CV_PI) + 90) % 180; // Eq. 14
				}

				int iNp = cvRound(Np * 100); // [0, 100]

				if (0 <= iNp	&& iNp < ACC_N_SIZE &&
					0 <= rhoDeg	&& rhoDeg < ACC_R_SIZE
					)
				{
					++accN[iNp];	// Increment N accumulator
					++accR[rhoDeg];	// Increment R accumulator
				}
			}

			// ij1 and ik2
			for (int ik2 = 0; ik2 < sz_ik2; ++ik2)
			{
				float q4 = data_ik.Sb[ik2];
				float q5xq4 = q5 * q4;

				float a = (q1xq2 - q5xq4);
				float b = (q5xq4 + 1) * (q1 + q2) - (q1xq2 + 1) * (q5 + q4);
				float Kp = (- b + sqrt(b * b + 4 * a * a)) / (2 * a);
				float zplus = ((q1 - Kp) * (q2 - Kp)) / ((1 + q1 * Kp) * (1 + q2 * Kp));

				if (zplus >= 0.0f) continue;

				float Np = sqrt(-zplus);
				float rho = atan(Kp);
				int rhoDeg;
				if (Np > 1.f)
				{
					Np = 1.f / Np;
					rhoDeg = cvRound((rho * 180 / CV_PI) + 180) % 180;				
				}
				else
				{
					rhoDeg = cvRound((rho * 180 / CV_PI) + 90) % 180;
				}

				int iNp = cvRound(Np * 100);

				if (0 <= iNp	&& iNp < ACC_N_SIZE &&
					0 <= rhoDeg	&& rhoDeg < ACC_R_SIZE
					)
				{
					++accN[iNp];
					++accR[rhoDeg];
				}
			}
		}
	}

	//ij2 and ik
	{
		float q1 = data_ij.rb;
		float q3 = data_ik.rb;
		float q5 = data_ik.ra;

		for (int ij2 = 0; ij2 < sz_ij2; ++ij2)
		{
			float q2 = data_ij.Sb[ij2];
			float q1xq2 = q1 * q2;

			//ij2 and ik2
			for (int ik2 = 0; ik2 < sz_ik2; ++ik2)
			{
				float q4 = data_ik.Sb[ik2];
				float q3xq4 = q3 * q4;

				float a = (q1xq2 - q3xq4);
				float b = (q3xq4 + 1) * (q1 + q2) - (q1xq2 + 1) * (q3 + q4);
				float Kp = (- b + sqrt(b * b + 4 * a * a)) / (2 * a);
				float zplus = ((q1 - Kp) * (q2 - Kp)) / ((1 + q1 * Kp) * (1 + q2 * Kp));

				if (zplus >= 0.0f) continue;

				float Np = sqrt(-zplus);
				float rho = atan(Kp);
				int rhoDeg;
				if (Np > 1.f)
				{
					Np = 1.f / Np;
					rhoDeg = cvRound((rho * 180 / CV_PI) + 180) % 180;
				}
				else
				{
					rhoDeg = cvRound((rho * 180 / CV_PI) + 90) % 180;
				}

				int iNp = cvRound(Np * 100);

				if (0 <= iNp	&& iNp < ACC_N_SIZE &&
					0 <= rhoDeg	&& rhoDeg < ACC_R_SIZE
					)
				{
					++accN[iNp];
					++accR[rhoDeg];
				}
			}

			//ij2 and ik1
			for (int ik1 = 0; ik1 < sz_ik1; ++ik1)
			{
				float q4 = data_ik.Sa[ik1];
				float q5xq4 = q5 * q4;

				float a = (q1xq2 - q5xq4);
				float b = (q5xq4 + 1) * (q1 + q2) - (q1xq2 + 1) * (q5 + q4);
				float Kp = (- b + sqrt(b * b + 4 * a * a)) / (2 * a);
				float zplus = ((q1 - Kp) * (q2 - Kp)) / ((1 + q1 * Kp) * (1 + q2 * Kp));

				if (zplus >= 0.0f) continue;

				float Np = sqrt(-zplus);
				float rho = atan(Kp);
				int rhoDeg;
				if (Np > 1.f)
				{
					Np = 1.f / Np;
					rhoDeg = cvRound((rho * 180 / CV_PI) + 180) % 180;
				}
				else
				{
					rhoDeg = cvRound((rho * 180 / CV_PI) + 90) % 180;
				}

				int iNp = cvRound(Np * 100);

				if (0 <= iNp	&& iNp < ACC_N_SIZE &&
					0 <= rhoDeg	&& rhoDeg < ACC_R_SIZE
					)
				{
					++accN[iNp];
					++accR[rhoDeg];
				}
			}

		}
	}

	// Find peak in N and K accumulator
	int iN = FindMaxN(accN);
	int iK = FindMaxK(accR);

	// 物理量に変換
	float fK = float(iK);
	float Np = float(iN) * 0.01f;
	float rho = fK * float(CV_PI) / 180.f;
	float Kp = tan(rho);

	// Estimate A（長軸長さ） See Eq. [19 - 22] in Sect [3.2.3] of the paper  
	for (ushort l = 0; l < sz_ei; ++l)
	{
		cv::Point& pp = edge_i[l];
		float sk = 1.f / sqrt(Kp * Kp + 1.f); // 分母を先に計算
		float x0 = ((pp.x - a0) * sk) + (((pp.y - b0) * Kp) * sk);
		float y0 = -(((pp.x - a0) * Kp) * sk) + ((pp.y - b0) * sk);
		float Ax = sqrt((x0 * x0 * Np * Np + y0 * y0) / ((Np * Np) * (1.f + Kp * Kp)));
		int A = cvRound(abs(Ax / cos(rho)));
		if ((0 <= A) && (A < ACC_A_SIZE))
		{
			++accA[A];
		}
	}

	for (ushort l = 0; l < sz_ej; ++l)
	{
		cv::Point& pp = edge_j[l];
		float sk = 1.f / sqrt(Kp * Kp + 1.f);
		float x0 = ((pp.x - a0) * sk) + (((pp.y - b0) * Kp) * sk);
		float y0 = -(((pp.x - a0) * Kp) * sk) + ((pp.y - b0) * sk);
		float Ax = sqrt((x0 * x0 * Np * Np + y0 * y0) / ((Np * Np) * (1.f + Kp * Kp)));
		int A = cvRound(abs(Ax / cos(rho)));
		if ((0 <= A) && (A < ACC_A_SIZE))
		{
			++accA[A];
		}
	}

	for (ushort l = 0; l < sz_ek; ++l)
	{
		cv::Point& pp = edge_k[l];
		float sk = 1.f / sqrt(Kp * Kp + 1.f);
		float x0 = ((pp.x - a0) * sk) + (((pp.y - b0) * Kp) * sk);
		float y0 = -(((pp.x - a0) * Kp) * sk) + ((pp.y - b0) * sk);
		float Ax = sqrt((x0 * x0 * Np * Np + y0 * y0) / ((Np * Np) * (1.f + Kp * Kp)));
		int A = cvRound(abs(Ax / cos(rho)));
		if ((0 <= A) && (A < ACC_A_SIZE))
		{
			++accA[A];
		}
	}

	// Find peak in A accumulator
	int A = FindMaxA(accA);
	float fA = float(A);

	// Find B value. See Eq [23] in the paper
	float fB = abs(fA * Np);

	// Got all ellipse parameters!
	Ellipse ell(a0, b0, fA, fB, fmod(rho + float(CV_PI) * 2.f, float(CV_PI))); // Ellipse型のell配列を定義（_xc, _yc, _a, _b, _rad, _score）

	Toc(3); //estimation
	Tac(4); //validation

	// Get the score. See Sect [3.3.1] in the paper
	// ピクセルが楕円の輪郭上にあるかチェック
	float _cos = cos(-ell._rad);
	float _sin = sin(-ell._rad);

	float invA2 = 1.f / (ell._a * ell._a);
	float invB2 = 1.f / (ell._b * ell._b);

	float invNofPoints = 1.f / float(sz_ei + sz_ej + sz_ek);
	int counter_on_perimeter = 0;

	for (ushort l = 0; l < sz_ei; ++l)
	{
		float tx = float(edge_i[l].x) - ell._xc;         // 楕円の中心を原点に
		float ty = float(edge_i[l].y) - ell._yc;
		float rx = (tx * _cos - ty * _sin);              // 楕円の傾きを打ち消すを方向に回転
		float ry = (tx * _sin + ty * _cos);

		float h = (rx * rx) * invA2 + (ry * ry) * invB2; // 楕円の方程式
		if (abs(h - 1.f) < _fDistanceToEllipseContour)   // ピクセルが輪郭上にあれば、h=1になるはず（fDistanceToEllipseContour = 0.1f）
		{
			++counter_on_perimeter;
		}
	}

	for (ushort l = 0; l < sz_ej; ++l)
	{
		float tx = float(edge_j[l].x) - ell._xc;
		float ty = float(edge_j[l].y) - ell._yc;
		float rx = (tx * _cos - ty * _sin);
		float ry = (tx * _sin + ty * _cos);

		float h = (rx * rx) * invA2 + (ry * ry) * invB2;
		if (abs(h - 1.f) < _fDistanceToEllipseContour)
		{
			++counter_on_perimeter;
		}
	}

	for (ushort l = 0; l < sz_ek; ++l)
	{
		float tx = float(edge_k[l].x) - ell._xc;
		float ty = float(edge_k[l].y) - ell._yc;
		float rx = (tx * _cos - ty * _sin);
		float ry = (tx * _sin + ty * _cos);

		float h = (rx * rx) * invA2 + (ry * ry) * invB2;
		if (abs(h - 1.f) < _fDistanceToEllipseContour)
		{
			++counter_on_perimeter;
		}
	}

	//no points found on the ellipse
	if (counter_on_perimeter <= 0)
	{
		Toc(4); //validation
		return;
	}

	// 適合率判定(Precision)
	float pre = float(counter_on_perimeter) * invNofPoints; // fMinScore以上のピクセルが楕円の輪郭上にあればOK
	if (pre < _fMinScore) // fMinScore = 0.7
	{
		Toc(4); //validation
		return;
	}

	// 円弧の長さが楕円の円周に対して短すぎると信頼性が低いと判断（上記の別の論文を参照）
	float di, dj, dk;
	{
		cv::Point2f p1(float(edge_i[0].x), float(edge_i[0].y)); // 始点
		cv::Point2f p2(float(edge_i[sz_ei - 1].x), float(edge_i[sz_ei - 1].y)); // 終点

		// 楕円中心を原点に
		p1.x -= ell._xc;
		p1.y -= ell._yc;
		p2.x -= ell._xc;
		p2.y -= ell._yc;

		// 楕円の傾きを補正した端点座標
		cv::Point2f r1((p1.x * _cos - p1.y * _sin), (p1.x * _sin + p1.y * _cos));
		cv::Point2f r2((p2.x * _cos - p2.y * _sin), (p2.x * _sin + p2.y * _cos));
		di = abs(r2.x - r1.x) + abs(r2.y - r1.y); // マンハッタン距離
	}
	{
		cv::Point2f p1(float(edge_j[0].x), float(edge_j[0].y));
		cv::Point2f p2(float(edge_j[sz_ej - 1].x), float(edge_j[sz_ej - 1].y));
		p1.x -= ell._xc;
		p1.y -= ell._yc;
		p2.x -= ell._xc;
		p2.y -= ell._yc;
		cv::Point2f r1((p1.x * _cos - p1.y * _sin), (p1.x * _sin + p1.y * _cos));
		cv::Point2f r2((p2.x * _cos - p2.y * _sin), (p2.x * _sin + p2.y * _cos));
		dj = abs(r2.x - r1.x) + abs(r2.y - r1.y);
	}
	{
		cv::Point2f p1(float(edge_k[0].x), float(edge_k[0].y));
		cv::Point2f p2(float(edge_k[sz_ek - 1].x), float(edge_k[sz_ek - 1].y));
		p1.x -= ell._xc;
		p1.y -= ell._yc;
		p2.x -= ell._xc;
		p2.y -= ell._yc;
		cv::Point2f r1((p1.x * _cos - p1.y * _sin), (p1.x * _sin + p1.y * _cos));
		cv::Point2f r2((p2.x * _cos - p2.y * _sin), (p2.x * _sin + p2.y * _cos));
		dk = abs(r2.x - r1.x) + abs(r2.y - r1.y);
	}

	// 信頼性判定(reliability)
	float C = CV_PI * (3.0f * (ell._a + ell._b) - sqrt((3.0f * ell._a + ell._b) * (ell._a + 3.0f * ell._b))); // ラマヌジャンの第2近似式、円周長さ
	float rel = counter_on_perimeter / C; // 円周近似は3(a+b)でも可
	if (rel < _fMinReliability)
	{
		Toc(4); //validation
		return;
	}

	// 重み付き和によるスコア計算（ゲインαの調整必須）
	ell._score = pre * _scoreAlpha + rel * (1.0 - _scoreAlpha);

	// 調和平均によるスコア計算
	//ell._score = 2 * pre * rel / (pre + rel);

	// The tentative detection has been confirmed. Save it!
	ellipses.push_back(ell);

	Toc(4); // Validation
};

#define T124 pjf,pjm,pjl,pif,pim,pil
#define T231 pil,pim,pif,pjf,pjm,pjl
#define T342 pif,pim,pil,pjf,pjm,pjl
#define T413 pif,pim,pil,pjl,pjm,pjf

// 第1, 2, 4象限の円弧の組み合わせから楕円を検出（124の順でijkに振り分け）
void CNEllipseDetector::Triplets124(VVP& pi, VVP& pj, VVP& pk, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses)
{
	// それぞれの象限に振り分けられた円弧数
	ushort sz_i = ushort(pi.size());
	ushort sz_j = ushort(pj.size());
	ushort sz_k = ushort(pk.size());

	for (ushort i = 0; i < sz_i; ++i)
	{
		VP& edge_i = pi[i];
		ushort sz_ei = ushort(edge_i.size());

		cv::Point& pif = edge_i[0];
		cv::Point& pim = edge_i[sz_ei / 2];
		cv::Point& pil = edge_i[sz_ei - 1];

		VP rev_i(edge_i.size());
		reverse_copy(edge_i.begin(), edge_i.end(), rev_i.begin()); // 円弧iを逆順にしたものをrev_iに格納

		for (ushort j = 0; j < sz_j; ++j)
		{
			VP& edge_j = pj[j];
			ushort sz_ej = ushort(edge_j.size());

			cv::Point& pjf = edge_j[0];
			cv::Point& pjm = edge_j[sz_ej / 2];
			cv::Point& pjl = edge_j[sz_ej - 1];

#ifndef DISCARD_CONSTRAINT_POSITION // 円弧が正しい位置関係にあるか判定（円弧iの始点xが円弧jの終点xよりも大きいはず）
			if (pjl.x > pif.x + _fThPosition) // _fThPosition = 1.0 はピクセルの許容誤差
			{
				continue;
			}
#endif

#ifndef DISCARD_CONSTRAINT_CNC // value4SixPointsは円弧から得られる不変量で、この値がほとんど変わらなければ同一円弧とみなす。詳細はtools.cpp参照
			if(myselect1 && fabs(value4SixPoints(T124) - 1) > tCNC) // tCNC = 0.3　→　不変量が1.3以上なら違う円弧とみなしてcontinue
			{
				continue;
			}
#endif
			uint key_ij = GenerateKey(PAIR_12, i, j); // 閾値を突破したペア円弧の番号i, jのkeyを生成（まだ登録してない！）

			for (ushort k = 0; k < sz_k; ++k)
			{
				VP& edge_k = pk[k];
				ushort sz_ek = ushort(edge_k.size());

				cv::Point& pkf = edge_k[0];
				cv::Point& pkm = edge_k[sz_ek / 2];
				cv::Point& pkl = edge_k[sz_ek - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
				if (pkl.y < pil.y - _fThPosition)
				{
					continue;
				}
#endif

#ifndef DISCARD_CONSTRAINT_CNC
				if(myselect2 && fabs(value4SixPoints(pif, pim, pil, pkf, pkm, pkl) - 1) > tCNC)
				{
					continue;
				}

				if(myselect3 && fabs(value4SixPoints(pjf, pjm, pjl, pkf, pkm, pkl) - 1) > tCNC)
				{
					continue;
				}
#endif
				uint key_ik = GenerateKey(PAIR_14, i, k);

				EllipseData data_ij, data_ik;

				if (data.count(key_ij) == 0) // keyに登録されているかどうか
				{
					GetFastCenter(edge_j, rev_i, data_ij);
					data.insert(std::pair<uint, EllipseData>(key_ij, data_ij)); // key登録、以降のkループでGetFastCenterを省略できる
				}
				else
				{
					data_ij = data.at(key_ij);
				}

				if (data.count(key_ik) == 0)
				{
					GetFastCenter(edge_i, edge_k, data_ik);
					data.insert(std::pair<uint, EllipseData>(key_ik, data_ik));
				}
				else
				{
					data_ik = data.at(key_ik);
				}

				if (!data_ij.isValid || !data_ik.isValid)
				{
					continue;
				}

#ifndef DISCARD_CONSTRAINT_CENTER // 2つの推定結果の差分が閾値内かチェック（_fMaxCenterDistance2 = 25）
				if (ed2(data_ij.Cab, data_ik.Cab) > _fMaxCenterDistance2)
				{
					continue;
				}
#endif
				// Get the coordinates of the center (xc, yc)
				cv::Point2f center = GetCenterCoordinates(data_ij, data_ik);

				// Find remaining paramters (A,B,rho)
				FindEllipses(center, edge_i, edge_j, edge_k, data_ij, data_ik, ellipses);
			}
		}
	}
};

void CNEllipseDetector::Triplets231(VVP& pi, VVP& pj, VVP& pk, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses)
{
	ushort sz_i = ushort(pi.size());
	ushort sz_j = ushort(pj.size());
	ushort sz_k = ushort(pk.size());

	for (ushort i = 0; i < sz_i; ++i)
	{
		VP& edge_i = pi[i];
		ushort sz_ei = ushort(edge_i.size());

		cv::Point& pif = edge_i[0];
		cv::Point& pim = edge_i[sz_ei / 2];
		cv::Point& pil = edge_i[sz_ei - 1];

		VP rev_i(edge_i.size());
		reverse_copy(edge_i.begin(), edge_i.end(), rev_i.begin());

		for (ushort j = 0; j < sz_j; ++j)
		{
			VP& edge_j = pj[j];
			ushort sz_ej = ushort(edge_j.size());

			cv::Point& pjf = edge_j[0];
			cv::Point& pjm = edge_j[sz_ej / 2];
			cv::Point& pjl = edge_j[sz_ej - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
			if (pjf.y < pif.y - _fThPosition)
			{
				continue;
			}
#endif

#ifndef DISCARD_CONSTRAINT_CNC
			if(myselect1 && fabs(value4SixPoints(T231) - 1) > tCNC)
			{
				continue;
			}
#endif
			VP rev_j(edge_j.size());
			reverse_copy(edge_j.begin(), edge_j.end(), rev_j.begin());

			uint key_ij = GenerateKey(PAIR_23, i, j);

			for (ushort k = 0; k < sz_k; ++k)
			{
				VP& edge_k = pk[k];
				ushort sz_ek = ushort(edge_k.size());

				cv::Point& pkf = edge_k[0];
				cv::Point& pkm = edge_k[sz_ek / 2];
				cv::Point& pkl = edge_k[sz_ek - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
				if (pkf.x < pil.x - _fThPosition)
				{
					continue;
				}
#endif

#ifndef DISCARD_CONSTRAINT_CNC
				if(myselect2 && fabs(value4SixPoints(pif, pim, pil, pkf, pkm, pkl) - 1) > tCNC)
				{
					continue;
				}

				if(myselect3 && fabs(value4SixPoints(pjf, pjm, pjl, pkf, pkm, pkl) - 1) > tCNC)
				{
					continue;
				}
#endif
				uint key_ik = GenerateKey(PAIR_12, k, i);

				EllipseData data_ij, data_ik;

				if (data.count(key_ij) == 0)
				{
					GetFastCenter(rev_i, rev_j, data_ij);
					data.insert(std::pair<uint, EllipseData>(key_ij, data_ij));
				}
				else
				{
					data_ij = data.at(key_ij);
				}

				if (data.count(key_ik) == 0)
				{
					VP rev_k(edge_k.size());
					reverse_copy(edge_k.begin(), edge_k.end(), rev_k.begin());

					GetFastCenter(edge_i, rev_k, data_ik);
					data.insert(std::pair<uint, EllipseData>(key_ik, data_ik));
				}
				else
				{
					data_ik = data.at(key_ik);
				}

				if (!data_ij.isValid || !data_ik.isValid)
				{
					continue;
				}

#ifndef DISCARD_CONSTRAINT_CENTER
				if (ed2(data_ij.Cab, data_ik.Cab) > _fMaxCenterDistance2)
				{
					continue;
				}
#endif
				cv::Point2f center = GetCenterCoordinates(data_ij, data_ik);

				FindEllipses(center, edge_i, edge_j, edge_k, data_ij, data_ik, ellipses);
			}
		}
	}
};

void CNEllipseDetector::Triplets342(VVP& pi, VVP& pj, VVP& pk, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses)
{
	ushort sz_i = ushort(pi.size());
	ushort sz_j = ushort(pj.size());
	ushort sz_k = ushort(pk.size());

	for (ushort i = 0; i < sz_i; ++i)
	{
		VP& edge_i = pi[i];
		ushort sz_ei = ushort(edge_i.size());

		cv::Point& pif = edge_i[0];
		cv::Point& pim = edge_i[sz_ei / 2];
		cv::Point& pil = edge_i[sz_ei - 1];

		VP rev_i(edge_i.size());
		reverse_copy(edge_i.begin(), edge_i.end(), rev_i.begin());

		for (ushort j = 0; j < sz_j; ++j)
		{
			VP& edge_j = pj[j];
			ushort sz_ej = ushort(edge_j.size());

			cv::Point& pjf = edge_j[0];
			cv::Point& pjm = edge_j[sz_ej / 2];
			cv::Point& pjl = edge_j[sz_ej - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
			if (pjf.x < pil.x - _fThPosition)
			{
				continue;
			}
#endif

#ifndef DISCARD_CONSTRAINT_CNC
			if(myselect1 && fabs(value4SixPoints(T342) - 1) > tCNC)
			{
				continue;
			}
#endif
			VP rev_j(edge_j.size());
			reverse_copy(edge_j.begin(), edge_j.end(), rev_j.begin());

			uint key_ij = GenerateKey(PAIR_34, i, j);

			for (ushort k = 0; k < sz_k; ++k)
			{
				VP& edge_k = pk[k];
				ushort sz_ek = ushort(edge_k.size());

				cv::Point& pkf = edge_k[0];
				cv::Point& pkm = edge_k[sz_ek / 2];
				cv::Point& pkl = edge_k[sz_ek - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
				if (pkf.y > pif.y + _fThPosition)
				{
					continue;
				}
#endif

#ifndef DISCARD_CONSTRAINT_CNC
				if(myselect2 && fabs(value4SixPoints(pif, pim, pil, pkf, pkm, pkl) - 1) > tCNC)
				{
					continue;
				}

				if(myselect3 && fabs(value4SixPoints(pjf, pjm, pjl, pkf, pkm, pkl) - 1) > tCNC)
				{
					continue;
				}
#endif
				uint key_ik = GenerateKey(PAIR_23, k, i);

				EllipseData data_ij, data_ik;

				if (data.count(key_ij) == 0)
				{
					GetFastCenter(edge_i, rev_j, data_ij);
					data.insert(std::pair<uint, EllipseData>(key_ij, data_ij));
				}
				else
				{
					data_ij = data.at(key_ij);
				}

				if (data.count(key_ik) == 0)
				{
					VP rev_k(edge_k.size());
					reverse_copy(edge_k.begin(), edge_k.end(), rev_k.begin());

					GetFastCenter(rev_i, rev_k, data_ik);

					data.insert(std::pair<uint, EllipseData>(key_ik, data_ik));
				}
				else
				{
					data_ik = data.at(key_ik);
				}

				if (!data_ij.isValid || !data_ik.isValid)
				{
					continue;
				}

#ifndef DISCARD_CONSTRAINT_CENTER
				if (ed2(data_ij.Cab, data_ik.Cab) > _fMaxCenterDistance2)
				{
					continue;
				}
#endif
				cv::Point2f center = GetCenterCoordinates(data_ij, data_ik);
				FindEllipses(center, edge_i, edge_j, edge_k, data_ij, data_ik, ellipses);
			}
		}
	}
};

void CNEllipseDetector::Triplets413(VVP& pi, VVP& pj, VVP& pk, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses)
{
	ushort sz_i = ushort(pi.size());
	ushort sz_j = ushort(pj.size());
	ushort sz_k = ushort(pk.size());

	for (ushort i = 0; i < sz_i; ++i)
	{
		VP& edge_i = pi[i];
		ushort sz_ei = ushort(edge_i.size());

		cv::Point& pif = edge_i[0];
		cv::Point& pim = edge_i[sz_ei / 2];
		cv::Point& pil = edge_i[sz_ei - 1];

		VP rev_i(edge_i.size());
		reverse_copy(edge_i.begin(), edge_i.end(), rev_i.begin());

		for (ushort j = 0; j < sz_j; ++j)
		{
			VP& edge_j = pj[j];
			ushort sz_ej = ushort(edge_j.size());

			cv::Point& pjf = edge_j[0];
			cv::Point& pjm = edge_j[sz_ej / 2];
			cv::Point& pjl = edge_j[sz_ej - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
			if (pjl.y > pil.y + _fThPosition)
			{
				continue;
			}
#endif

#ifndef DISCARD_CONSTRAINT_CNC
			if(myselect1 && fabs(value4SixPoints(T413) - 1) > tCNC)
			{
				continue;
			}
#endif
			uint key_ij = GenerateKey(PAIR_14, j, i);

			for (ushort k = 0; k < sz_k; ++k)
			{
				VP& edge_k = pk[k];
				ushort sz_ek = ushort(edge_k.size());

				cv::Point& pkf = edge_k[0];
				cv::Point& pkm = edge_k[sz_ek / 2];
				cv::Point& pkl = edge_k[sz_ek - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
				if (pkl.x > pif.x + _fThPosition)
				{
					continue;
				}
#endif

#ifndef DISCARD_CONSTRAINT_CNC
				if(myselect2 && fabs(value4SixPoints(pif, pim, pil, pkf, pkm, pkl) - 1) > tCNC)
				{
					continue;
				}

				if(myselect3 && fabs(value4SixPoints(pjf, pjm, pjl, pkf, pkm, pkl) - 1) > tCNC)
				{
					continue;
				}
#endif
				uint key_ik = GenerateKey(PAIR_34, k, i);

				EllipseData data_ij, data_ik;

				if (data.count(key_ij) == 0)
				{
					GetFastCenter(edge_i, edge_j, data_ij);
					data.insert(std::pair<uint, EllipseData>(key_ij, data_ij));
				}
				else
				{
					data_ij = data.at(key_ij);
				}

				if (data.count(key_ik) == 0)
				{
					GetFastCenter(rev_i, edge_k, data_ik);
					data.insert(std::pair<uint, EllipseData>(key_ik, data_ik));
				}
				else
				{
					data_ik = data.at(key_ik);
				}

				if (!data_ij.isValid || !data_ik.isValid)
				{
					continue;
				}

#ifndef DISCARD_CONSTRAINT_CENTER
				if (ed2(data_ij.Cab, data_ik.Cab) > _fMaxCenterDistance2)
				{
					continue;
				}
#endif
				cv::Point2f center = GetCenterCoordinates(data_ij, data_ik);

				FindEllipses(center, edge_i, edge_j, edge_k, data_ij, data_ik, ellipses);
			}
		}
	}
};

// 円弧2つの場合の楕円探索（最小二乗法）→Doublets13,24で呼び出し
void CNEllipseDetector::FindEllipses2(VP& edge_ij, std::vector<Ellipse>& ellipses)
{
	size_t sz_ij = edge_ij.size();
	cv::RotatedRect fit_ell = cv::fitEllipse(edge_ij);

	cv::Point2f center = fit_ell.center;
	float xc = center.x;
	float yc = center.y;
	float width = fit_ell.size.width / 2.0f;
	float height = fit_ell.size.height / 2.0f;
	float fA = std::max(width, height);
	float fB = std::min(width, height);
	float angle_deg = fit_ell.angle;
	if (width < height) {angle_deg += 90.0f;}
	float rho = fmod((angle_deg * CV_PI / 180.0f) + 2.0 * CV_PI, CV_PI);

	Ellipse ell(xc, yc, fA, fB, rho);
	
	float _cos = cos(-ell._rad);
	float _sin = sin(-ell._rad);
	float invA2 = 1.f / (ell._a * ell._a);
	float invB2 = 1.f / (ell._b * ell._b);
	float invNofPoints = 1.f / float(sz_ij);
	int counter_on_perimeter = 0;

	for (ushort l = 0; l < sz_ij; ++l)
	{
		float tx = float(edge_ij[l].x) - ell._xc;         // 楕円の中心を原点に
		float ty = float(edge_ij[l].y) - ell._yc;
		float rx = (tx * _cos - ty * _sin);              // 楕円の傾きを打ち消すを方向に回転
		float ry = (tx * _sin + ty * _cos);

		float h = (rx * rx) * invA2 + (ry * ry) * invB2; // 楕円の方程式
		if (abs(h - 1.f) < _fDistanceToEllipseContour)   // ピクセルが輪郭上にあれば、h=1になるはず（fDistanceToEllipseContour = 0.1f）
		{
			++counter_on_perimeter;
		}
	}

	if (counter_on_perimeter <= 0)
	{
		Toc(4); //validation
		return;
	}

	// 適合率判定
	float pre = float(counter_on_perimeter) * invNofPoints;
	if (pre < _fMinScore)
	{
		Toc(4); //validation
		return;
	}

	// 信頼性判定
	float C = CV_PI * (3.0f * (ell._a + ell._b) - sqrt((3.0f * ell._a + ell._b) * (ell._a + 3.0f * ell._b))); // ラマヌジャンの第2近似式、円周長さ
	float rel = counter_on_perimeter / C;
	if (rel < _fMinReliability)
	{
		Toc(4); //validation
		return;
	}

	ell._score = pre * _scoreAlpha + rel * (1.0 - _scoreAlpha);
	//ell._score = 2 * pre * rel / (pre + rel);
	ellipses.push_back(ell);
	Toc(4);
}

// 円弧2つの場合の楕円探索（Tripletsと同じ幾何計算）
void CNEllipseDetector::FindEllipses3(cv::Point2f& center, VP& edge_i, VP& edge_j, EllipseData& data_ij, std::vector<Ellipse>& ellipses)
{
	countsOfFindEllipse++;

	// アキュムレータの初期化
	memset(accN, 0, sizeof(int) * ACC_N_SIZE); // 長短軸比B/A
	memset(accR, 0, sizeof(int) * ACC_R_SIZE); // 傾き
	memset(accA, 0, sizeof(int) * ACC_A_SIZE); // 長軸長さ

	Tac(3); // 実行時間（estimation）の計算

	// 弦の傾きを格納したベクトルのサイズ計算
	int sz_ij1 = int(data_ij.Sa.size());
	int sz_ij2 = int(data_ij.Sb.size());

	// 円弧のサイズ計算
	size_t sz_ei = edge_i.size();
	size_t sz_ej = edge_j.size();

	// Center of the estimated ellipse
	float a0 = center.x;
	float b0 = center.y;

	// Uses 4 combinations of parameters. See Table 1 and Sect [3.2.3] of the paper.
	{
		float q1 = data_ij.ra;
		float q3 = data_ij.rb;

		for (int ij1 = 0; ij1 < sz_ij1; ++ij1)
		{
			float q2 = data_ij.Sa[ij1];
			float q1xq2 = q1 * q2; // パフォーマンス最適化のために先に計算

			for (int ij2 = 0; ij2 < sz_ij2; ++ij2)
			{
				float q4 = data_ij.Sb[ij2];
				float q3xq4 = q3 * q4;

				// See Eq. [13-18] in the paper
				float a = (q1xq2 - q3xq4); // gamma
				float b = (q3xq4 + 1) * (q1 + q2) - (q1xq2 + 1) * (q3 + q4); // beta
				float Kp = (- b + sqrt(b * b + 4 * a * a)) / (2 * a); // K+
				float zplus = ((q1 - Kp) * (q2 - Kp)) / ((1 + q1 * Kp) * (1 + q2 * Kp)); // N+の2乗
				
				if (zplus >= 0.0f) continue;

				float Np = sqrt(-zplus); // N+
				float rho = atan(Kp); // rho tmp
				int rhoDeg;
				if (Np > 1.f)
				{
					Np = 1.f / Np; // Eq. 13
					rhoDeg = cvRound((rho * 180 / CV_PI) + 180) % 180;				
				}
				else
				{
					rhoDeg = cvRound((rho * 180 / CV_PI) + 90) % 180; // Eq. 14
				}

				int iNp = cvRound(Np * 100); // [0, 100]

				if (0 <= iNp	&& iNp < ACC_N_SIZE &&
					0 <= rhoDeg	&& rhoDeg < ACC_R_SIZE
					)
				{
					++accN[iNp];	// Increment N accumulator
					++accR[rhoDeg];	// Increment R accumulator
				}
			}
        }
    }

	// Find peak in N and K accumulator
	int iN = FindMaxN(accN);
	int iK = FindMaxK(accR);

	// 物理量に変換
	float fK = float(iK);
	float Np = float(iN) * 0.01f;
	float rho = fK * float(CV_PI) / 180.f;
	float Kp = tan(rho);

	// Estimate A（長軸長さ） See Eq. [19 - 22] in Sect [3.2.3] of the paper  
	for (ushort l = 0; l < sz_ei; ++l)
	{
		cv::Point& pp = edge_i[l];
		float sk = 1.f / sqrt(Kp * Kp + 1.f); // 分母を先に計算
		float x0 = ((pp.x - a0) * sk) + (((pp.y - b0) * Kp) * sk);
		float y0 = -(((pp.x - a0) * Kp) * sk) + ((pp.y - b0) * sk);
		float Ax = sqrt((x0 * x0 * Np * Np + y0 * y0) / ((Np * Np) * (1.f + Kp * Kp)));
		int A = cvRound(abs(Ax / cos(rho)));
		if ((0 <= A) && (A < ACC_A_SIZE))
		{
			++accA[A];
		}
	}

	for (ushort l = 0; l < sz_ej; ++l)
	{
		cv::Point& pp = edge_j[l];
		float sk = 1.f / sqrt(Kp * Kp + 1.f);
		float x0 = ((pp.x - a0) * sk) + (((pp.y - b0) * Kp) * sk);
		float y0 = -(((pp.x - a0) * Kp) * sk) + ((pp.y - b0) * sk);
		float Ax = sqrt((x0 * x0 * Np * Np + y0 * y0) / ((Np * Np) * (1.f + Kp * Kp)));
		int A = cvRound(abs(Ax / cos(rho)));
		if ((0 <= A) && (A < ACC_A_SIZE))
		{
			++accA[A];
		}
	}

	// Find peak in A accumulator
	int A = FindMaxA(accA);
	float fA = float(A);

	// Find B value. See Eq [23] in the paper
	float fB = abs(fA * Np);

	// Got all ellipse parameters!
	Ellipse ell(a0, b0, fA, fB, fmod(rho + float(CV_PI) * 2.f, float(CV_PI))); // Ellipse型のell配列を定義（_xc, _yc, _a, _b, _rad, _score）

	Toc(3); //estimation
	Tac(4); //validation

	// Get the score. See Sect [3.3.1] in the paper
	// ピクセルが楕円の輪郭上にあるかチェック
	float _cos = cos(-ell._rad);
	float _sin = sin(-ell._rad);

	float invA2 = 1.f / (ell._a * ell._a);
	float invB2 = 1.f / (ell._b * ell._b);

	float invNofPoints = 1.f / float(sz_ei + sz_ej);
	int counter_on_perimeter = 0;

	for (ushort l = 0; l < sz_ei; ++l)
	{
		float tx = float(edge_i[l].x) - ell._xc;         // 楕円の中心を原点に
		float ty = float(edge_i[l].y) - ell._yc;
		float rx = (tx * _cos - ty * _sin);              // 楕円の傾きを打ち消すを方向に回転
		float ry = (tx * _sin + ty * _cos);

		float h = (rx * rx) * invA2 + (ry * ry) * invB2; // 楕円の方程式
		if (abs(h - 1.f) < _fDistanceToEllipseContour)   // ピクセルが輪郭上にあれば、h=1になるはず（fDistanceToEllipseContour = 0.1f）
		{
			++counter_on_perimeter;
		}
	}

	for (ushort l = 0; l < sz_ej; ++l)
	{
		float tx = float(edge_j[l].x) - ell._xc;
		float ty = float(edge_j[l].y) - ell._yc;
		float rx = (tx * _cos - ty * _sin);
		float ry = (tx * _sin + ty * _cos);

		float h = (rx * rx) * invA2 + (ry * ry) * invB2;
		if (abs(h - 1.f) < _fDistanceToEllipseContour)
		{
			++counter_on_perimeter;
		}
	}

	//no points found on the ellipse
	if (counter_on_perimeter <= 0)
	{
		Toc(4); //validation
		return;
	}

	// 適合率判定(Precision)
	float pre = float(counter_on_perimeter) * invNofPoints; // fMinScore以上のピクセルが楕円の輪郭上にあればOK
	if (pre < _fMinScore) // fMinScore = 0.7
	{
		Toc(4); //validation
		return;
	}

	// 円弧の長さが楕円の円周に対して短すぎると信頼性が低いと判断（上記の別の論文を参照）
	float di, dj;
	{
		cv::Point2f p1(float(edge_i[0].x), float(edge_i[0].y)); // 始点
		cv::Point2f p2(float(edge_i[sz_ei - 1].x), float(edge_i[sz_ei - 1].y)); // 終点

		// 楕円中心を原点に
		p1.x -= ell._xc;
		p1.y -= ell._yc;
		p2.x -= ell._xc;
		p2.y -= ell._yc;

		// 楕円の傾きを補正した端点座標
		cv::Point2f r1((p1.x * _cos - p1.y * _sin), (p1.x * _sin + p1.y * _cos));
		cv::Point2f r2((p2.x * _cos - p2.y * _sin), (p2.x * _sin + p2.y * _cos));
		di = abs(r2.x - r1.x) + abs(r2.y - r1.y); // マンハッタン距離
	}
	{
		cv::Point2f p1(float(edge_j[0].x), float(edge_j[0].y));
		cv::Point2f p2(float(edge_j[sz_ej - 1].x), float(edge_j[sz_ej - 1].y));
		p1.x -= ell._xc;
		p1.y -= ell._yc;
		p2.x -= ell._xc;
		p2.y -= ell._yc;
		cv::Point2f r1((p1.x * _cos - p1.y * _sin), (p1.x * _sin + p1.y * _cos));
		cv::Point2f r2((p2.x * _cos - p2.y * _sin), (p2.x * _sin + p2.y * _cos));
		dj = abs(r2.x - r1.x) + abs(r2.y - r1.y);
	}

	// 信頼性判定(reliability)
	float C = CV_PI * (3.0f * (ell._a + ell._b) - sqrt((3.0f * ell._a + ell._b) * (ell._a + 3.0f * ell._b))); // ラマヌジャンの第2近似式、円周長さ
	float rel = counter_on_perimeter / C; // 円周近似は3(a+b)でも可
	if (rel < _fMinReliability)
	{
		Toc(4); //validation
		return;
	}

	// 重み付き和によるスコア計算（ゲインαの調整必須）
	ell._score = pre * _scoreAlpha + rel * (1.0 - _scoreAlpha);

	// 調和平均によるスコア計算
	//ell._score = 2 * pre * rel / (pre + rel);

	// The tentative detection has been confirmed. Save it!
	ellipses.push_back(ell);

	Toc(4); // Validation
}

// 順番合ってるかわからん
#define D12 pjf,pjm,pjl,pif,pim,pil
#define D23 pil,pim,pif,pjf,pjm,pjl
#define D34 pif,pim,pil,pjf,pjm,pjl
#define D41 pif,pim,pil,pjl,pjm,pjf
#define D13 pif,pim,pil,pjf,pjm,pjl
#define D24 pil,pim,pif,pjf,pjm,pjl

// 自作、ダブレット
void CNEllipseDetector::Doublets12(VVP& pi, VVP& pj, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses)
{
	ushort sz_i = ushort(pi.size());
	ushort sz_j = ushort(pj.size());

	for (ushort i = 0; i < sz_i; ++i)
	{
		VP& edge_i = pi[i];
		ushort sz_ei = ushort(edge_i.size());

		cv::Point& pif = edge_i[0];
		cv::Point& pim = edge_i[sz_ei / 2];
		cv::Point& pil = edge_i[sz_ei - 1];

		VP rev_i(edge_i.size());
		reverse_copy(edge_i.begin(), edge_i.end(), rev_i.begin()); // 円弧iを逆順にしたものをrev_iに格納

		for (ushort j = 0; j < sz_j; ++j)
		{
			VP& edge_j = pj[j];
			ushort sz_ej = ushort(edge_j.size());

			cv::Point& pjf = edge_j[0];
			cv::Point& pjm = edge_j[sz_ej / 2];
			cv::Point& pjl = edge_j[sz_ej - 1];

#ifndef DISCARD_CONSTRAINT_POSITION // 円弧が正しい位置関係にあるか判定（円弧iの始点xが円弧jの終点xよりも大きいはず）
			if (pjl.x > pif.x + _fThPosition) // _fThPosition = 1.0 はピクセルの許容誤差
			{
				continue;
			}
#endif

#ifndef DISCARD_CONSTRAINT_CNC // value4SixPointsは円弧から得られる不変量で、この値がほとんど変わらなければ同一円弧とみなす。詳細はtools.cpp参照
			if(myselect1 && fabs(value4SixPoints(D12) - 1) > tCNC) // tCNC = 0.3　→　不変量が1.3以上なら違う円弧とみなしてcontinue
			{
				continue;
			}
#endif
			uint key_ij = GenerateKey(PAIR_12, i, j);
			EllipseData data_ij;

			if (data.count(key_ij) == 0) // keyに登録されているかどうか
			{
				GetFastCenter(edge_j, rev_i, data_ij);
				data.insert(std::pair<uint, EllipseData>(key_ij, data_ij));
			}
			else
			{
				data_ij = data.at(key_ij);
			}

			if (!data_ij.isValid)
			{
				continue;
			}

			// ペア判定となったエッジi, jを統合
			VP edge_ij = edge_i;
			edge_ij.insert(edge_ij.end(), edge_j.begin(), edge_j.end());

			// Find remaining paramters (A,B,rho)
			//FindEllipses2(edge_ij, ellipses);

            cv::Point2f& center = data_ij.Cab;
            FindEllipses3(center, edge_i, edge_j, data_ij, ellipses);
		}
	}
}

void CNEllipseDetector::Doublets23(VVP& pi, VVP& pj, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses)
{
	ushort sz_i = ushort(pi.size());
	ushort sz_j = ushort(pj.size());

	for (ushort i = 0; i < sz_i; ++i)
	{
		VP& edge_i = pi[i];
		ushort sz_ei = ushort(edge_i.size());

		cv::Point& pif = edge_i[0];
		cv::Point& pim = edge_i[sz_ei / 2];
		cv::Point& pil = edge_i[sz_ei - 1];

		VP rev_i(edge_i.size());
		reverse_copy(edge_i.begin(), edge_i.end(), rev_i.begin());

		for (ushort j = 0; j < sz_j; ++j)
		{
			VP& edge_j = pj[j];
			ushort sz_ej = ushort(edge_j.size());

			cv::Point& pjf = edge_j[0];
			cv::Point& pjm = edge_j[sz_ej / 2];
			cv::Point& pjl = edge_j[sz_ej - 1];

			VP rev_j(edge_j.size());
			reverse_copy(edge_j.begin(), edge_j.end(), rev_j.begin());

#ifndef DISCARD_CONSTRAINT_POSITION
			if (pjf.y < pif.y - _fThPosition)
			{
				continue;
			}
#endif

#ifndef DISCARD_CONSTRAINT_CNC
			if(myselect1 && fabs(value4SixPoints(D23) - 1) > tCNC)
			{
				continue;
			}
#endif
			uint key_ij = GenerateKey(PAIR_23, i, j);
			EllipseData data_ij;

			if (data.count(key_ij) == 0)
			{
				GetFastCenter(rev_i, rev_j, data_ij);
				data.insert(std::pair<uint, EllipseData>(key_ij, data_ij));
			}
			else
			{
				data_ij = data.at(key_ij);
			}

			if (!data_ij.isValid)
			{
				continue;
			}

			VP edge_ij = edge_i;
			edge_ij.insert(edge_ij.end(), edge_j.begin(), edge_j.end());

			//FindEllipses2(edge_ij, ellipses);

            cv::Point2f& center = data_ij.Cab;
            FindEllipses3(center, edge_i, edge_j, data_ij, ellipses);
		}
	}
}

void CNEllipseDetector::Doublets34(VVP& pi, VVP& pj, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses)
{
	ushort sz_i = ushort(pi.size());
	ushort sz_j = ushort(pj.size());

	for (ushort i = 0; i < sz_i; ++i)
	{
		VP& edge_i = pi[i];
		ushort sz_ei = ushort(edge_i.size());

		cv::Point& pif = edge_i[0];
		cv::Point& pim = edge_i[sz_ei / 2];
		cv::Point& pil = edge_i[sz_ei - 1];

		for (ushort j = 0; j < sz_j; ++j)
		{
			VP& edge_j = pj[j];
			ushort sz_ej = ushort(edge_j.size());

			cv::Point& pjf = edge_j[0];
			cv::Point& pjm = edge_j[sz_ej / 2];
			cv::Point& pjl = edge_j[sz_ej - 1];

			VP rev_j(edge_j.size());
			reverse_copy(edge_j.begin(), edge_j.end(), rev_j.begin());

#ifndef DISCARD_CONSTRAINT_POSITION
			if (pjf.x < pil.x - _fThPosition)
			{
				continue;
			}
#endif

#ifndef DISCARD_CONSTRAINT_CNC
			if(myselect1 && fabs(value4SixPoints(D34) - 1) > tCNC)
			{
				continue;
			}
#endif
			uint key_ij = GenerateKey(PAIR_34, i, j);
			EllipseData data_ij;

			if (data.count(key_ij) == 0)
			{
				GetFastCenter(edge_i, rev_j, data_ij);
				data.insert(std::pair<uint, EllipseData>(key_ij, data_ij));
			}
			else
			{
				data_ij = data.at(key_ij);
			}

			if (!data_ij.isValid)
			{
				continue;
			}

			VP edge_ij = edge_i;
			edge_ij.insert(edge_ij.end(), edge_j.begin(), edge_j.end());

			//FindEllipses2(edge_ij, ellipses);

            cv::Point2f& center = data_ij.Cab;
            FindEllipses3(center, edge_i, edge_j, data_ij, ellipses);
		}
	}
}

void CNEllipseDetector::Doublets41(VVP& pi, VVP& pj, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses)
{
	ushort sz_i = ushort(pi.size());
	ushort sz_j = ushort(pj.size());

	for (ushort i = 0; i < sz_i; ++i)
	{
		VP& edge_i = pi[i];
		ushort sz_ei = ushort(edge_i.size());

		cv::Point& pif = edge_i[0];
		cv::Point& pim = edge_i[sz_ei / 2];
		cv::Point& pil = edge_i[sz_ei - 1];

		for (ushort j = 0; j < sz_j; ++j)
		{
			VP& edge_j = pj[j];
			ushort sz_ej = ushort(edge_j.size());

			cv::Point& pjf = edge_j[0];
			cv::Point& pjm = edge_j[sz_ej / 2];
			cv::Point& pjl = edge_j[sz_ej - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
			if (pjl.y > pil.y + _fThPosition)
			{
				continue;
			}
#endif

#ifndef DISCARD_CONSTRAINT_CNC
			if(myselect1 && fabs(value4SixPoints(D41) - 1) > tCNC)
			{
				continue;
			}
#endif
			uint key_ij = GenerateKey(PAIR_14, i, j);
			EllipseData data_ij;

			if (data.count(key_ij) == 0)
			{
				GetFastCenter(edge_i, edge_j, data_ij);
				data.insert(std::pair<uint, EllipseData>(key_ij, data_ij));
			}
			else
			{
				data_ij = data.at(key_ij);
			}

			if (!data_ij.isValid)
			{
				continue;
			}

			VP edge_ij = edge_i;
			edge_ij.insert(edge_ij.end(), edge_j.begin(), edge_j.end());

			//FindEllipses2(edge_ij, ellipses);

            cv::Point2f& center = data_ij.Cab;
            FindEllipses3(center, edge_i, edge_j, data_ij, ellipses);
		}
	}
}

void CNEllipseDetector::Doublets13(VVP& pi, VVP& pj, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses)
{
	ushort sz_i = ushort(pi.size());
	ushort sz_j = ushort(pj.size());

	for (ushort i = 0; i < sz_i; ++i)
	{
		VP& edge_i = pi[i];
		ushort sz_ei = ushort(edge_i.size());

		cv::Point& pif = edge_i[0];
		cv::Point& pim = edge_i[sz_ei / 2];
		cv::Point& pil = edge_i[sz_ei - 1];

		for (ushort j = 0; j < sz_j; ++j)
		{
			VP& edge_j = pj[j];
			ushort sz_ej = ushort(edge_j.size());

			cv::Point& pjf = edge_j[0];
			cv::Point& pjm = edge_j[sz_ej / 2];
			cv::Point& pjl = edge_j[sz_ej - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
			if (pjf.x > pil.x || pif.y > pjl.y)
			{
				continue;
			}
#endif

#ifndef DISCARD_CONSTRAINT_CNC
			if(myselect1 && fabs(value4SixPoints(D13) - 1) > tCNC)
			{
				continue;
			}
#endif
			uint key_ij = GenerateKey(PAIR_13, i, j);
			EllipseData data_ij;

			if (data.count(key_ij) == 0)
			{
				GetFastCenter(edge_i, edge_j, data_ij);
				data.insert(std::pair<uint, EllipseData>(key_ij, data_ij));
			}
			else
			{
				data_ij = data.at(key_ij);
			}

			if (!data_ij.isValid)
			{
				continue;
			}

			VP edge_ij = edge_i;
			edge_ij.insert(edge_ij.end(), edge_j.begin(), edge_j.end());

			FindEllipses2(edge_ij, ellipses);

            //cv::Point2f& center = data_ij.Cab;
            //FindEllipses3(center, edge_i, edge_j, data_ij, ellipses);
		}
	}
}

void CNEllipseDetector::Doublets24(VVP& pi, VVP& pj, std::unordered_map<uint, EllipseData>& data, std::vector<Ellipse>& ellipses)
{
	ushort sz_i = ushort(pi.size());
	ushort sz_j = ushort(pj.size());

	for (ushort i = 0; i < sz_i; ++i)
	{
		VP& edge_i = pi[i];
		ushort sz_ei = ushort(edge_i.size());

		cv::Point& pif = edge_i[0];
		cv::Point& pim = edge_i[sz_ei / 2];
		cv::Point& pil = edge_i[sz_ei - 1];

		for (ushort j = 0; j < sz_j; ++j)
		{
			VP& edge_j = pj[j];
			ushort sz_ej = ushort(edge_j.size());

			cv::Point& pjf = edge_j[0];
			cv::Point& pjm = edge_j[sz_ej / 2];
			cv::Point& pjl = edge_j[sz_ej - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
			if (pif.x > pjl.x || pil.y > pjf.y)
			{
				continue;
			}
#endif

#ifndef DISCARD_CONSTRAINT_CNC
			if(myselect1 && fabs(value4SixPoints(D24) - 1) > tCNC)
			{
				continue;
			}
#endif
			uint key_ij = GenerateKey(PAIR_24, i, j);
			EllipseData data_ij;

			if (data.count(key_ij) == 0)
			{
				GetFastCenter(edge_i, edge_j, data_ij);
				data.insert(std::pair<uint, EllipseData>(key_ij, data_ij));
			}
			else
			{
				data_ij = data.at(key_ij);
			}

			if (!data_ij.isValid)
			{
				continue;
			}

			VP edge_ij = edge_i;
			edge_ij.insert(edge_ij.end(), edge_j.begin(), edge_j.end());

			FindEllipses2(edge_ij, ellipses);

            //cv::Point2f& center = data_ij.Cab;
            //FindEllipses3(center, edge_i, edge_j, data_ij, ellipses);
		}
	}
}

// 自作、楕円の選定（極端な楕円の削除）
void CNEllipseDetector::DeleteEllipses(std::vector<Ellipse>& ellipses)
{
	int iNumOfEllipses = int(ellipses.size());
	int half = iNumOfEllipses / 2;
	if (iNumOfEllipses == 0) return;

	std::vector<float> fA(iNumOfEllipses);

	for (int i = 0; i < iNumOfEllipses; i++)
	{
		fA[i] = ellipses[i]._a;
	}

	// 長軸長さaの中央値を取得
	std::nth_element(fA.begin(), fA.begin() + half, fA.end());
	float med_a = fA[half];

	float th_a = 4.0f;  // 中央値より4倍以上の長軸を削除
	float th_ba = 0.7f; // 長短軸比0.8未満は削除

	std::vector<Ellipse> noDelete;
	for (int i = 0; i < iNumOfEllipses; i++)
	{
		Ellipse& ell = ellipses[i]; // チェック対象
		float a_e = ell._a;
		float ba_e = ell._b / ell._a;

		if (a_e > med_a * th_a)
		{
			continue;
		}

		if (ba_e < th_ba)
		{
			continue;
		}

		noDelete.push_back(ell);
	}

	noDelete.swap(ellipses);
}

// 自作、重なる楕円選定（ピクセル単位で判断）
void CNEllipseDetector::OverlapEllipses(const cv::Size& sz, std::vector<Ellipse>& ellipses)
{
	int iNumOfEllipses = int(ellipses.size());
    if (iNumOfEllipses <= 1) return;

	std::vector<Ellipse> overlap;
    overlap.push_back(ellipses[0]); // 最初の（最高スコアの）楕円は無条件に合格
	
	// 楕円描画用のマスクを作成
	cv::Mat mask1 = cv::Mat::zeros(sz, CV_8UC1);
    cv::Mat mask2 = cv::Mat::zeros(sz, CV_8UC1);
    cv::Mat overlapMask = cv::Mat::zeros(sz, CV_8UC1);

	for (int i = 1; i < iNumOfEllipses; ++i) // スコア2番目以降の楕円をチェック
    {
        Ellipse& e1 = ellipses[i];
        bool bFoundOverlap = false;

        mask1.setTo(0);
        cv::ellipse(mask1,
                    cv::Point(cvRound(e1._xc), cvRound(e1._yc)), // cvRound : 四捨五入
                    cv::Size(cvRound(e1._a), cvRound(e1._b)),
                    e1._rad * 180.0 / CV_PI, 0.0, 360.0,
                    cv::Scalar(255), -1);

        for (const auto& e2 : overlap) // overlap内のベクトルをe2という名前で順番に取り出すループ（overlap内にはe1より高スコアの楕円しかない）
        {
            mask2.setTo(0);
            cv::ellipse(mask2,
                        cv::Point(cvRound(e2._xc), cvRound(e2._yc)),
                        cv::Size(cvRound(e2._a), cvRound(e2._b)),
                        e2._rad * 180.0 / CV_PI, 0.0, 360.0,
                        cv::Scalar(255), -1);

            cv::bitwise_and(mask1, mask2, overlapMask); // 2つのマスクの重なりを抽出

            if (cv::countNonZero(overlapMask) > 0)
            {
                bFoundOverlap = true;
                break;
            }
        }

        if (!bFoundOverlap)
        {
            overlap.push_back(e1);
        }
    }

    overlap.swap(ellipses);
};

// クラスタの統合（同一楕円に複数描画されるのを防ぐ）（未使用）
void CNEllipseDetector::ClusterEllipses(std::vector<Ellipse>& ellipses)
{
	float th_Da = 0.1f; // 長軸半径の差分
	float th_Db = 0.1f; // 短軸半径の差分
	float th_Dr = 0.1f; // 角度の差分
	float th_Dc_ratio = 0.1f; // 中心間距離
	float th_Dr_circle = 0.9f; // 長短軸比の閾値

	int iNumOfEllipses = int(ellipses.size());
	if (iNumOfEllipses == 0) return;

	std::vector<Ellipse> clusters;
	clusters.push_back(ellipses[0]); // 最初（最高スコア）の楕円は無条件に合格

	bool bFoundCluster = false;

	for (int i = 1; i < iNumOfEllipses; ++i)
	{
		Ellipse& e1 = ellipses[i]; // チェック対象

		int sz_clusters = int(clusters.size());

		float ba_e1 = e1._b / e1._a;
		float Decc1 = e1._b / e1._a;

		bool bFoundCluster = false;
		for (int j = 0; j < sz_clusters; ++j)
		{
			Ellipse& e2 = clusters[j]; // 比較対象

			float ba_e2 = e2._b / e2._a;
			float th_Dc = std::min(e1._b, e2._b) * th_Dc_ratio;
			th_Dc *= th_Dc;

			float Dc = ((e1._xc - e2._xc) * (e1._xc - e2._xc) + (e1._yc - e2._yc) * (e1._yc - e2._yc));
			if (Dc > th_Dc)
			{
				continue;
			}

			float Da = abs(e1._a - e2._a) / std::max(e1._a, e2._a);
			if (Da > th_Da)
			{
				continue;
			}

			float Db = abs(e1._b - e2._b) / std::min(e1._b, e2._b);
			if (Db > th_Db)
			{
				continue;
			}

			float Dr = GetMinAnglePI(e1._rad, e2._rad) / float(CV_PI);
			if ((Dr > th_Dr) && (ba_e1 < th_Dr_circle) && (ba_e2 < th_Dr_circle))
			{
				continue;
			}

			bFoundCluster = true;

			break;
		}

		if (!bFoundCluster)
		{			
			clusters.push_back(e1);
		}
	}

	clusters.swap(ellipses);
};

// 入力画像から楕円リスト作成までの処理フロー
void CNEllipseDetector::Detect(cv::Mat1b& I, std::vector<Ellipse>& ellipses)
{
	countsOfFindEllipse = 0;
	countsOfGetFastCenter = 0;
	Tic(1); //prepare data structure

	_szImg = I.size(); // 入力画像サイズ取得

	// エッジピクセルを格納するための配列を用意
	cv::Mat1b DP = cv::Mat1b::zeros(_szImg);
	cv::Mat1b DN = cv::Mat1b::zeros(_szImg);

	// アキュムレータのサイズ決定
	ACC_N_SIZE = 101;
	ACC_R_SIZE = 180;
	ACC_A_SIZE = std::max(_szImg.height, _szImg.width);

	// アキュムレータのメモリ確保
	accN = new int[ACC_N_SIZE];
	accR = new int[ACC_R_SIZE];
	accA = new int[ACC_A_SIZE];

	// 円弧分類リストを用意
	VVP points_1, points_2, points_3, points_4;

	// キャッシュ用意
	std::unordered_map<uint, EllipseData> centers;

	Toc(1); //prepare data structure

	// 入力画像Iのエッジ検出、勾配計算によるDPとDNの分類
	PreProcessing(I, DP, DN);

	// canny法によるエッジ画像保存
	cv::Mat1b canny	= DP + DN;	
	cv::imwrite("result/canny.jpg", canny);

	Tac(1); //preprocessing

	// 途中経過の楕円を描画するための用意
	cv::Mat grayImage = I.clone();
	cv::Mat3b colorImage;
	cv::cvtColor(grayImage, colorImage, cv::COLOR_GRAY2BGR);

	// エッジの凸方向による分類
	DetectEdges13(DP, points_1, points_3);
	DetectEdges24(DN, points_2, points_4);

	// ノイズ・直線除去後のエッジ画像保存
	cv::Mat picture(_szImg, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat picture_white(_szImg, CV_8UC3, cv::Scalar(255, 255, 255));
	std::vector<cv::Mat> imgs(4);
	cv::Mat picture1 = picture.clone();
	cv::Mat picture2 = picture.clone();
	cv::Mat picture3 = picture.clone();
	cv::Mat picture4 = picture.clone();
	showEdge(points_1, picture1);
	showEdge(points_2, picture2);
	showEdge(points_3, picture3);
	showEdge(points_4, picture4);
	
	cv::Mat picture5 = picture.clone(); // エッジ1色
	showEdge(points_1, picture5);
	showEdge(points_2, picture5);
	showEdge(points_3, picture5);
	showEdge(points_4, picture5);

	cv::Mat picture6 = picture_white.clone(); // エッジ2色
	showEdge_r(points_1, picture6);
	showEdge_b(points_2, picture6);
	showEdge_r(points_3, picture6);
	showEdge_b(points_4, picture6);

	imgs[0] = picture2;
	imgs[1] = picture1;
	imgs[2] = picture3;
	imgs[3] = picture4; 

	cv::imwrite("result/curveture.jpg", picture5);
	cv::imwrite("result/curveture_4.jpg", picture6);

	Toc(1); //preprocessing

	Tic(2); //grouping

	// トリプレットにより楕円パラメータ確定
	
	Triplets124(points_1, points_2, points_4, centers, ellipses);
	Triplets231(points_2, points_3, points_1, centers, ellipses);
	Triplets342(points_3, points_4, points_2, centers, ellipses);
	Triplets413(points_4, points_1, points_3, centers, ellipses);
	

	// ダブレットにより楕円パラメータ確定
	
	Doublets12(points_1, points_2, centers, ellipses);
	Doublets23(points_2, points_3, centers, ellipses);
	Doublets34(points_3, points_4, centers, ellipses);
	Doublets41(points_4, points_1, centers, ellipses);
	Doublets13(points_1, points_3, centers, ellipses);
	Doublets24(points_2, points_4, centers, ellipses);
	

	// Doublets後の楕円描画
	int sz_Doublets = ellipses.size();
	std::cout << "sz_Doublets: " << sz_Doublets << std::endl;
	cv::Mat3b debugImage1 = colorImage.clone();
	DrawDetectedEllipses(debugImage1, ellipses, 0, 2);
	cv::imwrite("result/debug1.jpg", debugImage1);

	Toc(2); //grouping	

	_times[2] -= (_times[3] + _times[4]); // time estimation, validation inside

	Tac(4); //validation

	// 楕円をスコア順に並べ替え
	sort(ellipses.begin(), ellipses.end());

	Toc(4); //validation

	// Free accumulator memory
	delete[] accN;
	delete[] accR;
	delete[] accA;

	Tic(5);

	// 自作、極端な楕円削除
	//DeleteEllipses(ellipses);

	// DeleteEllipses後の楕円描画
	int sz_Delete = ellipses.size();
	std::cout << "sz_Delete: " << sz_Delete << std::endl;
	cv::Mat3b debugImage2 = colorImage.clone();
	DrawDetectedEllipses(debugImage2, ellipses, 0, 2);
	cv::imwrite("result/debug2.jpg", debugImage2);
	

	// 自作、重なる楕円削除
	OverlapEllipses(_szImg, ellipses);

	// Cluster detections
	//ClusterEllipses(ellipses);

	// OverlapEllipses後の楕円描画（最終結果と等しいから意味ない）
	/*
	int sz_Overlap = ellipses.size();
	std::cout << "sz_Overlap: " << sz_Overlap << std::endl;
	cv::Mat3b debugImage3 = colorImage.clone();
	DrawDetectedEllipses(debugImage3, ellipses, 0, 2);
	cv::imwrite("result/debug3.jpg", debugImage3);
	*/

	Toc(5);

	CraterMatching(ellipses);
};

// 楕円を描画
void CNEllipseDetector::DrawDetectedEllipses(cv::Mat3b& output, std::vector<Ellipse>& ellipses, int iTopN, int thickness)
{
	int sz_ell = int(ellipses.size());
	int n = (iTopN == 0) ? sz_ell : std::min(iTopN, sz_ell);
	for (int i = 0; i < n; ++i)
	{
		Ellipse& e = ellipses[n - i - 1];
		int g = cvRound(e._score * 255.f);
		cv::Scalar color(0, g, 0);
		cv::ellipse(output, cv::Point(cvRound(e._xc), cvRound(e._yc)), cv::Size(cvRound(e._a), cvRound(e._b)), e._rad * 180.0 / CV_PI, 0.0, 360.0, color, thickness);
	}
}
