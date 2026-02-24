// Main
// g++ *.cpp -o cned $(pkg-config --cflags --libs opencv4)

#include "stdafx.h"
#include "tools.h"
#include "CNEllipseDetector.h"

int MethodId = 1;

float	fThScoreScore = 0.7f;
float	fMinReliability	= 0.5f;
float	fTaoCenters = 0.05f;
int		ThLength = 4;
float	MinOrientedRectSide = 1.0f;
float   alpha = 0.5f;

float	scale = 1;

void OnVideo();
void SetParameter(std::map<char, std::string> Parameter);
void ReadParameter(int argc, char** argv);

void showTime(std::vector<double> times)
{
	std::cout << "--------------------------------" << std::endl;
	std::cout << "Execution Time: " << std::endl;
	std::cout << "Edge Detection: \t" << times[0] << std::endl;
	std::cout << "Pre processing: \t" << times[1] << std::endl;
	std::cout << "Grouping:       \t" << times[2] << std::endl;
	std::cout << "Estimation:     \t" << times[3] << std::endl;
	std::cout << "Validation:     \t" << times[4] << std::endl;
	std::cout << "Clustering:     \t" << times[5] << std::endl;
	std::cout << "--------------------------------" << std::endl;
	std::cout << "Total:	         \t" << accumulate(times.begin(), times.begin() + 6, 0.0) << std::endl;
	std::cout << "F-Measure:      \t" << times[6] << std::endl;
	std::cout << "--------------------------------" << std::endl;
	if (times.size() == 9)
	{
		std::cout << "countsOfFindEllipse \t" << times[7] << std::endl;
		std::cout << "countsOfGetFastCenter \t" << times[8] << std::endl;
	}
}

void OnVideo()
{
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) return;

	CNEllipseDetector cned;

	cv::Mat1b gray;
	while (true)
	{
		cv::Mat3b image;
		cap >> image;
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

		std::vector<Ellipse> ellipses;
	
		cned.Detect(gray, ellipses);
		cned.DrawDetectedEllipses(image, ellipses);
		cv::imshow("Output", image);

		if (cv::waitKey(10) >= 0) break;
	}
}

std::vector<double> OnImage(std::string filename, float fThScoreScore, float fMinReliability, float fTaoCenters, bool showpic)
{
	std::vector<double> times;
	CNEllipseDetector cned;

	cv::Mat3b image = cv::imread(filename);

	if (!image.data)
	{
		std::cout << filename << " not exist" << std::endl;
		return times;
	}
	cv::Size sz = image.size();

	cv::Mat1b gray;
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

	// Parameters Settings (Sect. 4.2)
	int		iThLength = ThLength;
	float	fThObb = MinOrientedRectSide;
	float	fThPos = 1.0f;

	int 	iNs = 16;
	float	fMaxCenterDistance = sqrt(float(sz.width * sz.width + sz.height * sz.height)) * fTaoCenters;

	float   scoreAlpha = alpha;

	//float	fThScoreScore = 0.5f;//0.8	
	//fTaoCenters = 0.05f;
	// Other constant parameters settings.
	// Gaussian filter parameters, in pre-processing
	cv::Size	szPreProcessingGaussKernelSize	= cv::Size(5, 5);
	double	dPreProcessingGaussSigma		= 1.0;

	float	fDistanceToEllipseContour		= 0.1f;	// (Sect. 3.3.1 - Validation)
	//float	fMinReliability					= 0.4f;	// Const parameters to discard bad ellipses 0.5

	// Initialize Detector with selected parameters
	cned.SetParameters	(	szPreProcessingGaussKernelSize,	
							dPreProcessingGaussSigma,		
							fThPos,
							fMaxCenterDistance,
							iThLength,
							fThObb,
							fDistanceToEllipseContour,		
							fThScoreScore,
							fMinReliability,		
							iNs,
							scoreAlpha
						);
	// Detect 
	std::vector<Ellipse> ellsCned;
	cv::Mat1b gray_clone = gray.clone();
	cned.Detect(gray_clone, ellsCned);
	times = cned.GetTimes();

	cv::Mat3b resultImage = image.clone();
	cned.DrawDetectedEllipses(resultImage, ellsCned);
	cv::imshow("Cned", resultImage);
	if(showpic)
	{
		mkdir("result", 0777);
		cv::imwrite("result/result.jpg", resultImage);
		SaveEllipses("result/result.txt", ellsCned);
	}

	double fmeasure = 0;
	times.push_back(fmeasure);
	times.push_back(cned.countsOfFindEllipse);
	times.push_back(cned.countsOfGetFastCenter);
	return times;
}

void SetParameter(std::map<char, std::string> Parameter)
{
	/**
	-N image Name
	-D DataSet Name
	-S The threshold of ellipse score
	-R The threshold of Reliability
	-C The threshold of CenterDis
	-M The method id
	-P
	*/
	std::map<char, std::string>::iterator it;
	for(it = Parameter.begin(); it != Parameter.end(); ++it)
	{
		switch(it->first)
		{
			case 'S': fThScoreScore = atof(it -> second.c_str()); break;
			case 'C': fTaoCenters = atof(it -> second.c_str()); break;
			case 'R': fMinReliability = atof(it -> second.c_str()); break;
			case 'M': MethodId = atof(it -> second.c_str()); break;
		}
	}
}
void ReadParameter(int argc,char** argv)
{
    if(argc <= 1 || argv[1][0] != '-')
	{
		if(argc > 1) std::cout << argv[1] << std::endl;
		return;
    }
    std::map<char, std::string> Parameter;
    int targc = argc;
    int cur = 1, pre = 1;
	std::string sPa;
	char cP = argv[pre][1];
    for(int i = 2; i < argc; i++)
	{
        std::string stmp = argv[i];
        if(0 == stmp.find("-"))
		{
            cur = i;
			cP = argv[pre][1];
			Trim(sPa);
            Parameter.insert(std::pair<char, std::string>(cP, sPa));
			pre = cur; sPa = "";
        }
		else
		{
			sPa = sPa + " " + argv[i];
		}
    }
	cP = argv[pre][1];
	Trim(sPa);
    Parameter.insert(std::pair<char, std::string>(cP, sPa));
	std::map<char, std::string>::iterator it;
    for(it = Parameter.begin(); it != Parameter.end(); ++it)
        std::cout << "key: " << it->first << " value: " << it->second << std::endl;
	SetParameter(Parameter);
}

// main!!!
int main(int argc, char** argv) // argc = argument count, argv = argument std::vector
{
	tCNC = 0.5f;            // 0.2
	fThScoreScore = 0.6f;	// 0.6
	fMinReliability	= 0.1f;	// 0.4
	fTaoCenters = 0.04f;    // 0.04 	
	ThLength = 8;          // 16
	MinOrientedRectSide = 1.0f; // 3.0
	alpha = 0.4f;

	if (argc == 2) // 引数が2つの場合
	{
		std::string filename = argv[1];
		if (NULL != strstr(filename.c_str(), ".jpg") || NULL != strstr(filename.c_str(), ".bmp")
			|| NULL != strstr(filename.c_str(), ".JPG") || NULL != strstr(filename.c_str(), ".png"))
		{
			std::cout << filename << std::endl;
			std::vector<double> results = OnImage(filename, fThScoreScore, fMinReliability, fTaoCenters, true);

			if (!results.empty()) // 結果が空でない場合 = 検出が正常に行われた場合
			{
				showTime(results);                         // 処理時間や評価を表示
				cv::waitKey(0);	cv::destroyWindow("Cned"); // ウィンドウを閉じる
				MethodId = 0;                              // switch文での処理をスキップ
				system("pause");                           // 処理終了前に一時停止
			}
		}
	}
	
	ReadParameter(argc, argv);
	switch (MethodId)
	{
	case 1: OnVideo();
		break;
	default: break;
	}

	system("pause");
	//main_normalDB(argc, argv);
	//OnVideo();
	return 0;	   
}

/*****************************************************************************************************************
処理フロー（Method 0）
./cned ../images/test.jpgで実行（cnedはコンパイル名）

int main()が実行、argc = 2, argv[0] = "./cned", argv[1] = "../image/test.jpg" 
パラメータ宣言（tCNC, fThScoreScore, fMinReliability, fTaoCenters, ThLength, MinOrientedRectSide）
faile name(=argv[1])に必要端子があることを確認 → std::cout filename

std::vector<double> results = OnImage(filename, fThScoreScore, fMinReliability, fTaoCenters, true);
{
	入力画像を取得（image）→ サイズ取得（sz）、グレースケール変換（gray）
	パラメータ宣言（iThLength, fThObb, fThPos, iNs, fMaxCenterDistance, ガウス, fDistanceToEllipseContour）
	cned.SetParameters
		10個のパラメータ宣言
	
	cned.Detect(入力：画像, 出力：Ellipse型の楕円情報)
	{
		PreProcessing(入力：画像, 出力：Mat型 DP, DN)
		{
			Canny_v2 or v3(入力：画像、出力：エッジ画像、勾配計算方法はマンハッタン(false))
			勾配でエッジをDPとDNに分類
		}

		DetectEdges13 or 24(入力：DP or DN、出力：VVP points_1~4)
		{
			Labeling(入力：画像、出力：VVP型 エッジを構成するピクセル座標群)
			ノイズ除去(ThLength以下の大きさのエッジを除去)
			直線除去(fMinOrientedRectSide比)
			凸判定でエッジを4象限にまで分類
		}

		Triplets(入力：3種類のpoints_n、出力：楕円パラメータ)
		{
			value4SixPointsによる同一円弧判定
			GetFastCenter(入力：2つの円弧、出力：EllipseData型 中心線や参照線の傾き、推定中心など)
			{
				円弧ペアの端点と中点を結ぶ線（参照線）と、それに平行な弦を計iNs個探索
				GetMedianSlope(入力：中心点リスト、出力：中心点と傾きの中央値)
				{
					i番目と i+halfSIze 番目の中点から傾きを得る
				}

				傾きと中心点から中心線が決定、中心線同士の交点を楕円の推定中心とする

			GetCenterCoordinates(入力：2つのEllipseData、出力：楕円の中心座標)
			{
				3つの円弧から中心線が4本得られる → 交点6つ
				さらにペア円弧同士の推定中心2つの中央値を計算し、計7つの座標の中央値を楕円の中心座標とする
			}

			FindEllipse(入力：楕円中心 円弧のピクセル座標リスト 円弧の傾きなど、出力：長短軸半径、楕円の傾き)
			{
				Paper. Eq[13~17]で算出、ellipsesに格納
				閾値チェック（fDistanceToEllipeContour, fMinScore, fMinReliability）
			}

		ClusterEllipse(入力：ellipses)
		{
			楕円パラメータの差分が小さすぎる楕円を同一とみなして削除
		}
	
	cned.DrawDetectedEllipse
	SaveEllipses
	showTime


＊＊閾値について＊＊

＊mainですぐ宣言＊ → 主にこいつらを調整？
tCNC = 0.2 : value4SxPointsで同一円弧か判定するための閾値、Tripletsで使用
fThScoreScore = 0.6 : 
fMinReliability = 0.4 : 楕円の円周長に対する円弧長さ、FindEllipse()で使用
fTaoCenters = 0.04 : 2つの楕円中心距離の閾値であるfMaxCenterDistanceの値決定に用いられる、式はmain.cppで定義、Paper Sec 4.2
ThLength = 16 : エッジの最小必要長さ（短いエッジを削除）、Labeling()で使用（cnedではiMinEdgeLength、commonではiMinLengthとして宣言）
MinOrientedRectSide = 3.0 : エッジを囲む最小の長方形の辺比の閾値（直線除外）、DetectEdges13()で使用

＊呼び出し関数先で定義＊ → 変える必要がなさそうな閾値は # でチェック
# szPreProcessingGaussKernelSize = cv::Size(5, 5) : ガウシアンフィルタのカーネルサイズ、PreProcessing()で使用
# dPreProcessingGaussSigma = 1.0 : ガウシアンフィルタのsigma、PreProcessing()で使用
apertureSize = 3 : Sobelフィルタのカーネルサイズ、Canny_v3で使用。妥当、5だとノイズに強いがエッジの位置精度低下
percent_of_pixels_not_edges = 0.9 : high_threshを値決定に用いる、画像全体のうち90%はエッジではないという仮定
threshold_ratio = 0.3 : low_threshを決定するための値、ヒストグラムにより決定したhighにratioを掛けてlowを決定
tTCN1 = 0.05 : 始点・中点・終点からなる三角形の面積から曲率判定、DetectEdges13()で使用（TCNは現在無効化）
tTCN2 = 0.05 : 全ての点の曲率を計算して曲率判定、DetectEdges13()で使用（TCNは現在無効化）
# fThPosition = 1.0 : 円弧が正しい位置関係にあるか判定、Triplets()で使用
# fMaxCenterDistance = 5.0 : 2つの楕円中心距離、だがmainでは数式で表現されているから不要？
iNs = 16 : 参照線と平行な弦の探索数、GetFastCenter()で使用
fDistanceToEllipseContours = 0.1 : ある点が楕円の輪郭上にあると見なす閾値、FindEllipse()で使用
fMinScore = 0.7 : 楕円候補としてみなすために必要な、輪郭上にある点の割合、FindEllipse()で使用
th_Da = 0.1 : 2つの楕円の長軸の差分、ClusterEllipse()で使用
th_Db = 0.1 : 2つの楕円の短軸の差分、ClusterEllipse()で使用
th_Dr = 0.1 : 2つの楕円の回転角の差分、ClusterEllipse()で使用
th_Dc_ratio = 0.1 : 2つの楕円の中心間距離の閾値決定に用いる、ClusterEllipse()で使用
th_Dr_circle = 0.9 : 2つの楕円の短長軸比の閾値、ClusterEllipse()で使用
******************************************************************************************************************/