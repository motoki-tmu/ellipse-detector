#include "stdafx.h"
#include "tools.h"
#include "CNEllipseDetector.h"
#include "LSM.h"
#include "PnP.h"

// データの読み込み（模擬月面データ）
DataSet DataLoader()
{
	// arrayにmodel_info.csvを格納
	std::vector<std::vector<double>> array(NUM_CRATERS, std::vector<double>(4, 0.0));
	std::ifstream stream("../model_info.csv"); // num, x, y, r
	std::string line;
	int row = 0;

	while (getline(stream, line))
	{
		replace(line.begin(), line.end(), ',', ' ');
		std::stringstream ss(line);
		int col = 0;
        double value;

		while (ss >> value)
        {
            if (row < array.size() && col < array[row].size())
            {
                array[row][col] = value;
            }
            col++;
        }
        row++;
	}

	// modelにarrayのx, y座標のみを格納
	std::vector<point> model;
	model.reserve(NUM_CRATERS);

	for (const auto& row : array)
	{
		if (row.size() > 2)
		{
			model.emplace_back(point{row[1], row[2]});
		}
	}

	// array2にline_combi_sorted.csvを格納
	std::vector<std::vector<double>> array2(NUM_COMBINATIONS, std::vector<double>(4, 0.0));
	std::ifstream stream2("../line_combi_sorted.csv"); // key, length, a_num, b_num
	std::string line2;
	int row2 = 0;

	while (getline(stream2, line2))
	{
		replace(line2.begin(), line2.end(), ',', ' ');
		std::stringstream ss2(line2);
		int col2 = 0;
        double value;

		while (ss2 >> value)
        {
            if (row2 < array2.size() && col2 < array2[row2].size())
            {
                array2[row2][col2] = value;
            }
            col2++;
        }
        row2++;
	}

	// line_segmentに
	std::vector<std::vector<double>> line_segment(3, std::vector<double>(NUM_COMBINATIONS, 0.0));
	for (int i = 0; i < NUM_COMBINATIONS; i++)
	{
		line_segment[0][i] = i;
		line_segment[1][i] = array2[i][2];
		line_segment[2][i] = array2[i][3];
	}

	DataSet Data;
	Data.model = model;
	Data.line_segment = line_segment;
	return Data;
}


// main!!!
int main(int argc, char** argv) // argc = argument count, argv = argument std::vector
{
	std::string filename = argv[1];
	std::vector<cv::TickMeter> times(3);
	CNEllipseDetector cned;

	cv::Mat3b image = cv::imread(filename);

	// 入力画像の回転
	cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
	//cv::rotate(image, image, cv::ROTATE_180);
	//cv::rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE);
	
	cv::Size sz = image.size();
	cv::Mat1b gray;
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

	/* クレータ検出で用いるパラメータ */

	cv::Size GaussKernelSize = cv::Size(5, 5); // 変える必要ない
	double   GaussSigma      = 1.0;	// 変える必要ない
	double 	 CannyHighTh     = 130;	// 60
	double   CannyLowTh      = 50;	// 20
	int		 MinEdgeLength   = 16;	// 8
	float	 MinRectLength   = 2.0;	// 1.0
	int      StringNum       = 16;	// 多分変えなくてよい
	float    PositionTh      = 1.0;	// 多分変えなくてよい
	float    V4SPTh          = 0.5;	// 多分変えなくてよい
	float	 MaxCenterDistance = sqrt(float(sz.width * sz.width + sz.height * sz.height)) * 0.04;
	float	 ContourTh       = 0.1;	// 多分変えなくてよい
	float	 MinPrecision    = 0.1;	// この閾値いらんかも
	float	 MinReliability  = 0.1; // この閾値いらんかも
	float 	 ScoreAlpha      = 0.0; // 0.4が最良（大阪学会時）

	cned.SetParameters	(	GaussKernelSize,	
							GaussSigma,		
							CannyHighTh,
							CannyLowTh,
							MinEdgeLength,
							MinRectLength,
							StringNum,
							PositionTh,
							V4SPTh,
							MaxCenterDistance,
							ContourTh,		
							MinPrecision,
							MinReliability,
							ScoreAlpha
						);

	std::vector<Ellipse> ellsCned;
	cv::Mat1b gray_clone = gray.clone();
	times[0].start();
	cned.Detect(gray_clone, ellsCned);
	times[0].stop();

	DataSet Data = DataLoader();
	std::vector<MatchPairSet> MatchPair;
	times[1].start();
	CraterMatching(ellsCned, Data.model, Data.line_segment, MatchPair);
	times[1].stop();

	times[2].start();
	PnP(MatchPair);
	times[2].stop();

	
	cv::Mat3b resultImage = image.clone();
	cned.DrawDetectedEllipses(resultImage, ellsCned);
	cv::imshow("Cned", resultImage);
	mkdir("result", 0777);
	cv::imwrite("result/result.png", resultImage);
	SaveEllipses("result/result.txt", ellsCned);
	

	double total_time = 0.0;
	for (int i = 0; i < 3; i++){total_time += times[i].getTimeMilli();}

	std::cout << "--------------------------------" << std::endl;
	std::cout << "Execution Time: " << std::endl;
	std::cout << "Crater Detection: \t" << times[0].getTimeMilli() << std::endl;
	std::cout << "Crater Matching: \t" << times[1].getTimeMilli() << std::endl;
	std::cout << "Pose Estimation: \t" << times[2].getTimeMilli() << std::endl;
	std::cout << "--------------------------------" << std::endl;
	std::cout << "Total: \t" << total_time << std::endl;
	std::cout << "--------------------------------" << std::endl;
	
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;	   
}

/******************************************************
リアルタイム処理

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
*******************************************************/