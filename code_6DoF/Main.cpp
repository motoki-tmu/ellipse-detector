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
	std::vector<double> times;
	CNEllipseDetector cned;

	cv::Mat3b image = cv::imread(filename);
	cv::Size sz = image.size();
	cv::Mat1b gray;
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);


	/* クレータ検出で用いるパラメータ */

	cv::Size GaussKernelSize = cv::Size(5, 5); // 変える必要ない
	double   GaussSigma      = 1.0;	// 変える必要ない
	double 	 CannyHighTh     = 60;	// 60
	double   CannyLowTh      = 20;	// 20
	int		 MinEdgeLength   = 8;	// 8
	float	 MinRectLength   = 1.0;	// 1.0
	int      StringNum       = 16;	// 16
	float    PositionTh      = 1.0;	// 1.0
	float    V4SPTh          = 0.5;	// 0.5
	float	 MaxCenterDistance = sqrt(float(sz.width * sz.width + sz.height * sz.height)) * 0.04;
	float	 ContourTh       = 0.1;	// 0.1
	float	 MinPrecision    = 0.1;	// この閾値いらんかも
	float	 MinReliability  = 0.1; // この閾値いらんかも
	float 	 ScoreAlpha      = 0.4; // 0.4が最良（大阪学会時）

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
	cned.Detect(gray_clone, ellsCned);
	times = cned.GetTimes();

	DataSet Data = DataLoader();
	std::vector<MatchPairSet> MatchPair;
	CraterMatching(ellsCned, Data.model, Data.line_segment, MatchPair);

	PnP(MatchPair);

	cv::Mat3b resultImage = image.clone();
	cned.DrawDetectedEllipses(resultImage, ellsCned);
	cv::imshow("Cned", resultImage);
	mkdir("result", 0777);
	cv::imwrite("result/result.png", resultImage);
	SaveEllipses("result/result.txt", ellsCned);

	double fmeasure = 0;
	times.push_back(fmeasure);
	times.push_back(cned.countsOfFindEllipse);
	times.push_back(cned.countsOfGetFastCenter);

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