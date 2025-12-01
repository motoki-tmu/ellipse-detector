// 幾何計算、画像表示、ファイル操作、楕円検出の評価等

#include "tools.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <dirent.h>

// 2つの直線の交点を計算（cv::Point2f：float型のx,y座標を持つ2次元点）
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

// 2つの2次元点を行列に変換（自作関数、OpenCVには無い）
void point2Mat(cv::Point2f p1, cv::Point2f p2, float mat[2][2])
{
	mat[0][0] = p1.x;
	mat[0][1] = p1.y;
	mat[1][0] = p2.x;
	mat[1][1] = p2.y;
}

// 6点から不変量を計算（論文に記載されていない？）
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

// 複数の画像を一つの画面に表示
void MultiImage_OneWin(const std::string& MultiShow_WinName, const vector<cv::Mat>& SrcImg_V, cv::Size SubPlot, cv::Size ImgMax_Size)  
{  
	cv::Mat Disp_Img; // 出力画像を定義
	cv::Size Img_OrigSize = cv::Size(SrcImg_V[0].cols, SrcImg_V[0].rows); // 元画像のサイズを取得
	float WH_Ratio_Orig = Img_OrigSize.width/(float)Img_OrigSize.height; // 元画像の横縦比を計算
	cv::Size ImgDisp_Size = cv::Size(100, 100); // 出力画像のサイズを定義
	if(Img_OrigSize.width > ImgMax_Size.width)  
		ImgDisp_Size = cv::Size(ImgMax_Size.width, (int)(ImgMax_Size.width/WH_Ratio_Orig));  
	else if(Img_OrigSize.height > ImgMax_Size.height)  
		ImgDisp_Size = cv::Size((int)(ImgMax_Size.height*WH_Ratio_Orig), ImgMax_Size.height);  
	else  
		ImgDisp_Size = cv::Size(Img_OrigSize.width, Img_OrigSize.height);

	// 表示枚数の確認
	int Img_Num = (int)SrcImg_V.size();  
	if(Img_Num > SubPlot.width * SubPlot.height)
	{  
		cout<<"Your SubPlot Setting is too small !"<<endl;  
		exit(0);  
	}

	// 余白設定
	cv::Size DispBlank_Edge = cv::Size(80, 60);
	cv::Size DispBlank_Gap  = cv::Size(15, 15);

	// Create the display image
	Disp_Img.create(cv::Size(ImgDisp_Size.width * SubPlot.width + DispBlank_Edge.width + (SubPlot.width - 1) * DispBlank_Gap.width,
		ImgDisp_Size.height * SubPlot.height + DispBlank_Edge.height + (SubPlot.height - 1) * DispBlank_Gap.height), CV_8UC3);
	Disp_Img.setTo(0);
	
	int EdgeBlank_X = (Disp_Img.cols - (ImgDisp_Size.width * SubPlot.width + (SubPlot.width - 1) * DispBlank_Gap.width)) / 2;
	int EdgeBlank_Y = (Disp_Img.rows - (ImgDisp_Size.height * SubPlot.height + (SubPlot.height - 1) * DispBlank_Gap.height)) / 2;
	cv::Point LT_BasePos = cv::Point(EdgeBlank_X, EdgeBlank_Y);  
	cv::Point LT_Pos = LT_BasePos;  

	// Display all images
	for (int i=0; i < Img_Num; i++)  
	{  
		// Obtain the left top position  
		if ((i%SubPlot.width == 0) && (LT_Pos.x != LT_BasePos.x))  
		{  
			LT_Pos.x = LT_BasePos.x;  
			LT_Pos.y += (DispBlank_Gap.height + ImgDisp_Size.height);  
		}  
		// Writing each to Window's Image  
		cv::Mat imgROI = Disp_Img(cv::Rect(LT_Pos.x, LT_Pos.y, ImgDisp_Size.width, ImgDisp_Size.height));  
		cv::resize(SrcImg_V[i], imgROI, cv::Size(ImgDisp_Size.width, ImgDisp_Size.height));  

		LT_Pos.x += (DispBlank_Gap.width + ImgDisp_Size.width);  
	}  

	//Get the screen size of computer  #include<windows.h> #include <WinUser.h>
	int Scree_W = 1366;//GetSystemMetrics(SM_CXSCREEN);  
	int Scree_H = 768;//GetSystemMetrics(SM_CYSCREEN);  
	//cout<<Scree_W<<"\t"<<Scree_H<<endl;
	cv::namedWindow(MultiShow_WinName.c_str(), cv::WINDOW_NORMAL);
	cv::moveWindow(MultiShow_WinName.c_str(), (Scree_W - Disp_Img.cols) / 2, (Scree_H - Disp_Img.rows) / 2);//Centralize the window
	cv::imshow(MultiShow_WinName, Disp_Img);
	cv::waitKey(0);
	cv::destroyWindow(MultiShow_WinName); 
}  

/* ※未使用
void PyrDown(string picName)
{
	cv::Mat img1 = cv::imread(picName);
	cv::Mat img2;
	cv::Size sz;

	pyrUp(img1, img2, sz, cv::BORDER_DEFAULT);
	pyrUp(img2, img2, sz, cv::BORDER_DEFAULT);
	cv::namedWindow("WindowOrg");
	cv::namedWindow("WindowNew");
	cv::imshow("WindowOrg", img1);
	cv::imshow("WindowNew", img2);

	cv::waitKey(10000);
}
*/

/* ※未使用
cv::Mat matResize(cv::Mat src, double scale)
{
	cv::Mat img2;
	bool showtimeandpic = false;
	if(!showtimeandpic)
	{
		cv::Size dsize = cv::Size(int(src.cols * scale), int(src.rows * scale));
		img2 = cv::Mat(dsize, CV_32S);
		cv::resize(src, img2, dsize, cv::INTER_CUBIC);
	}
	else
	{
		clock_t start_time=clock();
		{
			cv::Size dsize = cv::Size(int(src.cols * scale), int(src.rows * scale));
			img2 = cv::Mat(dsize, CV_32S);
			cv::resize(src, img2, dsize, cv::INTER_CUBIC);
		}
		clock_t end_time=clock();
		cout << "Running time is: " << static_cast<double>(end_time-start_time) / CLOCKS_PER_SEC * 1000 << "ms" << endl;

		cv::namedWindow("WindowOrg", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("WindowNew", cv::WINDOW_AUTOSIZE);
		cv::imshow("WindowOrg", src);
		cv::imshow("WindowNew", img2);

		cv::waitKey(1000);
	}
	return img2;
}
*/

// エッジ描画
void showEdge(vector<vector<cv::Point>> points_, cv::Mat& picture)
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
		vector<cv::Point> Edge = points_.at(iEdge);
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

void showEdge1(vector<vector<cv::Point>> points_, cv::Mat& picture)
{
	srand( (unsigned)time( NULL )); int radius = 1; cv::Point center;
	int sEdge = points_.size(); cv::Point prev_point; cv::Point current_point;
	for (int iEdge = 0; iEdge < sEdge; iEdge++)
	{
		cv::Scalar color = cv::Scalar(0, 0, 255); // 赤
		vector<cv::Point> Edge = points_.at(iEdge);
		int sPoints = Edge.size();
		for (int iPoint = 0; iPoint < sPoints - 1; iPoint++)
		{
			center = Edge.at(iPoint); prev_point = Edge.at(iPoint); current_point = Edge.at(iPoint + 1);
			cv::line(picture, prev_point, current_point, color, 1, cv::LINE_AA);
		}
	}
}
void showEdge2(vector<vector<cv::Point>> points_, cv::Mat& picture)
{
	srand( (unsigned)time( NULL )); int radius = 1; cv::Point center;
	int sEdge = points_.size(); cv::Point prev_point; cv::Point current_point;
	for (int iEdge = 0; iEdge < sEdge; iEdge++)
	{
		cv::Scalar color = cv::Scalar(255, 0, 0); // 青
		vector<cv::Point> Edge = points_.at(iEdge);
		int sPoints = Edge.size();
		for (int iPoint = 0; iPoint < sPoints - 1; iPoint++)
		{
			center = Edge.at(iPoint); prev_point = Edge.at(iPoint); current_point = Edge.at(iPoint + 1);
			cv::line(picture, prev_point, current_point, color, 1, cv::LINE_AA);
		}
	}
}
void showEdge3(vector<vector<cv::Point>> points_, cv::Mat& picture)
{
	srand( (unsigned)time( NULL )); int radius = 1; cv::Point center;
	int sEdge = points_.size(); cv::Point prev_point; cv::Point current_point;
	for (int iEdge = 0; iEdge < sEdge; iEdge++)
	{
		cv::Scalar color = cv::Scalar(0, 255, 255); // 黄色
		vector<cv::Point> Edge = points_.at(iEdge);
		int sPoints = Edge.size();
		for (int iPoint = 0; iPoint < sPoints - 1; iPoint++)
		{
			center = Edge.at(iPoint); prev_point = Edge.at(iPoint); current_point = Edge.at(iPoint + 1);
			cv::line(picture, prev_point, current_point, color, 1, cv::LINE_AA);
		}
	}
}
void showEdge4(vector<vector<cv::Point>> points_, cv::Mat& picture)
{
	srand( (unsigned)time( NULL )); int radius = 1; cv::Point center;
	int sEdge = points_.size(); cv::Point prev_point; cv::Point current_point;
	for (int iEdge = 0; iEdge < sEdge; iEdge++)
	{
		cv::Scalar color = cv::Scalar(0, 255, 0); // 緑
		vector<cv::Point> Edge = points_.at(iEdge);
		int sPoints = Edge.size();
		for (int iPoint = 0; iPoint < sPoints - 1; iPoint++)
		{
			center = Edge.at(iPoint); prev_point = Edge.at(iPoint); current_point = Edge.at(iPoint + 1);
			cv::line(picture, prev_point, current_point, color, 1, cv::LINE_AA);
		}
	}
}

// file operation
int writeFile(string fileName_cpp, vector<string> vsContent)
{
	string line = "";
	vector<string> data;
	vector<string> data_split;
	ofstream out(fileName_cpp);
	if (!out)
	{
		cout << "Failed to open file for writing" << endl;
		return -1;
	}
	for (vector<string>::iterator i = vsContent.begin(); i < vsContent.end(); i++)
	{
		out << *i << endl;
	}
	out.close();
	return 1;
}

/* ※未使用
int readFile(string fileName_cpp)
{
	string line = "";
	vector<string> data;
	ifstream in(fileName_cpp);
	if (!in)
	{
		cout << "Failed to open file for reading" << endl;
		return -1;
	}
	while (getline(in, line))
	{
		data.push_back(line);
	}
	in.close();
	for (unsigned int i = 0; i < data.size(); i++)
	{
		cout << data.at(i) << endl;
	}
	return 0;
}
*/

/* ※未使用
int readFileByChar(string fileName_split)
{
	string line = "";
	vector<string> data;
	vector<string> data_split;
	ifstream in_split(fileName_split);
	if (!in_split)
	{
		cout << "Failed to open file for reading" << endl;
		return -1;
	}
	while (getline(in_split, line))
	{
		data_split.push_back(line);
	}
	in_split.close();

	for (unsigned int i = 0; i < data_split.size(); i++)
	{
		cout << "--------------------" << endl;
		for (unsigned int j = 0; j < getStr(data_split.at(i)).size(); j++)
		{
			cout << getStr(data_split.at(i)).at(j) << endl;
		}
	}
	return 0;
}
*/

void Trim(string &str)
{
	int s = str.find_first_not_of(" \t\n");
	int e = str.find_last_not_of(" \t\n");
	str = str.substr(s, e - s + 1);
}

vector<string> getStr(string str)
{
	int j = 0;
	string a[100];
	vector<string> v_a;
	//Split()
	for (unsigned int i = 0; i < str.size(); i++)
	{
		if ((str[i] != ',') && str[i] != '\0')
		{
			a[j] += str[i];
		}
		else j++;
	}

	for (int k = 0; k < j + 1; k++)
	{
		v_a.push_back(a[k]);
	}
	return v_a;
}

// ディレクトリ内のファイル一覧を取得
void listDir(string real_dir, vector<string>& files, bool r) 
{
	DIR *pDir;
	struct dirent *ent;
	string childpath;
	string absolutepath;
	pDir = opendir(real_dir.c_str());
	while ((ent = readdir(pDir)) != NULL)
	{
		if (ent->d_type & DT_DIR)
		{
			if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
			{
				continue;
			}
			if (r)
			{
				childpath = real_dir + ent->d_name + "/";
				listDir(childpath, files);
			}
		}
		else
		{
			absolutepath = real_dir + ent->d_name;
			files.push_back(ent->d_name);
		}
	}
	sort(files.begin(),files.end());
}

// 検出した楕円のパラメータを保存
void SaveEllipses(const string& fileName, const vector<Ellipse>& ellipses)
{
	unsigned n = ellipses.size();
	vector<string> resultString;
	stringstream resultsitem;

	resultsitem << n ;
	resultString.push_back(resultsitem.str());

	for (unsigned i = 0; i < n; ++i)
	{
		const Ellipse& e = ellipses[i];
		resultsitem.str("");
		resultsitem << e._xc << "\t" << e._yc << "\t" 
			<< e._a << "\t" << e._b << "\t" 
			<< e._rad << "\t" << e._score;
		resultString.push_back(resultsitem.str());
	}
	writeFile(fileName, resultString);
	for (int i = 0; i < resultString.size(); i++)
	{
		cout << resultString[i] << endl;
	}
}

// Should be checked
void SaveEllipses(const string& workingDir, const string& imgName, const vector<Ellipse>& ellipses /*, const vector<double>& times*/) 
{
	string path(workingDir + "/" + imgName + ".txt");
	ofstream out(path, ofstream::out | ofstream::trunc);
	if (!out.good())
	{
		cout << "Error saving: " << path << endl;
		return;
	}
	unsigned n = ellipses.size();

	out << n << "\n";

	for (unsigned i = 0; i<n; ++i)
	{
		const Ellipse& e = ellipses[i];
		out << e._xc << "\t" << e._yc << "\t" << e._a << "\t" << e._b << "\t" << e._rad << "\t" << e._score << "\n";
	}
	out.close();
}

// Ground Truthの読み込み
void LoadGT(vector<Ellipse>& gt, const string& sGtFileName, bool bIsAngleInRadians)
{
	ifstream in(sGtFileName);
	if (!in.good())
	{
		cout << "Error opening: " << sGtFileName << endl;
		return;
	}

	unsigned n;
	in >> n;

	gt.clear();
	gt.reserve(n);

	while (in.good() && n--)
	{
		Ellipse e;
		in >> e._xc >> e._yc >> e._a >> e._b >> e._rad;

		if (!bIsAngleInRadians)
		{
			e._rad = float(e._rad * CV_PI / 180.0);
		}

		if (e._a < e._b)
		{
			float temp = e._a;
			e._a = e._b;
			e._b = temp;

			e._rad = e._rad + float(0.5 * CV_PI);
		}

		e._rad = fmod(float(e._rad + 2.f * CV_PI), float(CV_PI));
		e._score = 1.f;
		gt.push_back(e);
	}
	in.close();
}

// Should be checked !!!!!
// 評価スコアの計算
float Evaluate(const vector<Ellipse>& ellGT, const vector<Ellipse>& ellTest, const float th_score, const cv::Mat3b& img)
{
	float threshold_overlap = 0.8f;
	//float threshold = 0.95f;

	unsigned sz_gt = ellGT.size();
	unsigned size_test = ellTest.size();

	unsigned sz_test = unsigned(min(1000, int(size_test)));

	vector<cv::Mat1b> gts(sz_gt);
	vector<cv::Mat1b> tests(sz_test);

	for (unsigned i = 0; i<sz_gt; ++i)
	{
		const Ellipse& e = ellGT[i];

		cv::Mat1b tmp(img.rows, img.cols, uchar(0));
		cv::ellipse(tmp, cv::Point((int)e._xc, (int)e._yc), cv::Size((int)e._a, (int)e._b), e._rad * 180.0 / CV_PI, 0.0, 360.0, cv::Scalar(255), -1);
		gts[i] = tmp;
	}
	for (unsigned i = 0; i<sz_test; ++i)
	{
		const Ellipse& e = ellTest[i];

		cv::Mat1b tmp(img.rows, img.cols, uchar(0));
		cv::ellipse(tmp, cv::Point((int)e._xc, (int)e._yc), cv::Size((int)e._a, (int)e._b), e._rad * 180.0 / CV_PI, 0.0, 360.0, cv::Scalar(255), -1);
		tests[i] = tmp;
	}

	cv::Mat1b overlap(sz_gt, sz_test, uchar(0));
	for (int r = 0; r < overlap.rows; ++r)
	{
		for (int c = 0; c < overlap.cols; ++c)
		{
			overlap(r, c) = TestOverlap(gts[r], tests[c], threshold_overlap) ? uchar(255) : uchar(0);
		}
	}

	int counter = 0;

	vector<bool> vec_gt(sz_gt, false);

	for (unsigned int i = 0; i < sz_test; ++i)
	{
		for (unsigned int j = 0; j < sz_gt; ++j)
		{
			if (vec_gt[j]) { continue; }

			bool bTest = overlap(j, i) != 0;

			if (bTest)
			{
				vec_gt[j] = true;
				break;
			}
		}
	}

	int tp = Count(vec_gt);
	int fn = int(sz_gt) - tp;
	int fp = size_test - tp; // !!!!

	float pr(0.f);
	float re(0.f);
	float fmeasure(0.f);

	if (tp == 0)
	{
		if (fp == 0)
		{
			pr = 1.f;
			re = 0.f;
			fmeasure = (2.f * pr * re) / (pr + re);
		}
		else
		{
			pr = 0.f;
			re = 0.f;
			fmeasure = 0.f;
		}
	}
	else
	{
		pr = float(tp) / float(tp + fp);
		re = float(tp) / float(tp + fn);
		fmeasure = (2.f * pr * re) / (pr + re);
	}

	return fmeasure;
}

// 楕円の重なりを計算（Jaccard係数、閾値はth）
bool TestOverlap(const cv::Mat1b& gt, const cv::Mat1b& test, float th)
{
	float fAND = float(cv::countNonZero(gt & test));
	float fOR = float(cv::countNonZero(gt | test));
	float fsim = fAND / fOR;

	return (fsim >= th);
}

int Count(const vector<bool> v)
{
	int counter = 0;
	for (unsigned i = 0; i < v.size(); ++i)
	{
		if (v[i]) { ++counter; }
	}
	return counter;
}

// ノイズ付加
void salt(cv::Mat& image, int n){
	for(int k = 0; k < n; k++){
		int i = rand()%image.cols;
		int j = rand()%image.rows;

		if(image.channels() == 1){
			image.at<uchar>(j,i) = 255;
		}else{
			image.at<cv::Vec3b>(j,i)[0] = 255;
			image.at<cv::Vec3b>(j,i)[1] = 255;
			image.at<cv::Vec3b>(j,i)[2] = 255;
		}
	}
}