// 画像処理

#include "common.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// 改良したCanny関数（Sobelフィルタのxとy方向の勾配画像出力）※未使用
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

	cout << high_thresh << endl << low_thresh << endl;

    // 計算した閾値を使って標準Canny関数を呼び出す
    cv::Canny(src, edges, low_thresh, high_thresh, apertureSize, L2gradient);

	// 勾配の大きさがhigh以上 → エッジ、high未満low以上 → エッジと繋がっていればエッジ、low未満 → エッジではない
	// 勾配の大きさの計算方法 → L2gradientがTrueならユークリッド、Falseならマンハッタン（前者の方が正確で低速）
}

// セグメント化（自作関数）、segments（参照渡し）に独立したエッジを構成するピクセルの座標が格納される
void Labeling(cv::Mat1b& image, vector<vector<cv::Point>>& segments, int iMinLength)
{
    cv::Mat labels, stats, centroids; // bboxの座標や重心も計算してくれるが使わない？
    int num_labels = cv::connectedComponentsWithStats(image, labels, stats, centroids); // num_labelsはラベル（独立したエッジ）の総数

    vector<vector<cv::Point>> temp_segments(num_labels - 1); // 背景ラベル0を除いた総数の配列を用意

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

/* ※未使用
void LabelingRect(cv::Mat1b& image, VVP& segments, int iMinLength, vector<cv::Rect>& bboxes)
{
	#define _RG_STACK_SIZE 10000

	// Uso stack globali per velocizzare l'elaborazione (anche a scapito della memoria occupata)
	int stack2[_RG_STACK_SIZE];
	#define _RG_PUSH2(a) (stack2[sp2] = (a) , sp2++)
	#define _RG_POP2(a) (sp2-- , (a) = stack2[sp2])

	// Uso stack globali per velocizzare l'elaborazione (anche a scapito della memoria occupata)
	cv::Point stack3[_RG_STACK_SIZE];
	#define _RG_PUSH3(a) (stack3[sp3] = (a) , sp3++)
	#define _RG_POP3(a) (sp3-- , (a) = stack3[sp3])

	int i,w,h, iDim;
	int x,y;
	int x2,y2;	
	int sp2;
    int sp3;

	cv::Mat_<uchar> src = image.clone();
	w = src.cols;
	h = src.rows;
	iDim = w*h;

	cv::Point point;
	for (y=0; y<h; y++)
	{
		for (x=0; x<w; x++)
		{
			if ((src(y,x))!=0)   //punto non etichettato: seme trovato
			{
				// per ogni oggetto	
				sp2 = 0;
				i = x + y*w;
				_RG_PUSH2(i);

				// vuoto la lista dei punti
	    		sp3=0;
  		  		while (sp2>0) 
				{// rg tradizionale
		
					_RG_POP2(i);
					x2=i%w;
					y2=i/w;

					src(y2,x2) = 0;

					point.x=x2;
					point.y=y2;
					_RG_PUSH3(point);

					// Inserisco i nuovi punti nello stack solo se esistono
					// e sono punti da etichettare

					// 4 connessi
					// sx
					if (x2>0 &&   (src(y2, x2-1)!=0))
						_RG_PUSH2(i-1);
					// sotto
					if (y2>0 &&   (src(y2-1, x2)!=0))
						_RG_PUSH2(i-w);
					// sopra
					if (y2<h-1 &&   (src(y2+1, x2)!=0))
						_RG_PUSH2(i+w);
					// dx
					if (x2<w-1 &&   (src(y2, x2+1)!=0))
						_RG_PUSH2(i+1);

					// 8 connessi
					if (x2>0 && y2>0 &&   (src(y2-1,x2-1)!=0))
						_RG_PUSH2(i-w-1);
					if (x2>0 && y2<h-1 &&   (src(y2+1, x2-1)!=0))
						_RG_PUSH2(i+w-1);
					if (x2<w-1 && y2>0 &&   (src(y2-1, x2+1)!=0))
						_RG_PUSH2(i-w+1);
					if (x2<w-1 && y2<h-1 &&   (src(y2+1, x2+1)!=0))
						_RG_PUSH2(i+w+1);

				}

				if (sp3 >= iMinLength)
				{
					vector<cv::Point> component;

					int iMinx, iMaxx, iMiny,iMaxy;
					iMinx = iMaxx = stack3[0].x;
					iMiny = iMaxy = stack3[0].y;

					// etichetto il punto
					for (i=0; i<sp3; i++){
						point = stack3[i];
						// etichetto
						component.push_back(point);

						if (iMinx > point.x)  iMinx = point.x;
						if (iMiny > point.y)  iMiny = point.y;
						if (iMaxx < point.x)  iMaxx = point.x;
						if (iMaxy < point.y)  iMaxy = point.y;
					}

					bboxes.push_back(cv::Rect(cv::Point(iMinx, iMiny), cv::Point(iMaxx+1, iMaxy+1)));
					segments.push_back(component);
					
				}
			}
		}	
	}
}
*/


// Thinning 3*3 （opencv-contribをビルドしない場合）（使うか未定）
/**
 * @brief 画像の細線化をZhang-Suen法を用いて行う
 * @param imgMask   [入出力] 8ビット単一チャンネルの二値化画像 (CV_8UC1)。前景ピクセルが細線化される。
 * @param byF       前景を示す画素値 (デフォルト: 255)
 * @param byB       背景を示す画素値 (デフォルト: 0)
 */

/*
void Thinning(cv::Mat1b& imgMask, uchar byF, uchar byB)
{
    // 1. 処理用の二値画像を作成 (前景: 1, 背景: 0)
    cv::Mat thinningImage = cv::Mat::zeros(imgMask.size(), CV_8UC1);
    for (int y = 0; y < imgMask.rows; ++y) {
        for (int x = 0; x < imgMask.cols; ++x) {
            if (imgMask.at<uchar>(y, x) == byF) {
                thinningImage.at<uchar>(y, x) = 1;
            }
        }
    }

    // 削除対象のピクセルをマークするための行列
    cv::Mat marker = cv::Mat::zeros(imgMask.size(), CV_8UC1);

    bool isFinished = false;
    while (!isFinished) {
        isFinished = true; // 今回のループで何も削除されなければ、このフラグがtrueのままループを抜ける

        // === ステップ1: 下側と右側の輪郭ピクセルを削除 ===
        for (int y = 1; y < thinningImage.rows - 1; ++y) {
            for (int x = 1; x < thinningImage.cols - 1; ++x) {
                // (y, x)が前景ピクセルでなければスキップ
                if (thinningImage.at<uchar>(y, x) == 0) {
                    continue;
                }

                // 周囲8ピクセルの値を取得 (P2からP9)
                uchar p2 = thinningImage.at<uchar>(y - 1, x);
                uchar p3 = thinningImage.at<uchar>(y - 1, x + 1);
                uchar p4 = thinningImage.at<uchar>(y, x + 1);
                uchar p5 = thinningImage.at<uchar>(y + 1, x + 1);
                uchar p6 = thinningImage.at<uchar>(y + 1, x);
                uchar p7 = thinningImage.at<uchar>(y + 1, x - 1);
                uchar p8 = thinningImage.at<uchar>(y, x - 1);
                uchar p9 = thinningImage.at<uchar>(y - 1, x - 1);

                // 条件A: 0から1へのパターンの数が1であること (連結性を保つ)
                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                        (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                        (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);

                // 条件B: 前景の隣接ピクセルが2以上6以下であること (端点や孤立点を消さない)
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

                // ステップ1の固有条件
                int C = p2 * p4 * p6; // 北・東・南のいずれかが背景
                int D = p4 * p6 * p8; // 東・南・西のいずれかが背景

                if (A == 1 && (B >= 2 && B <= 6) && C == 0 && D == 0) {
                    marker.at<uchar>(y, x) = 1; // 削除対象としてマーク
                    isFinished = false; // ピクセルが削除されたので、処理を続行
                }
            }
        }

        // マークされたピクセルを実際に削除
        thinningImage &= ~marker;

        // === ステップ2: 上側と左側の輪郭ピクセルを削除 ===
        marker.setTo(0); // マーカーをリセット
        for (int y = 1; y < thinningImage.rows - 1; ++y) {
            for (int x = 1; x < thinningImage.cols - 1; ++x) {
                if (thinningImage.at<uchar>(y, x) == 0) {
                    continue;
                }

                uchar p2 = thinningImage.at<uchar>(y - 1, x);
                uchar p3 = thinningImage.at<uchar>(y - 1, x + 1);
                uchar p4 = thinningImage.at<uchar>(y, x + 1);
                uchar p5 = thinningImage.at<uchar>(y + 1, x + 1);
                uchar p6 = thinningImage.at<uchar>(y + 1, x);
                uchar p7 = thinningImage.at<uchar>(y + 1, x - 1);
                uchar p8 = thinningImage.at<uchar>(y, x - 1);
                uchar p9 = thinningImage.at<uchar>(y - 1, x - 1);

                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                        (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                        (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

                // ステップ2の固有条件 (CとDがステップ1と異なる)
                int C = p2 * p4 * p8; // 北・東・西のいずれかが背景
                int D = p2 * p6 * p8; // 北・南・西のいずれかが背景

                if (A == 1 && (B >= 2 && B <= 6) && C == 0 && D == 0) {
                    marker.at<uchar>(y, x) = 1;
                    isFinished = false;
                }
            }
        }
        
        // マークされたピクセルを実際に削除
        thinningImage &= ~marker;
    }

    // 2. 処理結果を元のMatに戻す
    for (int y = 0; y < imgMask.rows; ++y) {
        for (int x = 0; x < imgMask.cols; ++x) {
            if (thinningImage.at<uchar>(y, x) == 1) {
                imgMask.at<uchar>(y, x) = byF;
            } else {
                imgMask.at<uchar>(y, x) = byB;
            }
        }
    }
}
*/

bool SortBottomLeft2TopRight(const cv::Point& lhs, const cv::Point& rhs)
{
	if(lhs.x == rhs.x)
	{
		return lhs.y > rhs.y;
	}
	return lhs.x < rhs.x;
};

/* ※未使用
bool SortBottomLeft2TopRight2f(const cv::Point2f& lhs, const cv::Point2f& rhs)
{
	if(lhs.x == rhs.x)
	{
		return lhs.y > rhs.y;
	}
	return lhs.x < rhs.x;
};
*/

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
		swap(a, b);
	}

	float diff1 = b - a;
	float diff2 = pi - diff1;
	return min(diff1, diff2);
}

/* 元の自作関数
void cvCanny2(	const void* srcarr, void* dstarr,
				double low_thresh, double high_thresh,
				void* dxarr, void* dyarr,
                int aperture_size )
{
    //cv::Ptr<cv::Mat> dx, dy;
    cv::AutoBuffer<char> buffer;
    std::vector<uchar*> stack;
    uchar **stack_top = 0, **stack_bottom = 0;

    cv::Mat srcstub, *src = cv::GetMat( srcarr, &srcstub );
    cv::Mat dststub, *dst = cv::GetMat( dstarr, &dststub );

	cv::Mat dxstub, *dx = cv::GetMat( dxarr, &dxstub );
	cv::Mat dystub, *dy = cv::GetMat( dyarr, &dystub );


    cv::Size size;
    int flags = aperture_size;
    int low, high;
    int* mag_buf[3];
    uchar* map;
    ptrdiff_t mapstep;
    int maxsize;
    int i, j;
    cv::Mat mag_row;

    if( CV_MAT_TYPE( src->type ) != CV_8UC1 ||
        CV_MAT_TYPE( dst->type ) != CV_8UC1 ||
		CV_MAT_TYPE( dx->type  ) != CV_16SC1 ||
		CV_MAT_TYPE( dy->type  ) != CV_16SC1 )
        CV_Error( cv::Error::StsUnsupportedFormat, "" );

    if( !CV_ARE_SIZES_EQ( src, dst ))
        CV_Error( cv::Error::StsUnmatchedSizes, "" );

    if( low_thresh > high_thresh )
    {
        std::swap( low_thresh, high_thresh );
    }

    aperture_size &= INT_MAX;
    if( (aperture_size & 1) == 0 || aperture_size < 3 || aperture_size > 7 )
        CV_Error( cv::Error::StsBadFlag, "" );
	
	size.width = src->cols;
    size.height = src->rows;

    //size = cvGetMatSize( src );

    //dx = cvCreateMat( size.height, size.width, CV_16SC1 );
    //dy = cvCreateMat( size.height, size.width, CV_16SC1 );

	//aperture_size = -1; //SCHARR
    cv::Sobel( src, dx, 1, 0, aperture_size );
    cv::Sobel( src, dy, 0, 1, aperture_size );

	//Mat ddx(dx,true);
	//Mat ddy(dy,true);


    if( flags & CV_CANNY_L2_GRADIENT )
    {
        cv::32suf ul, uh;
        ul.f = (float)low_thresh;
        uh.f = (float)high_thresh;

        low = ul.i;
        high = uh.i;
    }
    else
    {
        low = std::floor( low_thresh );
        high =std::floor( high_thresh );
    }

    buffer.allocate( (size.width+2)*(size.height+2) + (size.width+2)*3*sizeof(int) );

    mag_buf[0] = (int*)(char*)buffer;
    mag_buf[1] = mag_buf[0] + size.width + 2;
    mag_buf[2] = mag_buf[1] + size.width + 2;
    map = (uchar*)(mag_buf[2] + size.width + 2);
    mapstep = size.width + 2;

    maxsize = MAX( 1 << 10, size.width*size.height/10 );
    stack.resize( maxsize );
    stack_top = stack_bottom = &stack[0];

    memset( mag_buf[0], 0, (size.width+2)*sizeof(int) );
    memset( map, 1, mapstep );
    memset( map + mapstep*(size.height + 1), 1, mapstep );

    // sector numbers
    // (Top-Left Origin)

    //    1   2   3
    //     *  *  *
    //      * * *
    //    0*******0
    //      * * *
    //     *  *  *
    //    3   2   1


    #define CANNY_PUSH(d)    *(d) = (uchar)2, *stack_top++ = (d)
    #define CANNY_POP(d)     (d) = *--stack_top

    mag_row = cv::Mat( 1, size.width, CV_32F );

    // calculate magnitude and angle of gradient, perform non-maxima supression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for( i = 0; i <= size.height; i++ )
    {
        int* _mag = mag_buf[(i > 0) + 1] + 1;
        float* _magf = (float*)_mag;
        const short* _dx = (short*)(dx->data.ptr + dx->step*i);
        const short* _dy = (short*)(dy->data.ptr + dy->step*i);
        uchar* _map;
        int x, y;
        ptrdiff_t magstep1, magstep2;
        int prev_flag = 0;

        if( i < size.height )
        {
            _mag[-1] = _mag[size.width] = 0;

            if( !(flags & CV_CANNY_L2_GRADIENT) )
                for( j = 0; j < size.width; j++ )
                    _mag[j] = abs(_dx[j]) + abs(_dy[j]);

            else
            {
                for( j = 0; j < size.width; j++ )
                {
                    x = _dx[j]; y = _dy[j];
                    _magf[j] = (float)std::sqrt((double)x*x + (double)y*y);
                }
            }
        }
        else
            memset( _mag-1, 0, (size.width + 2)*sizeof(int) );

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if( i == 0 )
            continue;

        _map = map + mapstep*i + 1;
        _map[-1] = _map[size.width] = 1;

        _mag = mag_buf[1] + 1; // take the central row
        _dx = (short*)(dx->data.ptr + dx->step*(i-1));
        _dy = (short*)(dy->data.ptr + dy->step*(i-1));

        magstep1 = mag_buf[2] - mag_buf[1];
        magstep2 = mag_buf[0] - mag_buf[1];

        if( (stack_top - stack_bottom) + size.width > maxsize )
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = MAX( maxsize * 3/2, maxsize + 8 );
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        for( j = 0; j < size.width; j++ )
        {
            #define CANNY_SHIFT 15
            #define TG22  (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5)

            x = _dx[j];
            y = _dy[j];
            int s = x ^ y;
            int m = _mag[j];

            x = abs(x);
            y = abs(y);
            if( m > low )
            {
                int tg22x = x * TG22;
                int tg67x = tg22x + ((x + x) << CANNY_SHIFT);

                y <<= CANNY_SHIFT;

                if( y < tg22x )
                {
                    if( m > _mag[j-1] && m >= _mag[j+1] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
                else if( y > tg67x )
                {
                    if( m > _mag[j+magstep2] && m >= _mag[j+magstep1] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
                else
                {
                    s = s < 0 ? -1 : 1;
                    if( m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
            }
            prev_flag = 0;
            _map[j] = (uchar)1;
        }

        // scroll the ring buffer
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }

    // now track the edges (hysteresis thresholding)
    while( stack_top > stack_bottom )
    {
        uchar* m;
        if( (stack_top - stack_bottom) + 8 > maxsize )
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = MAX( maxsize * 3/2, maxsize + 8 );
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        CANNY_POP(m);

        if( !m[-1] )
            CANNY_PUSH( m - 1 );
        if( !m[1] )
            CANNY_PUSH( m + 1 );
        if( !m[-mapstep-1] )
            CANNY_PUSH( m - mapstep - 1 );
        if( !m[-mapstep] )
            CANNY_PUSH( m - mapstep );
        if( !m[-mapstep+1] )
            CANNY_PUSH( m - mapstep + 1 );
        if( !m[mapstep-1] )
            CANNY_PUSH( m + mapstep - 1 );
        if( !m[mapstep] )
            CANNY_PUSH( m + mapstep );
        if( !m[mapstep+1] )
            CANNY_PUSH( m + mapstep + 1 );
    }

    // the final pass, form the final image
    for( i = 0; i < size.height; i++ )
    {
        const uchar* _map = map + mapstep*(i+1) + 1;
        uchar* _dst = dst->data.ptr + dst->step*i;

        for( j = 0; j < size.width; j++ )
		{
            _dst[j] = (uchar)-(_map[j] >> 1);
		}
	}
};

void Canny2(	cv::InputArray image, cv::OutputArray _edges,
				cv::OutputArray _sobel_x, cv::OutputArray _sobel_y,
                double threshold1, double threshold2,
                int apertureSize, bool L2gradient )
{
    cv::Mat src = image.getMat();
    _edges.create(src.size(), CV_8U);
	_sobel_x.create(src.size(), CV_16S);
	_sobel_y.create(src.size(), CV_16S);


    cv::Mat c_src = cv::Mat(src), c_dst = cv::Mat(_edges.getMat());
	cv::Mat c_dx = cv::Mat(_sobel_x.getMat());
	cv::Mat c_dy = cv::Mat(_sobel_y.getMat());


    cvCanny2(	&c_src, &c_dst, threshold1, threshold2,
				&c_dx, &c_dy,
				apertureSize + (L2gradient ? CV_CANNY_L2_GRADIENT : 0));
};
*/

/* 元の自作関数
void cvCanny3(	const void* srcarr, void* dstarr,
				void* dxarr, void* dyarr,
                int aperture_size )
{
    //cv::Ptr<cv::Mat> dx, dy;
    cv::AutoBuffer<char> buffer;
    std::vector<uchar*> stack;
    uchar **stack_top = 0, **stack_bottom = 0;

    cv::Mat srcstub, *src = cvGetMat( srcarr, &srcstub );//IplImage ��cvMat��ת��
    cv::Mat dststub, *dst = cvGetMat( dstarr, &dststub );

	cv::Mat dxstub, *dx = cvGetMat( dxarr, &dxstub );
	cv::Mat dystub, *dy = cvGetMat( dyarr, &dystub );


    cv::Size size;
    int flags = aperture_size;
    int low, high;
    int* mag_buf[3];
    uchar* map;
    ptrdiff_t mapstep;
    int maxsize;
    int i, j;
    cv::Mat mag_row;

    if( CV_MAT_TYPE( src->type ) != CV_8UC1 ||
        CV_MAT_TYPE( dst->type ) != CV_8UC1 ||
		CV_MAT_TYPE( dx->type  ) != CV_16SC1 ||
		CV_MAT_TYPE( dy->type  ) != CV_16SC1 )
        CV_Error( cv::Error::StsUnsupportedFormat, "" );

    if( !CV_ARE_SIZES_EQ( src, dst ))
        CV_Error( cv::Error::StsUnmatchedSizes, "" );
	
    aperture_size &= INT_MAX;
    if( (aperture_size & 1) == 0 || aperture_size < 3 || aperture_size > 7 )
        CV_Error( cv::Error::StsBadFlag, "" );


	size.width = src->cols;
    size.height = src->rows;
   // size = cvGetMatSize( src );
	

    //dx = cvCreateMat( size.height, size.width, CV_16SC1 );
    //dy = cvCreateMat( size.height, size.width, CV_16SC1 );

	//aperture_size = -1; //SCHARR
    cv::Sobel( src, dx, 1, 0, aperture_size );
    cv::Sobel( src, dy, 0, 1, aperture_size );

	//const Mat sobel_x(dx);	//Mat_<unsigned short>
	//const Mat sobel_y(dy);

	//% Calculate Magnitude of Gradient
    //magGrad = hypot(dx, dy);

	cv::Mat1f magGrad(size.height, size.width, 0.f);
	float maxGrad(0);
	float val(0);
	for(i=0; i<size.height; ++i)
	{
		float* _pmag = magGrad.ptr<float>(i);
		const short* _dx = (short*)(dx->data.ptr + dx->step*i);
        const short* _dy = (short*)(dy->data.ptr + dy->step*i);
		for(j=0; j<size.width; ++j)
		{
			val = float(abs(_dx[j]) + abs(_dy[j]));
			_pmag[j] = val;
			maxGrad = (val > maxGrad) ? val : maxGrad;
		}
	}
	
	//% Normalize for threshold selection
	//normalize(magGrad, magGrad, 0.0, 1.0, NORM_MINMAX);

	//% Determine Hysteresis Thresholds
	
	//set magic numbers
	const int NUM_BINS = 64;	
	const double percent_of_pixels_not_edges = 0.9;
	const double threshold_ratio = 0.3;

	//compute histogram
	int bin_size = std::floor(maxGrad / float(NUM_BINS) + 0.5f) + 1;
	if (bin_size < 1) bin_size = 1;
	int bins[NUM_BINS] = { 0 }; 
	for (i=0; i<size.height; ++i) 
	{
		float *_pmag = magGrad.ptr<float>(i);
		for(j=0; j<size.width; ++j)
		{
			int hgf = int(_pmag[j]);
			bins[int(_pmag[j]) / bin_size]++;
		}
	}	

	
	

	//% Select the thresholds
	float total(0.f);	
	float target = float(size.height * size.width * percent_of_pixels_not_edges);
	int low_thresh, high_thresh(0);
	
	while(total < target)
	{
		total+= bins[high_thresh];
		high_thresh++;
	}
	high_thresh *= bin_size;
	low_thresh = std::floor(threshold_ratio * float(high_thresh));
	
    if( flags & CV_CANNY_L2_GRADIENT )
    {
        cv::32suf ul, uh;
        ul.f = (float)low_thresh;
        uh.f = (float)high_thresh;

        low = ul.i;
        high = uh.i;
    }
    else
    {
        low = std::floor( low_thresh );
        high = std::floor( high_thresh );
    }

    
	buffer.allocate( (size.width+2)*(size.height+2) + (size.width+2)*3*sizeof(int) );
    mag_buf[0] = (int*)(char*)buffer;
    mag_buf[1] = mag_buf[0] + size.width + 2;
    mag_buf[2] = mag_buf[1] + size.width + 2;
    map = (uchar*)(mag_buf[2] + size.width + 2);
    mapstep = size.width + 2;

    maxsize = MAX( 1 << 10, size.width*size.height/10 );
    stack.resize( maxsize );
    stack_top = stack_bottom = &stack[0];

    memset( mag_buf[0], 0, (size.width+2)*sizeof(int) );
    memset( map, 1, mapstep );
    memset( map + mapstep*(size.height + 1), 1, mapstep );

    // sector numbers
    // (Top-Left Origin)

    //    1   2   3
    //     *  *  *
    //     * * *
    //    0*******0
    //      * * *
    //     *  *  *
    //    3   2   1


    #define CANNY_PUSH(d)    *(d) = (uchar)2, *stack_top++ = (d)
    #define CANNY_POP(d)     (d) = *--stack_top

    mag_row = cv::Mat( 1, size.width, CV_32F );

    // calculate magnitude and angle of gradient, perform non-maxima supression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for( i = 0; i <= size.height; i++ )
    {
        int* _mag = mag_buf[(i > 0) + 1] + 1;
        float* _magf = (float*)_mag;
        const short* _dx = (short*)(dx->data.ptr + dx->step*i);
        const short* _dy = (short*)(dy->data.ptr + dy->step*i);
        uchar* _map;
        int x, y;
        ptrdiff_t magstep1, magstep2;
        int prev_flag = 0;

        if( i < size.height )
        {
            _mag[-1] = _mag[size.width] = 0;

            if( !(flags & CV_CANNY_L2_GRADIENT) )
                for( j = 0; j < size.width; j++ )
                    _mag[j] = abs(_dx[j]) + abs(_dy[j]);

            else
            {
                for( j = 0; j < size.width; j++ )
                {
                    x = _dx[j]; y = _dy[j];
                    _magf[j] = (float)std::sqrt((double)x*x + (double)y*y);
                }
            }
        }
        else
            memset( _mag-1, 0, (size.width + 2)*sizeof(int) );

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if( i == 0 )
            continue;

        _map = map + mapstep*i + 1;
        _map[-1] = _map[size.width] = 1;

        _mag = mag_buf[1] + 1; // take the central row
        _dx = (short*)(dx->data.ptr + dx->step*(i-1));
        _dy = (short*)(dy->data.ptr + dy->step*(i-1));

        magstep1 = mag_buf[2] - mag_buf[1];
        magstep2 = mag_buf[0] - mag_buf[1];

        if( (stack_top - stack_bottom) + size.width > maxsize )
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = MAX( maxsize * 3/2, maxsize + 8 );
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        for( j = 0; j < size.width; j++ )
        {
            #define CANNY_SHIFT 15
            #define TG22  (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5)

            x = _dx[j];
            y = _dy[j];
            int s = x ^ y;
            int m = _mag[j];

            x = abs(x);
            y = abs(y);
            if( m > low )
            {
                int tg22x = x * TG22;
                int tg67x = tg22x + ((x + x) << CANNY_SHIFT);

                y <<= CANNY_SHIFT;

                if( y < tg22x )
                {
                    if( m > _mag[j-1] && m >= _mag[j+1] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
                else if( y > tg67x )
                {
                    if( m > _mag[j+magstep2] && m >= _mag[j+magstep1] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
                else
                {
                    s = s < 0 ? -1 : 1;
                    if( m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
            }
            prev_flag = 0;
            _map[j] = (uchar)1;
        }

        // scroll the ring buffer
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }

    // now track the edges (hysteresis thresholding)
    while( stack_top > stack_bottom )
    {
        uchar* m;
        if( (stack_top - stack_bottom) + 8 > maxsize )
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = MAX( maxsize * 3/2, maxsize + 8 );
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        CANNY_POP(m);

        if( !m[-1] )
            CANNY_PUSH( m - 1 );
        if( !m[1] )
            CANNY_PUSH( m + 1 );
        if( !m[-mapstep-1] )
            CANNY_PUSH( m - mapstep - 1 );
        if( !m[-mapstep] )
            CANNY_PUSH( m - mapstep );
        if( !m[-mapstep+1] )
            CANNY_PUSH( m - mapstep + 1 );
        if( !m[mapstep-1] )
            CANNY_PUSH( m + mapstep - 1 );
        if( !m[mapstep] )
            CANNY_PUSH( m + mapstep );
        if( !m[mapstep+1] )
            CANNY_PUSH( m + mapstep + 1 );
    }

    // the final pass, form the final image
    for( i = 0; i < size.height; i++ )
    {
        const uchar* _map = map + mapstep*(i+1) + 1;
        uchar* _dst = dst->data.ptr + dst->step*i;

        for( j = 0; j < size.width; j++ )
		{
            _dst[j] = (uchar)-(_map[j] >> 1);
		}
	}
};

void Canny3(	cv::InputArray image, cv::OutputArray _edges,
				cv::OutputArray _sobel_x, cv::OutputArray _sobel_y,
                int apertureSize, bool L2gradient )
{
    cv::Mat src = image.getMat();
    _edges.create(src.size(), CV_8U);
	_sobel_x.create(src.size(), CV_16S);
	_sobel_y.create(src.size(), CV_16S);


    cv::Mat c_src = cv::Mat(src), c_dst = cv::Mat(_edges.getMat());
	cv::Mat c_dx = cv::Mat(_sobel_x.getMat());
	cv::Mat c_dy = cv::Mat(_sobel_y.getMat());


    cvCanny3(	&c_src, &c_dst, 
				&c_dx, &c_dy,
				apertureSize + (L2gradient ? CV_CANNY_L2_GRADIENT : 0));
};
*/

/* セグメント化（元コード）
void Labeling(cv::Mat1b& image, vector<vector<cv::Point> >& segments, int iMinLength)
{
	#define RG_STACK_SIZE 2048

	// Uso stack globali per velocizzare l'elaborazione (anche a scapito della memoria occupata)
	int stack2[RG_STACK_SIZE];
	#define RG_PUSH2(a) (stack2[sp2] = (a) , sp2++)
	#define RG_POP2(a) (sp2-- , (a) = stack2[sp2])

	// Uso stack globali per velocizzare l'elaborazione (anche a scapito della memoria occupata)
	cv::Point stack3[RG_STACK_SIZE];
	#define RG_PUSH3(a) (stack3[sp3] = (a) , sp3++)
	#define RG_POP3(a) (sp3-- , (a) = stack3[sp3])

	int i,w,h, iDim;
	int x,y;
	int x2,y2;
	int sp2; // stack pointer
    int sp3;

	cv::Mat_<uchar> src = image.clone();
	w = src.cols;
	h = src.rows;
	iDim = w*h;

	cv::Point point;
	for (y=0; y<h; ++y)
	{
		for (x=0; x<w; ++x)
		{
			if ((src(y,x))!=0)   //punto non etichettato: seme trovato
			{
				// per ogni oggetto
				sp2 = 0;
				i = x + y*w;
				RG_PUSH2(i);

				// vuoto la lista dei punti
	    		sp3=0;
  		  		while (sp2>0)
				{// rg tradizionale

					RG_POP2(i);
					x2=i%w;
					y2=i/w;



					point.x=x2;
					point.y=y2;

					if(src(y2,x2))
					{
						RG_PUSH3(point);
						src(y2,x2) = 0;
					}
					
					// Inserisco i nuovi punti nello stack solo se esistono
					// e sono punti da etichettare

					// 4 connessi
					// sx
					if (x2>0 &&   (src(y2, x2-1)!=0))
						RG_PUSH2(i-1);
					// sotto
					if (y2>0 &&   (src(y2-1, x2)!=0))
						RG_PUSH2(i-w);
					// sopra
					if (y2<h-1 &&   (src(y2+1, x2)!=0))
						RG_PUSH2(i+w);
					// dx
					if (x2<w-1 &&   (src(y2, x2+1)!=0))
						RG_PUSH2(i+1);

					// 8 connessi
					if (x2>0 && y2>0 &&   (src(y2-1,x2-1)!=0))
						RG_PUSH2(i-w-1);
					if (x2>0 && y2<h-1 &&   (src(y2+1, x2-1)!=0))
						RG_PUSH2(i+w-1);
					if (x2<w-1 && y2>0 &&   (src(y2-1, x2+1)!=0))
						RG_PUSH2(i-w+1);
					if (x2<w-1 && y2<h-1 &&   (src(y2+1, x2+1)!=0))
						RG_PUSH2(i+w+1);

				}

				if (sp3 >= iMinLength)
				{
					vector<cv::Point> component;
					component.reserve(sp3);

					// etichetto il punto
					for (i=0; i<sp3; i++){
						// etichetto
						component.push_back(stack3[i]);
					}
					segments.push_back(component);
				}
			}
		}
	}
};
*/

/* Thinning Zhang e Suen（元のコード、フィルタ領域が4×4）
void Thinning(cv::Mat1b& imgMask, uchar byF, uchar byB) 
{
	std::cout << "DEBUG: Thinning() function was called." << std::endl;

	int r = imgMask.rows;
	int c = imgMask.cols;

	cv::Mat_<uchar> imgIT(r, c),imgM(r, c);

	for(int i = 0; i < r; ++i)
	{		
		for(int j = 0; j < c; ++j)
		{
			imgIT(i, j) = imgMask(i, j) == byF ? 1 : 0; // 前景（byF = 255）なら1、背景なら0に変換（二値化）
		}
	}

	// デバック
	cv::Mat binary_to_save;
    imgIT.convertTo(binary_to_save, CV_8U, 255);
    cv::imwrite("thinning_01_binary.png", binary_to_save);

	bool bSomethingDone = true;
	int iCount = 0;

	while (bSomethingDone) {
		bSomethingDone = false;
		fill(imgM.begin(), imgM.end(), 0);

		//prima iterazione
		for(int y = 1; y < r - 2; y++) {
			for(int x = 1; x < c - 2; x++) {

#define c_P0 imgIT(y-1,x-1)==1
#define c_P1 imgIT(y-1,x)==1
#define c_P2 imgIT(y-1,x+1)==1
#define c_P3 imgIT(y-1,x+2)==1
#define c_P4 imgIT(y,x-1)==1
#define c_P5 imgIT(y,x)==1
#define c_P6 imgIT(y,x+1)==1
#define c_P7 imgIT(y,x+2)==1
#define c_P8 imgIT(y+1,x-1)==1
#define c_P9 imgIT(y+1,x)==1
#define c_P10 imgIT(y+1,x+1)==1
#define c_P11 imgIT(y+1,x+2)==1
#define c_P12 imgIT(y+2,x-1)==1
#define c_P13 imgIT(y+2,x)==1
#define c_P14 imgIT(y+2,x+1)==1
#define c_P15 imgIT(y+2,x+2)==1

				if (c_P5) {
					if (c_P9) {
						if (c_P6) {
							if (c_P10) {
								if (c_P4) {
									if (c_P8) {
										if (c_P1) {
											continue;
										}
										else {
											if (c_P13) {
												if (c_P2) {
													if (c_P0) {
														continue;
													}
													else {
														goto a_2;
													}
												}
												else {
													goto a_2;
												}
											}
											else {
												if (c_P14) {
													if (c_P12) {
														if (c_P2) {
															if (c_P0) {
																continue;
															}
															else {
																goto a_2;
															}
														}
														else {
															goto a_2;
														}
													}
													else {
														continue;
													}
												}
												else {
													continue;
												}
											}
										}
									}
									else {
										continue;
									}
								}
								else {
									if (c_P1) {
										if (c_P2) {
											if (c_P7) {
												if (c_P8) {
													if (c_P0) {
														continue;
													}
													else {
														goto a_2;
													}
												}
												else {
													goto a_2;
												}
											}
											else {
												if (c_P11) {
													if (c_P3) {
														if (c_P8) {
															if (c_P0) {
																continue;
															}
															else {
																goto a_2;
															}
														}
														else {
															goto a_2;
														}
													}
													else {
														continue;
													}
												}
												else {
													continue;
												}
											}
										}
										else {
											continue;
										}
									}
									else {
										if (c_P0) {
											continue;
										}
										else {
											if (c_P8) {
												goto a_2;
											}
											else {
												if (c_P2) {
													goto a_2;
												}
												else {
													if (c_P14) {
														if (c_P13) {
															if (c_P11) {
																goto a_2;
															}
															else {
																if (c_P7) {
																	goto a_2;
																}
																else {
																	if (c_P3) {
																		goto a_2;
																	}
																	else {
																		continue;
																	}
																}
															}
														}
														else {
															goto a_2;
														}
													}
													else {
														if (c_P13) {
															goto a_2;
														}
														else {
															if (c_P12) {
																goto a_2;
															}
															else {
																if (c_P11) {
																	if (c_P7) {
																		continue;
																	}
																	else {
																		goto a_2;
																	}
																}
																else {
																	if (c_P15) {
																		goto a_2;
																	}
																	else {
																		if (c_P7) {
																			goto a_2;
																		}
																		else {
																			if (c_P3) {
																				goto a_2;
																			}
																			else {
																				continue;
																			}
																		}
																	}
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
							else {
								continue;
							}
						}
						else {
							if (c_P0) {
								if (c_P8) {
									if (c_P4) {
										if (c_P2) {
											if (c_P10) {
												continue;
											}
											else {
												if (c_P1) {
													goto a_2;
												}
												else {
													continue;
												}
											}
										}
										else {
											goto a_2;
										}
									}
									else {
										continue;
									}
								}
								else {
									continue;
								}
							}
							else {
								if (c_P2) {
									continue;
								}
								else {
									if (c_P1) {
										continue;
									}
									else {
										if (c_P8) {
											goto a_2;
										}
										else {
											if (c_P10) {
												if (c_P4) {
													continue;
												}
												else {
													goto a_2;
												}
											}
											else {
												continue;
											}
										}
									}
								}
							}
						}
					}
					else {
						if (c_P6) {
							if (c_P0) {
								if (c_P2) {
									if (c_P1) {
										if (c_P8) {
											if (c_P10) {
												continue;
											}
											else {
												if (c_P4) {
													goto a_2;
												}
												else {
													continue;
												}
											}
										}
										else {
											goto a_2;
										}
									}
									else {
										continue;
									}
								}
								else {
									continue;
								}
							}
							else {
								if (c_P8) {
									continue;
								}
								else {
									if (c_P4) {
										continue;
									}
									else {
										if (c_P2) {
											goto a_2;
										}
										else {
											if (c_P10) {
												if (c_P1) {
													continue;
												}
												else {
													goto a_2;
												}
											}
											else {
												continue;
											}
										}
									}
								}
							}
						}
						else {
							if (c_P10) {
								continue;
							}
							else {
								if (c_P4) {
									if (c_P1) {
										if (c_P0) {
											goto a_2;
										}
										else {
											continue;
										}
									}
									else {
										if (c_P2) {
											continue;
										}
										else {
											if (c_P8) {
												goto a_2;
											}
											else {
												if (c_P0) {
													goto a_2;
												}
												else {
													continue;
												}
											}
										}
									}
								}
								else {
									if (c_P8) {
										continue;
									}
									else {
										if (c_P1) {
											if (c_P2) {
												goto a_2;
											}
											else {
												if (c_P0) {
													goto a_2;
												}
												else {
													continue;
												}
											}
										}
										else {
											continue;
										}
									}
								}
							}
						}
					}
				}
				else {
					continue;
				}


a_2:
				imgM(y, x) = 1;
				bSomethingDone = true;
			}
		}
		
		for (int r = 0; r < imgIT.rows; ++r) {
			for (int c = 0; c < imgIT.cols; ++c) {
				if (imgM(r, c) == 1)
					imgIT(r, c) = 0;
			}
		}
	}

	// デバック
	cv::Mat thinned_to_save;
    imgIT.convertTo(thinned_to_save, CV_8U, 255);
    cv::imwrite("thinning_02_thinned.png", thinned_to_save);

	for(int i = 0; i < r; ++i)
	{		
		for(int j = 0; j < c; ++j)
		{
			imgMask(i, j) = imgIT(i, j) == 1 ? byF : byB;
		}
	}
};
*/