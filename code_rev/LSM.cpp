#include "LSM.h"

#include <cstdio>
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>
// #include "windows.h"

const int NUM_CRATERS = 119;
const int NUM_COMBINATIONS = NUM_CRATERS * (NUM_CRATERS - 1) / 2;

int CraterMatching(std::vector<Ellipse>& ellipses) 
{
	typedef struct {double x; double y;} point; // pointという名前の構造体を宣言

	// csvファイルからmodelクレータ情報を取り出すための配列arrayを作成
	std::vector<std::vector<double>> array(NUM_CRATERS, std::vector<double>(4, 0.0));
	std::ifstream stream("../model_info.csv"); // num, x, y, r
	std::string line;
	int row = 0;

	if (!stream.is_open())
	{
		std::cerr << "error: file not open" << std::endl;
		return 1;
	}

    // model_info.csvをarrayに格納
	while (getline(stream, line))
	{
		std::stringstream ss(line);
		int col = 0;

        replace(line.begin(), line.end(), ',', ' ');
        ss.str(line);
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

	// 構造体modelに各クレータの座標を登録
	std::vector<point> model;
	model.reserve(NUM_CRATERS);
	for (const auto& row : array)
	{
		if (row.size() > 2)
		{
			model.emplace_back(point{(float)row[1], (float)row[2]});
		}
	}

	// Hashmapの線分を格納する配列line_combiを作成
	std::vector<std::vector<double>> line_combi(NUM_COMBINATIONS, std::vector<double>(4, 0.0));

	// csvファイルの中身をline_combi配列に格納
	std::ifstream stream2("../line_combi_sorted.csv"); // key, length, a_num, b_num
	std::string line2;
	int row2 = 0;

	if (!stream2.is_open())
	{
		std::cerr << "error: file not open" << std::endl;
		return 1;
	}

	while (getline(stream2, line2))
	{
		std::stringstream ss2(line2);
		int col2 = 0;

        replace(line2.begin(), line2.end(), ',', ' ');
        ss2.str(line2);
        double value;

		while (ss2 >> value)
        {
            if (row2 < line_combi.size() && col2 < line_combi[row2].size())
            {
                line_combi[row2][col2] = value;
            }
            col2++;
        }
        row2++;
	}

    //std::cout << line_combi[406][0] << std::endl;

	std::vector<std::vector<double>> l_c(3, std::vector<double>(NUM_COMBINATIONS, 0.0));
	for (int i = 0; i < NUM_COMBINATIONS; i++)
	{
		l_c[0][i] = i;
		l_c[1][i] = line_combi[i][2];
		l_c[2][i] = line_combi[i][3];
	}

    // ハッシュテーブルを作成(キー値は線分の長さの10の位，要素は線分の管理番号)
	std::unordered_multimap<int, int> line_length;
	for (int i = 0; i < 7; i++) {
		line_length.insert(std::make_pair(20, i));
	}
	for (int i = 7; i < 29; i++) {
		line_length.insert(std::make_pair(30, i));
	}
	for (int i = 29; i < 66; i++) {
		line_length.insert(std::make_pair(40, i));
	}
	for (int i = 66; i < 115; i++) {
		line_length.insert(std::make_pair(50, i));
	}
	for (int i = 115; i < 177; i++) {
		line_length.insert(std::make_pair(60, i));
	}
	for (int i = 177; i < 244; i++) {
		line_length.insert(std::make_pair(70, i));
	}
	for (int i = 244; i < 320; i++) {
		line_length.insert(std::make_pair(80, i));
	}
	for (int i = 320; i < 406; i++) {
		line_length.insert(std::make_pair(90, i));
	}
	for (int i = 406; i < 489; i++) {
		line_length.insert(std::make_pair(100, i));
	}
	for (int i = 489; i < 570; i++) {
		line_length.insert(std::make_pair(110, i));
	}
	for (int i = 570; i < 672; i++) {
		line_length.insert(std::make_pair(120, i));
	}
	for (int i = 672; i < 795; i++) {
		line_length.insert(std::make_pair(130, i));
	}
	for (int i = 795; i < 893; i++) {
		line_length.insert(std::make_pair(140, i));
	}
	for (int i = 893; i < 1018; i++) {
		line_length.insert(std::make_pair(150, i));
	}
	for (int i = 1018; i < 1135; i++) {
		line_length.insert(std::make_pair(160, i));
	}
	for (int i = 1135; i < 1273; i++) {
		line_length.insert(std::make_pair(170, i));
	}
	for (int i = 1273; i < 1400; i++) {
		line_length.insert(std::make_pair(180, i));
	}
	for (int i = 1400; i < 1518; i++) {
		line_length.insert(std::make_pair(190, i));
	}
	for (int i = 1518; i < 1651; i++) {
		line_length.insert(std::make_pair(200, i));
	}
	for (int i = 1651; i < 1787; i++) {
		line_length.insert(std::make_pair(210, i));
	}
	for (int i = 1787; i < 1939; i++) {
		line_length.insert(std::make_pair(220, i));
	}
    for (int i = 1939; i < 2085; i++) {
		line_length.insert(std::make_pair(230, i));
	}
    for (int i = 2085; i < 2239; i++) {
		line_length.insert(std::make_pair(240, i));
	}
    for (int i = 2239; i < 2363; i++) {
		line_length.insert(std::make_pair(250, i));
	}
    for (int i = 2363; i < 2518; i++) {
		line_length.insert(std::make_pair(260, i));
	}
    for (int i = 2518; i < 2674; i++) {
		line_length.insert(std::make_pair(270, i));
	}
    for (int i = 2674; i < 2836; i++) {
		line_length.insert(std::make_pair(280, i));
	}
    for (int i = 2836; i < 2975; i++) {
		line_length.insert(std::make_pair(290, i));
	}
    for (int i = 2975; i < 3127; i++) {
		line_length.insert(std::make_pair(300, i));
	}
    for (int i = 3127; i < 3286; i++) {
		line_length.insert(std::make_pair(310, i));
	}
    for (int i = 3286; i < 3421; i++) {
		line_length.insert(std::make_pair(320, i));
	}
    for (int i = 3421; i < 3578; i++) {
		line_length.insert(std::make_pair(330, i));
	}
    for (int i = 3578; i < 3711; i++) {
		line_length.insert(std::make_pair(340, i));
	}
    for (int i = 3711; i < 3839; i++) {
		line_length.insert(std::make_pair(350, i));
	}
    for (int i = 3839; i < 4002; i++) {
		line_length.insert(std::make_pair(360, i));
	}
    for (int i = 4002; i < 4161; i++) {
		line_length.insert(std::make_pair(370, i));
	}
    for (int i = 4161; i < 4301; i++) {
		line_length.insert(std::make_pair(380, i));
	}
    for (int i = 4301; i < 4442; i++) {
		line_length.insert(std::make_pair(390, i));
	}
    for (int i = 4442; i < 4579; i++) {
		line_length.insert(std::make_pair(400, i));
	}
    for (int i = 4579; i < 4721; i++) {
		line_length.insert(std::make_pair(410, i));
	}
    for (int i = 4721; i < 4852; i++) {
		line_length.insert(std::make_pair(420, i));
	}
    for (int i = 4852; i < 4972; i++) {
		line_length.insert(std::make_pair(430, i));
	}
    for (int i = 4972; i < 5111; i++) {
		line_length.insert(std::make_pair(440, i));
	}
    for (int i = 5111; i < 5230; i++) {
		line_length.insert(std::make_pair(450, i));
	}
    for (int i = 5230; i < 5352; i++) {
		line_length.insert(std::make_pair(460, i));
	}
    for (int i = 5352; i < 5475; i++) {
		line_length.insert(std::make_pair(470, i));
	}
    for (int i = 5475; i < 5595; i++) {
		line_length.insert(std::make_pair(480, i));
	}
    for (int i = 5595; i < 5713; i++) {
		line_length.insert(std::make_pair(490, i));
	}
    for (int i = 5713; i < 5815; i++) {
		line_length.insert(std::make_pair(500, i));
	}
    for (int i = 5815; i < 5916; i++) {
		line_length.insert(std::make_pair(510, i));
	}
    for (int i = 5916; i < 6001; i++) {
		line_length.insert(std::make_pair(520, i));
	}
    for (int i = 6001; i < 6118; i++) {
		line_length.insert(std::make_pair(530, i));
	}
    for (int i = 6118; i < 6203; i++) {
		line_length.insert(std::make_pair(540, i));
	}
    for (int i = 6203; i < 6279; i++) {
		line_length.insert(std::make_pair(550, i));
	}
    for (int i = 6279; i < 6371; i++) {
		line_length.insert(std::make_pair(560, i));
	}
    for (int i = 6371; i < 6444; i++) {
		line_length.insert(std::make_pair(570, i));
	}
    for (int i = 6444; i < 6503; i++) {
		line_length.insert(std::make_pair(580, i));
	}
    for (int i = 6503; i < 6567; i++) {
		line_length.insert(std::make_pair(590, i));
	}
    for (int i = 6567; i < 6630; i++) {
		line_length.insert(std::make_pair(600, i));
	}
    for (int i = 6630; i < 6682; i++) {
		line_length.insert(std::make_pair(610, i));
	}
    for (int i = 6682; i < 6722; i++) {
		line_length.insert(std::make_pair(620, i));
	}
    for (int i = 6722; i < 6775; i++) {
		line_length.insert(std::make_pair(630, i));
	}
    for (int i = 6775; i < 6814; i++) {
		line_length.insert(std::make_pair(640, i));
	}
    for (int i = 6814; i < 6838; i++) {
		line_length.insert(std::make_pair(650, i));
	}
    for (int i = 6838; i < 6871; i++) {
		line_length.insert(std::make_pair(660, i));
	}
    for (int i = 6871; i < 6900; i++) {
		line_length.insert(std::make_pair(670, i));
	}
    for (int i = 6900; i < 6918; i++) {
		line_length.insert(std::make_pair(680, i));
	}
    for (int i = 6918; i < 6934; i++) {
		line_length.insert(std::make_pair(690, i));
	}
    for (int i = 6934; i < 6945; i++) {
		line_length.insert(std::make_pair(700, i));
	}
    for (int i = 6945; i < 6962; i++) {
		line_length.insert(std::make_pair(710, i));
	}
    for (int i = 6962; i < 6972; i++) {
		line_length.insert(std::make_pair(720, i));
	}
    for (int i = 6972; i < 6979; i++) {
		line_length.insert(std::make_pair(730, i));
	}
    for (int i = 6979; i < 6986; i++) {
		line_length.insert(std::make_pair(740, i));
	}
    for (int i = 6986; i < 6995; i++) {
		line_length.insert(std::make_pair(750, i));
	}
    for (int i = 6995; i < 7001; i++) {
		line_length.insert(std::make_pair(760, i));
	}
    for (int i = 7001; i < 7006; i++) {
		line_length.insert(std::make_pair(770, i));
	}
    for (int i = 7006; i < 7010; i++) {
		line_length.insert(std::make_pair(780, i));
	}
    for (int i = 7010; i < 7013; i++) {
		line_length.insert(std::make_pair(790, i));
	}
    for (int i = 7013; i < 7015; i++) {
		line_length.insert(std::make_pair(800, i));
	}

    // world（未完成）
    int num_world = ellipses.size();
    std::vector<point> world(num_world);
    for (int i = 0; i < num_world; i++)
    {
        world[i].x = ellipses[i]._xc;
        world[i].y = ellipses[i]._yc;
    }


    double r, Wtheta;
    double ex = 1.0;
    double satellite_tilt_rad = 0.0;
    double pic_c_x = ex * cos(satellite_tilt_rad) * 350 - ex * sin(satellite_tilt_rad) * 350 + 350 - 350 * cos(satellite_tilt_rad) + 350 * sin(satellite_tilt_rad);
	double pic_c_y = ex * sin(satellite_tilt_rad) * 350 + ex * cos(satellite_tilt_rad) * 350 + 350 - 350 * sin(satellite_tilt_rad) - 350 * cos(satellite_tilt_rad);

    double q, q_min, q_max;
	double Mtheta;
	double deltatheta;
	double tx, ty;
	double s;

    int matching = 0, best = 0;
    int th = 10;
    int R = 10;
    double estimate_m_x = 0, estimate_m_y = 0;

    int num_matching;
    double best_s, best_deltatheta;

	
    for (int a = 0; a < num_world; a++)
    {
        for (int b = 0; b < num_world; b++)
        {
            if (a == b) continue;

            // 点a, bからなる線分とその周りの点による幾何計算
            r = (sqrt((pow(abs(world[a].x - world[b].x), 2.0)) + (pow(abs(world[a].y - world[b].y), 2.0))));
            Wtheta = atan2(world[a].y - world[b].y, world[a].x - world[b].x);

            if (r < 50) continue;

            // 使用するHashtableの範囲の最小と最大のHashのキー値を決定
            double q_min = r * 0.3;
            double q_max = r * 0.7;
            int lq_min_round = (q_min / 10) * 10;
            int lq_max_round = ((q_max + 9) / 10) * 10;

            // 使用するHashtableの範囲内で線分を探索
            for (int key = lq_min_round; key < lq_max_round; key += 10)
            {
                auto range = line_length.equal_range(key); // キー(i)とその開始・終了番号を格納（2×2ベクトル）

                for (auto itr = range.first; itr != range.second; ++itr) // キーの範囲（開始番号から終了番号）内をループ
                {
                    int i = l_c[1][itr->second], j = l_c[2][itr->second]; // 点a, bの番号をi, jに格納
			        q = (sqrt((pow(abs(model[i].x - model[j].x), 2.0)) + (pow(abs(model[i].y - model[j].y), 2.0)))); // modelの2点間の距離
			        s = q / r;                                                                                       // スケーリングファクタ
                    Mtheta = atan2(model[j].y - model[i].y, model[j].x - model[i].x);                                // 線分とx軸のなす角

                    // 正負両方チェック
                    for (int try_inv = 0; try_inv < 2; try_inv++)
                    {
                        if(try_inv == 0) deltatheta = Mtheta - Wtheta;
                        else             deltatheta = Mtheta - Wtheta + 3.1415926535;

                        tx = model[j].x - s * cos(deltatheta) * world[a].x + s * sin(deltatheta) * world[a].y;
                        ty = model[j].y - s * sin(deltatheta) * world[a].x - s * cos(deltatheta) * world[a].y;

                        for (int k = 0; k < num_world; k++)
                        {
                            if (k == a || k == b) continue;
                            double Tc_x = s * cos(deltatheta) * world[k].x - s * sin(deltatheta) * world[k].y + tx;
                            double Tc_y = s * sin(deltatheta) * world[k].x + s * cos(deltatheta) * world[k].y + ty;
                        
                            for (int l = 0; l < NUM_CRATERS; l++)
                            {
                                if (l == i || l == j) continue;
                                if (abs(model[l].x - Tc_x) <= th && abs(model[l].y - Tc_y) <= th)
                                {
                                    matching++;
                                    break;
                                }
                            }
                        }

                        if (matching >= R && matching > best)
                        {
                            best = matching;
                            estimate_m_x = s * cos(deltatheta) * pic_c_x - s * sin(deltatheta) * pic_c_y + tx;
					        estimate_m_y = s * sin(deltatheta) * pic_c_x + s * cos(deltatheta) * pic_c_y + ty;
                            best_s = s; best_deltatheta = deltatheta;
                        }

                        matching = 0;
                    }
                }
            }
        }
    }

    if (best > 0)
    {
        std::cout << "matching points: " << best << std::endl;
        std::cout << "deltatheta: " << best_deltatheta << std::endl;
        std::cout << "scale: " << best_s << std::endl;
        std::cout << "estimated result: " << estimate_m_x << ", " << estimate_m_y << std::endl;
    }
    else
    {
        std::cout << "unestimable" << std::endl;
    }





    /*
	// 点a, bからなる線分とその周りの点による幾何計算
	r = (sqrt((pow(abs(world[a].x - world[b].x), 2.0)) + (pow(abs(world[a].y - world[b].y), 2.0))));
	Wtheta = atan2(world[a].y - world[b].y, world[a].x - world[b].x); // 線分abとx軸がなす角を計算


	//時間の計測（windows.hの参照が必要、linuxの場合他の方法でやる）
	//LARGE_INTEGER freq;
	//QueryPerformanceFrequency(&freq);
	//LARGE_INTEGER start, end;
	//QueryPerformanceCounter(&start);
	

	// world点a-bと対応する線分ベクトルをmodel mapから見つける
	double q, q_min, q_max;
	double Mtheta;
	double deltatheta_1, deltatheta_2;
	double tx_1, ty_1, tx_2, ty_2;
	double s;

	// 使用するHashtableの範囲の最小と最大のHashのキー値を決定
	// 実際は観測領域がmodel領域のスケールを上回ることはないため、要調整。観測領域が1/16~1/4なら0.25~0.50倍?
	q_min = r * 0.20;
	int lq_min = q_min;
	q_max = r * 1.50;
	int lq_max = q_max;
	
	// 10の倍数に切り捨て・切り上げ（ハッシュテーブルは10刻みだから）
	int lq_min_round = (lq_min / 10) * 10;
	int lq_max_round = ((lq_max + 9) / 10) * 10;

	// 使用するHashtableの範囲内で線分を探索
	for (int i = lq_min_round; i < lq_max_round; i += 10)
	{
		auto range = line_length.equal_range(i); // キー(i)とその開始・終了番号を格納（2×2ベクトル）
		auto itr = range.first; // キーと開始番号

		for (auto itr = range.first; itr != range.second; ++itr) // キーの範囲（開始番号から終了番号）内をループ
		{
			int i = l_c[1][itr->second], j = l_c[2][itr->second]; // 点a, bの番号をi, jに格納
			q = (sqrt((pow(abs(model[i].x - model[j].x), 2.0)) + (pow(abs(model[i].y - model[j].y), 2.0)))); // modelの2点間の距離
			s = q / r;                                                                                       // スケーリングファクタ
			Mtheta = atan2(model[j].y - model[i].y, model[j].x - model[i].x);                                // 線分とx軸のなす角
			deltatheta_1 = Mtheta - Wtheta;
			deltatheta_2 = Mtheta - Wtheta + 3.14159265358979; // 180度逆向きも検証が必要

			// 重ね合わせの変換式
			tx_1 = model[j].x - s * cos(deltatheta_1) * world[a].x + s * sin(deltatheta_1) * world[a].y;
			ty_1 = model[j].y - s * sin(deltatheta_1) * world[a].x - s * cos(deltatheta_1) * world[a].y;
			tx_2 = model[i].x - s * cos(deltatheta_2) * world[a].x + s * sin(deltatheta_2) * world[a].y;
			ty_2 = model[i].y - s * sin(deltatheta_2) * world[a].x - s * cos(deltatheta_2) * world[a].y;

			// 観測画像の点a, b以外の点をデータベース画像に座標変換
			int segment_value1 = 0, segment_value2 = 0; // 対応する点の数による線分の評価値
			int kaitenn_degree_number = 0;              // 回転角1か2のどっちを使ったか

			for (int k = 0; k < num_world; k++) // 検査する外点クレータの数を7thと同じするために+3にしてある（実験時は観測画像のクレータ数をcrater_num_worldに代入）
			{
                if (k == a || k == b) continue;

				double Tc_x_1 = 0, Tc_y_1 = 0;
				double Tc_x_2 = 0, Tc_y_2 = 0;
				Tc_x_1 = s * cos(deltatheta_1) * world[k].x - s * sin(deltatheta_1) * world[k].y + tx_1;
				Tc_y_1 = s * sin(deltatheta_1) * world[k].x + s * cos(deltatheta_1) * world[k].y + ty_1;
				Tc_x_2 = s * cos(deltatheta_2) * world[k].x - s * sin(deltatheta_2) * world[k].y + tx_2;
				Tc_y_2 = s * sin(deltatheta_2) * world[k].x + s * cos(deltatheta_2) * world[k].y + ty_2;

				// 対応点(=上で座標変換した後の座標と一致する点)がデータベース上にあるか（閾値 th pixel）
                int th = 15;
				for (int l = 0; l < NUM_CRATERS; l++)
				{
					if ((l != i) && (l != j))
					{
						if (((model[l].x - th <= Tc_x_1) && (Tc_x_1 <= model[l].x + th)) && ((model[l].y - th <= Tc_y_1) && (Tc_y_1 <= model[l].y + th)))
						{
							segment_value1 = segment_value1 + 1;
							kaitenn_degree_number = 1;
						}
						if (((model[l].x - th <= Tc_x_2) && (Tc_x_2 <= model[l].x + th)) && ((model[l].y - th <= Tc_y_2) && (Tc_y_2 <= model[l].y + th)))
						{
							segment_value2 = segment_value2 + 1;
							kaitenn_degree_number = 2;
						}
					}
				}
			}

            double estimate_m_x = 0, estimate_m_y = 0;
            int R = 5;

			// マッチングに成功したら（R個以上の対応点を見つけたら）
			if (segment_value1 >= R) 
			{
				if (kaitenn_degree_number == 1)
				{
					//printf("aに対応するのは観測画像内の点%d, bに対応するのは観測画像内の点%d, %f\n", j, i, s);
					//robotarm_idouryou_x = cos((-1)*deltatheta_1) / s * (goal_point_in_model.x - model[i].x) - sin((-1)*deltatheta_1) / s * (goal_point_in_model.y - model[i].y);
					//robotarm_idouryou_y = sin((-1)*deltatheta_1) / s * (goal_point_in_model.x - model[i].x) + cos((-1)*deltatheta_1) / s * (goal_point_in_model.y - model[i].y);
					estimate_m_x = s * cos(deltatheta_1) * pic_c_x - s * sin(deltatheta_1) * pic_c_y + tx_1;
					estimate_m_y = s * sin(deltatheta_1) * pic_c_x + s * cos(deltatheta_1) * pic_c_y + ty_1;
					//printf("ロボットアームの移動量は，x軸方向に%f，y軸方向に%f\n", robotarm_idouryou_x, robotarm_idouryou_y);
                    std::cout << "matching points: " << segment_value1 << std::endl; 
                    std::cout << "estimated result: " << estimate_m_x << ", " << estimate_m_y << std::endl;

					//QueryPerformanceCounter(&end);
					//double time = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
					//printf("time %lf[ms]\n", time);
				}
			}
			if (segment_value2 >= R)
			{
				if (kaitenn_degree_number == 2)
				{
					//printf("aに対応するのは観測画像内の点%d, bに対応するのは観測画像内の点%d, %f, %f\n", i, j, s, q);
					//robotarm_idouryou_x = cos((-1)*deltatheta_2) / s * (goal_point_in_model.x - model[j].x) - sin((-1)*deltatheta_2) / s * (goal_point_in_model.y - model[j].y);
					//robotarm_idouryou_y = sin((-1)*deltatheta_2) / s * (goal_point_in_model.x - model[j].x) + cos((-1)*deltatheta_2) / s * (goal_point_in_model.y - model[j].y);
					estimate_m_x = s * cos(deltatheta_2) * pic_c_x - s * sin(deltatheta_2) * pic_c_y + tx_2;
					estimate_m_y = s * sin(deltatheta_2) * pic_c_x + s * cos(deltatheta_2) * pic_c_y + ty_2;
					//printf("ロボットアームの移動量は，x軸方向に%f，y軸方向に%f\n", robotarm_idouryou_x, robotarm_idouryou_y);
                    std::cout << "matching points: " << segment_value2 << std::endl; 
					std::cout << "estimated result: " << estimate_m_x << ", " << estimate_m_y << std::endl;

					//QueryPerformanceCounter(&end);
					//double time = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
					//printf("time %lf[ms]\n", time);
				}
			}            
		}
	}*/

    return 0;
}
