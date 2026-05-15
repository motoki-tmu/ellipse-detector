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

void CraterMatching(std::vector<Ellipse>& ellipses, std::vector<point> model, std::vector<std::vector<double>> line_segment, std::vector<MatchPairSet>& MatchPair) 
{
    // ハッシュテーブルを作成(キー値は線分の長さの10の位，要素は線分の管理番号)
	static const std::unordered_multimap<int, int> line_length = []()
    {
        std::unordered_multimap<int, int> line_length;
        for (int i = 0; i < 7; i++) {line_length.insert(std::make_pair(20, i));}
	    for (int i = 7; i < 29; i++) {line_length.insert(std::make_pair(30, i));}
	    for (int i = 29; i < 66; i++) {line_length.insert(std::make_pair(40, i));}
	    for (int i = 66; i < 115; i++) {line_length.insert(std::make_pair(50, i));}
	    for (int i = 115; i < 177; i++) {line_length.insert(std::make_pair(60, i));}
	    for (int i = 177; i < 244; i++) {line_length.insert(std::make_pair(70, i));}
	    for (int i = 244; i < 320; i++) {line_length.insert(std::make_pair(80, i));}
	    for (int i = 320; i < 406; i++) {line_length.insert(std::make_pair(90, i));}
	    for (int i = 406; i < 489; i++) {line_length.insert(std::make_pair(100, i));}
	    for (int i = 489; i < 570; i++) {line_length.insert(std::make_pair(110, i));}
	    for (int i = 570; i < 672; i++) {line_length.insert(std::make_pair(120, i));}
	    for (int i = 672; i < 795; i++) {line_length.insert(std::make_pair(130, i));}
	    for (int i = 795; i < 893; i++) {line_length.insert(std::make_pair(140, i));}
	    for (int i = 893; i < 1018; i++) {line_length.insert(std::make_pair(150, i));}
	    for (int i = 1018; i < 1135; i++) {line_length.insert(std::make_pair(160, i));}
	    for (int i = 1135; i < 1273; i++) {line_length.insert(std::make_pair(170, i));}
	    for (int i = 1273; i < 1400; i++) {line_length.insert(std::make_pair(180, i));}
	    for (int i = 1400; i < 1518; i++) {line_length.insert(std::make_pair(190, i));}
	    for (int i = 1518; i < 1651; i++) {line_length.insert(std::make_pair(200, i));}
	    for (int i = 1651; i < 1787; i++) {line_length.insert(std::make_pair(210, i));}
	    for (int i = 1787; i < 1939; i++) {line_length.insert(std::make_pair(220, i));}
        for (int i = 1939; i < 2085; i++) {line_length.insert(std::make_pair(230, i));}
        for (int i = 2085; i < 2239; i++) {line_length.insert(std::make_pair(240, i));}
        for (int i = 2239; i < 2363; i++) {line_length.insert(std::make_pair(250, i));}
        for (int i = 2363; i < 2518; i++) {line_length.insert(std::make_pair(260, i));}
        for (int i = 2518; i < 2674; i++) {line_length.insert(std::make_pair(270, i));}
        for (int i = 2674; i < 2836; i++) {line_length.insert(std::make_pair(280, i));}
        for (int i = 2836; i < 2975; i++) {line_length.insert(std::make_pair(290, i));}
        for (int i = 2975; i < 3127; i++) {line_length.insert(std::make_pair(300, i));}
        for (int i = 3127; i < 3286; i++) {line_length.insert(std::make_pair(310, i));}
        for (int i = 3286; i < 3421; i++) {line_length.insert(std::make_pair(320, i));}
        for (int i = 3421; i < 3578; i++) {line_length.insert(std::make_pair(330, i));}
        for (int i = 3578; i < 3711; i++) {line_length.insert(std::make_pair(340, i));}
        for (int i = 3711; i < 3839; i++) {line_length.insert(std::make_pair(350, i));}
        for (int i = 3839; i < 4002; i++) {line_length.insert(std::make_pair(360, i));}
        for (int i = 4002; i < 4161; i++) {line_length.insert(std::make_pair(370, i));}
        for (int i = 4161; i < 4301; i++) {line_length.insert(std::make_pair(380, i));}
        for (int i = 4301; i < 4442; i++) {line_length.insert(std::make_pair(390, i));}
        for (int i = 4442; i < 4579; i++) {line_length.insert(std::make_pair(400, i));}
        for (int i = 4579; i < 4721; i++) {line_length.insert(std::make_pair(410, i));}
        for (int i = 4721; i < 4852; i++) {line_length.insert(std::make_pair(420, i));}
        for (int i = 4852; i < 4972; i++) {line_length.insert(std::make_pair(430, i));}
        for (int i = 4972; i < 5111; i++) {line_length.insert(std::make_pair(440, i));}
        for (int i = 5111; i < 5230; i++) {line_length.insert(std::make_pair(450, i));}
        for (int i = 5230; i < 5352; i++) {line_length.insert(std::make_pair(460, i));}
        for (int i = 5352; i < 5475; i++) {line_length.insert(std::make_pair(470, i));}
        for (int i = 5475; i < 5595; i++) {line_length.insert(std::make_pair(480, i));}
        for (int i = 5595; i < 5713; i++) {line_length.insert(std::make_pair(490, i));}
        for (int i = 5713; i < 5815; i++) {line_length.insert(std::make_pair(500, i));}
        for (int i = 5815; i < 5916; i++) {line_length.insert(std::make_pair(510, i));}
        for (int i = 5916; i < 6001; i++) {line_length.insert(std::make_pair(520, i));}
        for (int i = 6001; i < 6118; i++) {line_length.insert(std::make_pair(530, i));}
        for (int i = 6118; i < 6203; i++) {line_length.insert(std::make_pair(540, i));}
        for (int i = 6203; i < 6279; i++) {line_length.insert(std::make_pair(550, i));}
        for (int i = 6279; i < 6371; i++) {line_length.insert(std::make_pair(560, i));}
        for (int i = 6371; i < 6444; i++) {line_length.insert(std::make_pair(570, i));}
        for (int i = 6444; i < 6503; i++) {line_length.insert(std::make_pair(580, i));}
        for (int i = 6503; i < 6567; i++) {line_length.insert(std::make_pair(590, i));}
        for (int i = 6567; i < 6630; i++) {line_length.insert(std::make_pair(600, i));}
        for (int i = 6630; i < 6682; i++) {line_length.insert(std::make_pair(610, i));}
        for (int i = 6682; i < 6722; i++) {line_length.insert(std::make_pair(620, i));}
        for (int i = 6722; i < 6775; i++) {line_length.insert(std::make_pair(630, i));}
        for (int i = 6775; i < 6814; i++) {line_length.insert(std::make_pair(640, i));}
        for (int i = 6814; i < 6838; i++) {line_length.insert(std::make_pair(650, i));}
        for (int i = 6838; i < 6871; i++) {line_length.insert(std::make_pair(660, i));}
        for (int i = 6871; i < 6900; i++) {line_length.insert(std::make_pair(670, i));}
        for (int i = 6900; i < 6918; i++) {line_length.insert(std::make_pair(680, i));}
        for (int i = 6918; i < 6934; i++) {line_length.insert(std::make_pair(690, i));}
        for (int i = 6934; i < 6945; i++) {line_length.insert(std::make_pair(700, i));}
        for (int i = 6945; i < 6962; i++) {line_length.insert(std::make_pair(710, i));}
        for (int i = 6962; i < 6972; i++) {line_length.insert(std::make_pair(720, i));}
        for (int i = 6972; i < 6979; i++) {line_length.insert(std::make_pair(730, i));}
        for (int i = 6979; i < 6986; i++) {line_length.insert(std::make_pair(740, i));}
        for (int i = 6986; i < 6995; i++) {line_length.insert(std::make_pair(750, i));}
        for (int i = 6995; i < 7001; i++) {line_length.insert(std::make_pair(760, i));}
        for (int i = 7001; i < 7006; i++) {line_length.insert(std::make_pair(770, i));}
        for (int i = 7006; i < 7010; i++) {line_length.insert(std::make_pair(780, i));}
        for (int i = 7010; i < 7013; i++) {line_length.insert(std::make_pair(790, i));}
        for (int i = 7013; i < 7015; i++) {line_length.insert(std::make_pair(800, i));}
        return line_length;
    }();

    std::vector<MatchPairSet> best_match; // マッチングしたworld点とmodel点をセットで格納
	
	int num_world = ellipses.size();
    std::vector<point> world(num_world);
    for (int i = 0; i < num_world; i++)
    {
        world[i].x = ellipses[i]._xc;
        world[i].y = ellipses[i]._yc;
    }

    // スコア順のworld点を確認
    /*
    for (int i = 0; i < num_world; i++)
	{
		std::cout << "[" << i << "] " << "World:" << world[i].x << ", " << world[i].y << std::endl;
	}
    */

    std::vector<MatchPairSet> current_match; // bestならbest_matchにコピー
    current_match.reserve(num_world);

    double r, Wtheta;		// worldの2点間の距離と角度
    double pic_c_x = 350;	// 入力画像中心
	double pic_c_y = 350; 	// 700pix

    double q, q_min, q_max; // modelの2点間の距離と閾値
	double Mtheta; 			// modelの2点間の角度
	double deltatheta; 		// = Mtheta - Wtheta
	double tx, ty; 			// 平行移動量
	double s; 				// スケーリングファクタ

    int th = 10; 			// マッチング点のピクセル誤差許容値
    int R = 10; 			// 必要マッチング数
	int num_matching = 0, best = 0; 
    double estimate_m_x = 0, estimate_m_y = 0;

    double best_s, best_deltatheta;

    int count = 0;

    // worldで作成する線分はスコア上位p点のみでループ
    int p = 2;
    for (int a = 0; a < p; a++)
    {
        for (int b = a + 1; b < p; b++)
        {
            // 点a, bからなる線分とその周りの点による幾何計算
			double dxw = world[a].x - world[b].x;
			double dyw = world[a].y - world[b].y;
            r = sqrt(dxw * dxw + dyw * dyw);
            Wtheta = atan2(dyw, dxw);

            if (r < 50) continue;

            // 使用するHashtableの範囲の最小と最大のHashのキー値を決定
            double q_min = r * 0.3;
            double q_max = r * 0.7;
            int lq_min_round = (static_cast<int>(q_min) / 10) * 10;
            int lq_max_round = ((static_cast<int>(q_max) + 9) / 10) * 10;

            // 使用するHashtableの範囲内で線分を探索
            for (int key = lq_min_round; key < lq_max_round; key += 10)
            {
                auto range = line_length.equal_range(key); // キー(i)とその開始・終了番号を格納（2×2ベクトル）

				// キーの範囲（開始番号から終了番号）内をループ
                for (auto itr = range.first; itr != range.second; ++itr)
                {
                    int i = line_segment[1][itr->second], j = line_segment[2][itr->second]; // 点a, bの番号をi, jに格納
					double dxm = model[i].x - model[j].x;
					double dym = model[i].y - model[j].y;
			        q = sqrt(dxm * dxm + dym * dym);
			        s = q / r;
                    Mtheta = atan2(dym, dxm);

                    // 正負両方チェック
                    for (int try_inv = 0; try_inv < 2; try_inv++)
                    {
                        current_match.clear();
                        if(try_inv == 0) // worldの点aとmodelの点iが対応
						{
							deltatheta = std::atan2(std::sin(Mtheta - Wtheta), std::cos(Mtheta - Wtheta)); // -π~πの範囲に収めるために丸め込む
							tx = model[i].x - s * cos(deltatheta) * world[a].x + s * sin(deltatheta) * world[a].y;
                        	ty = model[i].y - s * sin(deltatheta) * world[a].x - s * cos(deltatheta) * world[a].y;
							current_match.push_back({world[a], model[i]});
							current_match.push_back({world[b], model[j]});
						}
                        else			 // worldの点aとmodelの点jが対応
						{
							deltatheta = std::atan2(std::sin(Mtheta - Wtheta + M_PI), std::cos(Mtheta - Wtheta + M_PI));
                        	tx = model[j].x - s * cos(deltatheta) * world[a].x + s * sin(deltatheta) * world[a].y;
                        	ty = model[j].y - s * sin(deltatheta) * world[a].x - s * cos(deltatheta) * world[a].y;
							current_match.push_back({world[a], model[j]});
							current_match.push_back({world[b], model[i]});
						}

                        double scos = s * cos(deltatheta); double ssin = s * sin(deltatheta);

						// 線分周りの点の一致度合いをチェック
                        for (int k = 0; k < num_world; k++)
                        {
                            if (k == a || k == b) continue;

                            if (num_matching + (num_world - k) <= best) break; // 無駄な計算を省く

                            double Tc_x = scos * world[k].x - ssin * world[k].y + tx;
                            double Tc_y = ssin * world[k].x + scos * world[k].y + ty;
                        
                            for (int l = 0; l < NUM_CRATERS; l++)
                            {
                                if (l == i || l == j) continue;
                                if (std::abs(model[l].x - Tc_x) <= th && std::abs(model[l].y - Tc_y) <= th)
                                {
									current_match.push_back({world[k], model[l]});
									num_matching++;
                                    break;
                                }
                            }
                            count++;
                        }

                        if (num_matching >= R && num_matching > best)
                        {
                            best = num_matching;
							best_match = current_match;
                            estimate_m_x = s * cos(deltatheta) * pic_c_x - s * sin(deltatheta) * pic_c_y + tx;
					        estimate_m_y = s * sin(deltatheta) * pic_c_x + s * cos(deltatheta) * pic_c_y + ty;
                            best_s = s; best_deltatheta = deltatheta;
                        }

                        num_matching = 0;
                    }
                }
            }
        }
    }

    if (best > 0)
    {
		MatchPair = best_match;
		
        std::cout << "matching points: " << best + 2 << std::endl; // 線分を形成する2点もマッチング済みに考慮
        std::cout << "deltatheta: " << best_deltatheta << std::endl;
        std::cout << "scale: " << best_s << std::endl;
        std::cout << "estimated result: " << estimate_m_x << ", " << estimate_m_y << std::endl;
		std::cout << "best_match_size: " << best_match.size() << std::endl;
        std::cout << "count: " << count << std::endl;

		/*
		for (size_t i = 0; i < best_match.size(); i++)
		{
			std::cout << "[" << i << "] " << "World: (" << best_match[i].world_pt.x << ", " << best_match[i].world_pt.y << ") => "
					  << "Model: (" << best_match[i].model_pt.x << ", " << best_match[i].model_pt.y << ")" << std::endl;
		}
		*/
    }
    else
    {
        std::cout << "unestimable" << std::endl;
    }
}
