#include "PnP.h"

#include <vector>
#include <cmath>

void PnP(std::vector<MatchPairSet>& MatchPair, ResultData& resultdata)
{
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;

    for (int i =0; i < MatchPair.size(); i++)
    {
        objectPoints.push_back(cv::Point3f(MatchPair[i].model_pt.x, MatchPair[i].model_pt.y, 0.0f)); // 模擬月面のクレータが同一平面上にあると仮定
        imagePoints.push_back(cv::Point2f(MatchPair[i].world_pt.x, MatchPair[i].world_pt.y));
    }

    if (objectPoints.size() < 4) return;

    // カメラ内部パラメータ（Logicool 720p）
    double FOV_rad = 30.0 * M_PI / 180.0;           // 水平視野角60degなら
    double fx = (1280.0 / 2.0) / std::tan(FOV_rad); // 画像を切り取る前の大きさを用いる
    double fy = fx;                                 // ピクセルが正方形なら
    double cx = 350.0; double cy = 350.0;           // 700pix四方に切り取るなら
    
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
        fx,  0, cx,
         0, fy, cy,
         0,  0,  1);

    // 歪み係数，ゼロとする
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
    
    // 回転ベクトルと並進ベクトル 
    cv::Mat rvec, tvec;

    bool success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
    //bool success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP);
    //bool success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_SQPNP);
    //bool success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE);
    
    
    if (success)
    {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        //tvec = - R.t() * tvec; // カメラの絶対位置を出力するための計算
        std::cout << "estimated_position: " << tvec.at<double>(0) << ", " << tvec.at<double>(1) << ", " << tvec.at<double>(2) << std::endl;
        std::cout << "estimated_attitude: " << rvec.at<double>(0) << ", " << rvec.at<double>(1) << ", " << rvec.at<double>(2) << std::endl;
    }
    else {std::cout << "PnP false" << std::endl;}

    resultdata.PnPposition[0] = tvec.at<double>(0);
    resultdata.PnPposition[1] = tvec.at<double>(1);
    resultdata.PnPposition[2] = tvec.at<double>(2);
    resultdata.PnPattitude[0] = rvec.at<double>(0);
    resultdata.PnPattitude[1] = rvec.at<double>(1);
    resultdata.PnPattitude[2] = rvec.at<double>(2);
}