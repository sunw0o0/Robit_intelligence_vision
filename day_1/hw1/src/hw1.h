#pragma once
// 과제 1: HSV inRange 를 가우시안 블러 전/후로 비교 (Qt에 의존하지 않는 모듈)
#include <opencv2/core.hpp>
#include <string>

namespace hw1 {

enum Ball { RED = 0, GREEN = 1, BLUE = 2, BALL_COUNT = 3 };

// 이진 마스크가 얼마나 깨끗한지 재는 수치 (보고서에 그대로 쓸 수 있음)
struct Stats {
    int whitePixels = 0;   // 흰색(255) 픽셀 수
    int blobs       = 0;   // 흰색 덩어리 개수 (8-연결)
    int tinyBlobs   = 0;   // 그중 30픽셀 미만인 작은 덩어리 = 잡음 점
};

struct Result {
    cv::Mat blurred;                       // 가우시안 필터를 적용한 이미지
    cv::Mat maskRaw[BALL_COUNT];           // 블러 없이 inRange 한 마스크
    cv::Mat maskBlur[BALL_COUNT];          // 블러 후 inRange 한 마스크
    cv::Mat compositeRaw;                  // 3색 마스크를 색깔로 합쳐 그린 영상 (블러 X)
    cv::Mat compositeBlur;                 // 3색 마스크를 색깔로 합쳐 그린 영상 (블러 O)
    Stats statsRaw[BALL_COUNT];
    Stats statsBlur[BALL_COUNT];
};

// kernel: 가우시안 커널 크기 (홀수)
Result run(const cv::Mat& bgr, int kernel = 9);

const char* ballName(int i);               // "red" / "green" / "blue"

// 원본, 블러 이미지, 마스크 6장, 합성 2장, metrics.txt 를 dir 에 저장
bool saveAll(const cv::Mat& bgr, const Result& r, const std::string& dir, int kernel);

}  // namespace hw1
