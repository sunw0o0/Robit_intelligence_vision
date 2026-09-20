#include "hw1.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <fstream>

namespace hw1 {

namespace {

// 색별 HSV 범위 (OpenCV: H 0~179, S/V 0~255). 사진에 맞게 여기 값을 고치면 된다.
constexpr int kRedLow1[3]  = {   0, 120,  70 }, kRedHigh1[3] = {  10, 255, 255 };
constexpr int kRedLow2[3]  = { 170, 120,  70 }, kRedHigh2[3] = { 179, 255, 255 };   // 빨강은 H 0 근처와 179 근처 양쪽
constexpr int kGreenLow[3] = {  35,  80,  70 }, kGreenHigh[3] = {  85, 255, 255 };
constexpr int kBlueLow[3]  = { 100, 120,  50 }, kBlueHigh[3]  = { 130, 255, 255 };

cv::Scalar sc(const int v[3]) { return cv::Scalar(v[0], v[1], v[2]); }

// BGR -> HSV -> inRange (마스크 3장)
void makeMasks(const cv::Mat& bgr, cv::Mat masks[BALL_COUNT])
{
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    cv::Mat r1, r2;
    cv::inRange(hsv, sc(kRedLow1), sc(kRedHigh1), r1);
    cv::inRange(hsv, sc(kRedLow2), sc(kRedHigh2), r2);
    cv::bitwise_or(r1, r2, masks[RED]);

    cv::inRange(hsv, sc(kGreenLow), sc(kGreenHigh), masks[GREEN]);
    cv::inRange(hsv, sc(kBlueLow),  sc(kBlueHigh),  masks[BLUE]);
}

Stats measure(const cv::Mat& mask)
{
    Stats s;
    s.whitePixels = cv::countNonZero(mask);

    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    for (int i = 1; i < n; ++i) {
        ++s.blobs;
        if (stats.at<int>(i, cv::CC_STAT_AREA) < 30) ++s.tinyBlobs;
    }

    return s;
}

// 빨강/초록/파랑 마스크를 각자의 색으로 칠해서 한 장으로 합친다 (검은 배경)
cv::Mat makeComposite(const cv::Mat masks[BALL_COUNT])
{
    cv::Mat out = cv::Mat::zeros(masks[RED].size(), CV_8UC3);
    out.setTo(cv::Scalar(0, 0, 255), masks[RED]);     // BGR: 빨강
    out.setTo(cv::Scalar(0, 255, 0), masks[GREEN]);   // 초록
    out.setTo(cv::Scalar(255, 0, 0), masks[BLUE]);    // 파랑
    return out;
}

}  // namespace

const char* ballName(int i)
{
    static const char* names[BALL_COUNT] = { "red", "green", "blue" };
    return (i >= 0 && i < BALL_COUNT) ? names[i] : "?";
}

Result run(const cv::Mat& bgr, int kernel)
{
    Result r;
    if (bgr.empty()) return r;
    if (kernel < 1) kernel = 1;
    if (kernel % 2 == 0) ++kernel;                      // 가우시안 커널은 홀수여야 함

    // A) 블러 없이 바로 HSV 변환 후 inRange
    makeMasks(bgr, r.maskRaw);

    // B) 가우시안 필터 적용 후 같은 작업
    cv::GaussianBlur(bgr, r.blurred, cv::Size(kernel, kernel), 0);
    makeMasks(r.blurred, r.maskBlur);

    r.compositeRaw  = makeComposite(r.maskRaw);
    r.compositeBlur = makeComposite(r.maskBlur);

    for (int i = 0; i < BALL_COUNT; ++i) {
        r.statsRaw[i]  = measure(r.maskRaw[i]);
        r.statsBlur[i] = measure(r.maskBlur[i]);
    }
    return r;
}

bool saveAll(const cv::Mat& bgr, const Result& r, const std::string& dir, int kernel)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return false;

    auto path = [&](const std::string& name) { return (fs::path(dir) / name).string(); };

    cv::imwrite(path("01_original.png"), bgr);
    cv::imwrite(path("02_gaussian_blurred.png"), r.blurred);
    for (int i = 0; i < BALL_COUNT; ++i) {
        cv::imwrite(path(std::string("mask_") + ballName(i) + "_raw.png"),  r.maskRaw[i]);
        cv::imwrite(path(std::string("mask_") + ballName(i) + "_blur.png"), r.maskBlur[i]);
    }
    cv::imwrite(path("composite_raw.png"),  r.compositeRaw);
    cv::imwrite(path("composite_blur.png"), r.compositeBlur);

    std::ofstream f(path("metrics.txt"));
    if (!f) return false;
    f << "가우시안 커널 크기: " << kernel << "x" << kernel << "\n";
    f << "(작은 덩어리 = 30픽셀 미만의 흰 점)\n\n";
    f << "색\t\t구분\t\t흰 픽셀\t덩어리\t작은 덩어리\n";
    for (int i = 0; i < BALL_COUNT; ++i) {
        const Stats& a = r.statsRaw[i];
        const Stats& b = r.statsBlur[i];
        f << ballName(i) << "\t\t블러 전\t\t" << a.whitePixels << "\t" << a.blobs << "\t"
          << a.tinyBlobs << "\n";
        f << ballName(i) << "\t\t블러 후\t\t" << b.whitePixels << "\t" << b.blobs << "\t"
          << b.tinyBlobs << "\n";
    }
    return true;
}

}  // namespace hw1
