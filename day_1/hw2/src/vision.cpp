#include "vision.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace vision {

namespace {

constexpr double kMinLineArea = 200.0;   // 이보다 작으면 선이 없다고 판단
constexpr double kMinConeArea = 300.0;   // 이보다 작으면 노이즈로 보고 콘이 없다고 판단
constexpr double kLineMargin  = 5.0;     // "선 위" 판정 여유 (픽셀)

// 기준선: fitLine으로 기울어진 선도 처리
struct Line {
    bool ok = false;
    double vx = 0, vy = 0, x0 = 0, y0 = 0;   // 직선의 방향벡터와 지나는 점
    double thickness = 0;                    // 선 굵기 (면적 / 길이)
    cv::Rect box;                            // 선이 차지하는 영역 (그릴 때 길이로 사용)

    double xAt(double y) const {             // 세로선: 주어진 y에서의 x
        return (std::fabs(vy) < 1e-9) ? x0 : x0 + (y - y0) * vx / vy;
    }
    double yAt(double x) const {             // 가로선: 주어진 x에서의 y
        return (std::fabs(vx) < 1e-9) ? y0 : y0 + (x - x0) * vy / vx;
    }
};

Line analyzeLine(const cv::Mat& mask, bool vertical)
{
    Line l;

    // 작은 노이즈 덩어리는 버리고, 가장 큰 덩어리의 20% 이상인 것만 선으로 인정
    // (파란선은 흰선에 의해 두 조각으로 끊겨 있어도 둘 다 살아남음)
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    int maxArea = 0;
    for (int i = 1; i < n; ++i)
        maxArea = std::max(maxArea, stats.at<int>(i, cv::CC_STAT_AREA));
    if (maxArea < kMinLineArea) return l;

    cv::Mat clean = cv::Mat::zeros(mask.size(), CV_8UC1);
    for (int i = 1; i < n; ++i)
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= 0.2 * maxArea)
            clean.setTo(255, labels == i);

    const double area = static_cast<double>(cv::countNonZero(clean));
    std::vector<cv::Point> pts;
    cv::findNonZero(clean, pts);
    cv::Vec4f f;
    cv::fitLine(pts, f, cv::DIST_L2, 0, 0.01, 0.01);

    l.box = cv::boundingRect(clean);
    l.ok = true;
    l.vx = f[0]; l.vy = f[1]; l.x0 = f[2]; l.y0 = f[3];
    l.thickness = area / std::max(1, vertical ? l.box.height : l.box.width);
    return l;
}

Pos classify(const cv::Point& c, const Line& vLine, const Line& hLine)
{
    if (!vLine.ok || !hLine.ok) return Pos::Unknown;

    const double dx = c.x - vLine.xAt(c.y);   // 흰선(세로)에서 좌우로 떨어진 거리
    const double dy = c.y - hLine.yAt(c.x);   // 파란선(가로)에서 상하로 떨어진 거리

    const bool onV = std::fabs(dx) <= vLine.thickness * 0.5 + kLineMargin;
    const bool onH = std::fabs(dy) <= hLine.thickness * 0.5 + kLineMargin;

    if (onV && onH) return Pos::Center;
    if (onV)        return dy < 0 ? Pos::WhiteUp : Pos::WhiteDown;
    if (onH)        return dx < 0 ? Pos::BlueLeft : Pos::BlueRight;

    if (dx > 0 && dy < 0)  return Pos::Q1;   // 오른쪽 위
    if (dx < 0 && dy < 0)  return Pos::Q2;   // 왼쪽 위
    if (dx < 0 && dy >= 0) return Pos::Q3;   // 왼쪽 아래
    return Pos::Q4;                          // 오른쪽 아래
}

ConeResult findCone(const cv::Mat& mask, const cv::Mat& bgr, const Line& vLine, const Line& hLine)
{
    ConeResult r;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double best = 0;
    int idx = -1;
    for (size_t i = 0; i < contours.size(); ++i) {
        const double a = cv::contourArea(contours[i]);
        if (a > best) { best = a; idx = static_cast<int>(i); }
    }
    if (idx < 0 || best < kMinConeArea) return r;   // 콘 없음

    r.found  = true;
    r.box    = cv::boundingRect(contours[idx]) & cv::Rect(0, 0, bgr.cols, bgr.rows);
    r.center = (r.box.tl() + r.box.br()) / 2;
    r.pos    = classify(r.center, vLine, hLine);
    r.roi    = bgr(r.box).clone();                  // 바운딩박스 내부만 추출
    return r;
}

void drawCone(cv::Mat& img, const ConeResult& c, const cv::Scalar& color, const char* name)
{
    if (!c.found) return;
    cv::rectangle(img, c.box, color, 2);
    cv::circle(img, c.center, 3, color, -1);
    const std::string text = std::string(name) + ": " + posName(c.pos);
    cv::putText(img, text, cv::Point(c.box.x, std::max(15, c.box.y - 6)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
}

}  // namespace

Result process(const cv::Mat& bgr, const HsvRange ranges[TARGET_COUNT])
{
    Result res;
    if (bgr.empty()) return res;

    // 1) 가우시안 블러로 노이즈 완화 -> HSV 변환
    cv::Mat blurred, hsv;
    cv::GaussianBlur(bgr, blurred, cv::Size(5, 5), 0);
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

    // 2) 색상별 inRange + 열기(작은 점 제거) + 닫기(구멍 메우기)
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    for (int i = 0; i < TARGET_COUNT; ++i) {
        const HsvRange& r = ranges[i];
        cv::inRange(hsv,
                    cv::Scalar(r.hLow, r.sLow, r.vLow),
                    cv::Scalar(r.hHigh, r.sHigh, r.vHigh),
                    res.masks[i]);
        cv::morphologyEx(res.masks[i], res.masks[i], cv::MORPH_OPEN, kernel);
        cv::morphologyEx(res.masks[i], res.masks[i], cv::MORPH_CLOSE, kernel);
    }

    // 3) 기준선 검출 (흰선=세로, 파란선=가로)
    const Line vLine = analyzeLine(res.masks[WHITE], true);
    const Line hLine = analyzeLine(res.masks[BLUE], false);
    res.hasWhiteLine = vLine.ok;
    res.hasBlueLine  = hLine.ok;

    // 4) 콘 검출 + 위치 판별
    res.neon   = findCone(res.masks[NEON],   bgr, vLine, hLine);
    res.orange = findCone(res.masks[ORANGE], bgr, vLine, hLine);

    // 5) 결과 영상 그리기
    res.annotated = bgr.clone();
    if (vLine.ok) {                                   // 흰선: 노란색
        const int y1 = vLine.box.y, y2 = vLine.box.y + vLine.box.height - 1;
        cv::line(res.annotated, cv::Point(cvRound(vLine.xAt(y1)), y1),
                 cv::Point(cvRound(vLine.xAt(y2)), y2), cv::Scalar(0, 255, 255), 3);
    }
    if (hLine.ok) {                                   // 파란선: 분홍색
        const int x1 = hLine.box.x, x2 = hLine.box.x + hLine.box.width - 1;
        cv::line(res.annotated, cv::Point(x1, cvRound(hLine.yAt(x1))),
                 cv::Point(x2, cvRound(hLine.yAt(x2))), cv::Scalar(255, 0, 255), 3);
    }
    drawCone(res.annotated, res.neon,   cv::Scalar(255, 0, 0), "neon");
    drawCone(res.annotated, res.orange, cv::Scalar(0, 0, 255), "orange");

    return res;
}

const char* posName(Pos p)
{
    switch (p) {
    case Pos::None:      return "None";
    case Pos::Unknown:   return "Unknown";
    case Pos::Q1:        return "Q1";
    case Pos::Q2:        return "Q2";
    case Pos::Q3:        return "Q3";
    case Pos::Q4:        return "Q4";
    case Pos::WhiteUp:   return "White-Up";
    case Pos::WhiteDown: return "White-Down";
    case Pos::BlueLeft:  return "Blue-Left";
    case Pos::BlueRight: return "Blue-Right";
    case Pos::Center:    return "Center";
    }
    return "?";
}

const char* posNameKo(Pos p)
{
    switch (p) {
    case Pos::None:      return "없음";
    case Pos::Unknown:   return "판별 불가 (기준선 미검출)";
    case Pos::Q1:        return "1사분면";
    case Pos::Q2:        return "2사분면";
    case Pos::Q3:        return "3사분면";
    case Pos::Q4:        return "4사분면";
    case Pos::WhiteUp:   return "위쪽 하얀선 위";
    case Pos::WhiteDown: return "아래쪽 하얀선 위";
    case Pos::BlueLeft:  return "왼쪽 파란선 위";
    case Pos::BlueRight: return "오른쪽 파란선 위";
    case Pos::Center:    return "중앙";
    }
    return "?";
}

}  // namespace vision