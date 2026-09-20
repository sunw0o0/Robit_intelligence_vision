#include "vision.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace vision {

namespace {

constexpr double kMinLineArea   = 200.0;   // 이보다 작으면 선이 없다고 판단
constexpr double kMinConeArea   = 300.0;   // 이보다 작으면 노이즈로 보고 콘이 없다고 판단
constexpr double kLineMargin    = 5.0;     // "선 위" 판정 여유 (픽셀)
constexpr double kMinElongation = 2.5;     // 길쭉함(긴 축/짧은 축) 이보다 작으면 선이 아님
constexpr double kMinPerpSin    = 0.7071;  // 흰선과 파란선이 이루는 각: 45° 이상 벌어져야 함

// 기준선: fitLine으로 기울어진 선도 처리. 가로/세로는 고정이 아니다.
struct Line
{
    bool ok = false;
    double vx = 0, vy = 0, x0 = 0, y0 = 0;   // 직선의 방향벡터와 지나는 점
    double thickness = 0;                    // 선 굵기 (면적 / 길이)
    double tMin = 0, tMax = 0;               // 선의 양 끝: 점 = (x0,y0) + t*(vx,vy)

    cv::Point endpoint(double t) const {
        return cv::Point(cvRound(x0 + t * vx), cvRound(y0 + t * vy));
    }
    // 선에서 점까지의 수직 거리 (기울어진 선도 정확)
    double perp(const cv::Point& c) const {
        return std::fabs((c.x - x0) * vy - (c.y - y0) * vx);
    }
};

// 덩어리가 길쭉하면 true, 긴 축의 각도를 theta(라디안)로 돌려준다
bool lineShape(const cv::Mat& comp, double& theta)
{
    const cv::Moments m = cv::moments(comp, true);
    if (m.m00 <= 0) return false;
    const double a = m.mu20 / m.m00;      // x 분산
    const double b = m.mu11 / m.m00;
    const double c = m.mu02 / m.m00;      // y 분산
    const double d = std::sqrt(std::max(0.0, (a - c) * (a - c) / 4 + b * b));
    const double l1 = (a + c) / 2 + d, l2 = (a + c) / 2 - d;
    if (l1 <= 0) return false;
    const double ratio = (l2 < 1e-6) ? 1e9 : std::sqrt(l1 / l2);
    if (ratio < kMinElongation) return false;
    theta = 0.5 * std::atan2(2 * b, a - c);
    return true;
}

// keepOnlyLargest: 흰선처럼 1개짜리는 가장 큰 후보만, 파란선처럼 끊긴 조각들은 20% 이상 전부
// perpTo: 지정하면 그 선과 수직에 가까운 덩어리만 후보 (바닥 반사 같은 엉뚱한 덩어리 제거)
Line analyzeLine(const cv::Mat& mask, bool keepOnlyLargest, const Line* perpTo)
{
    Line l;

    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    const double refAngle = perpTo ? std::atan2(perpTo->vy, perpTo->vx) : 0.0;

    std::vector<int> cand;
    int maxArea = 0, best = -1;
    for (int i = 1; i < n; ++i) {
        const int a = stats.at<int>(i, cv::CC_STAT_AREA);
        if (a < kMinLineArea) continue;
        const cv::Rect r(stats.at<int>(i, cv::CC_STAT_LEFT),  stats.at<int>(i, cv::CC_STAT_TOP),
                         stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
        double theta = 0;
        if (!lineShape(labels(r) == i, theta)) continue;                       // 길쭉하지 않음
        if (perpTo && std::fabs(std::sin(theta - refAngle)) < kMinPerpSin) continue;  // 수직이 아님
        cand.push_back(i);
        if (a > maxArea) { maxArea = a; best = i; }
    }
    if (best < 0) return l;

    cv::Mat clean = cv::Mat::zeros(mask.size(), CV_8UC1);
    for (int i : cand) {
        const bool keep = keepOnlyLargest ? (i == best)
                                          : (stats.at<int>(i, cv::CC_STAT_AREA) >= 0.2 * maxArea);
        if (keep) clean.setTo(255, labels == i);
    }

    const double area = static_cast<double>(cv::countNonZero(clean));
    std::vector<cv::Point> pts;
    cv::findNonZero(clean, pts);
    cv::Vec4f f;
    cv::fitLine(pts, f, cv::DIST_L2, 0, 0.01, 0.01);

    l.ok = true;
    l.vx = f[0]; l.vy = f[1]; l.x0 = f[2]; l.y0 = f[3];

    // 방향벡터 기준으로 선의 양 끝(길이)을 구한다
    double tMin = 1e18, tMax = -1e18;
    for (const cv::Point& p : pts) {
        const double t = (p.x - l.x0) * l.vx + (p.y - l.y0) * l.vy;
        tMin = std::min(tMin, t);
        tMax = std::max(tMax, t);
    }
    l.tMin = tMin; l.tMax = tMax;
    l.thickness = area / std::max(1.0, tMax - tMin);
    return l;
}

// 흰선 = x축, 파란선 = y축 인 좌표계
struct Axes
{
    bool ok = false;
    cv::Point2d origin;      // 두 선의 교점
    cv::Point2d ex, ey;      // +x, +y 방향 단위벡터 (화면 좌표)
    double det = 0;
};

Axes makeAxes(const Line& white, const Line& blue)
{
    Axes ax;
    if (!white.ok || !blue.ok) return ax;

    cv::Point2d ex(white.vx, white.vy);
    cv::Point2d ey(blue.vx, blue.vy);

    // 두 직선의 교점: white.p + t*ex = blue.p + s*ey
    const double d = ex.x * (-ey.y) - (-ey.x) * ex.y;          // det [ex, -ey]
    if (std::fabs(d) < 1e-6) return ax;                        // 평행
    const double rx = blue.x0 - white.x0, ry = blue.y0 - white.y0;
    const double t = (rx * (-ey.y) - (-ey.x) * ry) / d;
    ax.origin = cv::Point2d(white.x0 + t * ex.x, white.y0 + t * ex.y);

    // +x 방향: 가로에 가까우면 오른쪽, 세로에 가까우면 위쪽
    if (std::fabs(ex.x) >= std::fabs(ex.y)) { if (ex.x < 0) ex = -ex; }
    else                                    { if (ex.y > 0) ex = -ex; }

    // +y 방향: +x에서 반시계 90° (화면 y가 아래로 증가하므로 det < 0 이어야 함)
    if (ex.x * ey.y - ey.x * ex.y > 0) ey = -ey;

    ax.ex = ex; ax.ey = ey;
    ax.det = ex.x * ey.y - ey.x * ex.y;
    ax.ok = std::fabs(ax.det) > 0.2;   // 두 선이 너무 평행하면 실패
    return ax;
}

Pos classify(const cv::Point& c, const Axes& ax, const Line& white, const Line& blue,
             int& whiteSide, int& blueSide)
{
    whiteSide = blueSide = 0;
    if (!ax.ok) return Pos::Unknown;

    // c = origin + a*ex + b*ey 를 풀어서 (a, b) = 새 좌표계의 (x, y)
    const double px = c.x - ax.origin.x, py = c.y - ax.origin.y;
    const double a = (px * ax.ey.y - ax.ey.x * py) / ax.det;
    const double b = (ax.ex.x * py - px * ax.ex.y) / ax.det;

    const bool onX = white.perp(c) <= white.thickness * 0.5 + kLineMargin;   // 흰선 위
    const bool onY = blue.perp(c)  <= blue.thickness  * 0.5 + kLineMargin;   // 파란선 위

    // 하얀선 기준 위/아래(b의 부호), 파란선 기준 오른쪽/왼쪽(a의 부호)
    whiteSide = onX ? 0 : (b > 0 ? +1 : -1);
    blueSide  = onY ? 0 : (a > 0 ? +1 : -1);

    if (onX && onY) return Pos::Center;
    if (onX)        return a >= 0 ? Pos::WhitePlus : Pos::WhiteMinus;
    if (onY)        return b >= 0 ? Pos::BluePlus  : Pos::BlueMinus;

    if (a > 0 && b > 0) return Pos::Q1;
    if (a < 0 && b > 0) return Pos::Q2;
    if (a < 0 && b < 0) return Pos::Q3;
    return Pos::Q4;
}

ConeResult findCone(const cv::Mat& mask, const cv::Mat& bgr, const Axes& ax,
                    const Line& white, const Line& blue)
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
    r.pos    = classify(r.center, ax, white, blue, r.whiteSide, r.blueSide);
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

    // 3) 기준선 검출 (방향은 자동 판단)
    //    파란선을 먼저 찾고, 흰선은 파란선과 수직에 가까운 덩어리만 인정한다.
    const Line blueLine  = analyzeLine(res.masks[BLUE],  false, nullptr);
    const Line whiteLine = analyzeLine(res.masks[WHITE], true,  blueLine.ok ? &blueLine : nullptr);
    res.hasWhiteLine = whiteLine.ok;
    res.hasBlueLine  = blueLine.ok;

    // 4) 좌표계(흰선=x축, 파란선=y축) + 콘 검출 + 위치 판별
    const Axes axes = makeAxes(whiteLine, blueLine);
    res.neon   = findCone(res.masks[NEON],   bgr, axes, whiteLine, blueLine);
    res.orange = findCone(res.masks[ORANGE], bgr, axes, whiteLine, blueLine);

    // 5) 결과 영상 그리기 (흰선: 노란색, 파란선: 분홍색)
    res.annotated = bgr.clone();
    auto drawLine = [&](const Line& l, const cv::Scalar& color) {
        if (!l.ok) return;
        cv::line(res.annotated, l.endpoint(l.tMin), l.endpoint(l.tMax), color, 3);
    };
    drawLine(whiteLine, cv::Scalar(0, 255, 255));
    drawLine(blueLine,  cv::Scalar(255, 0, 255));
    if (axes.ok) {   // +x, +y 방향 화살표 (좌표계가 맞게 잡혔는지 눈으로 확인용)
        const cv::Point o(cvRound(axes.origin.x), cvRound(axes.origin.y));
        const cv::Point px(cvRound(o.x + 60 * axes.ex.x), cvRound(o.y + 60 * axes.ex.y));
        const cv::Point py(cvRound(o.x + 60 * axes.ey.x), cvRound(o.y + 60 * axes.ey.y));
        cv::arrowedLine(res.annotated, o, px, cv::Scalar(0, 255, 255), 2, cv::LINE_AA, 0, 0.25);
        cv::arrowedLine(res.annotated, o, py, cv::Scalar(255, 0, 255), 2, cv::LINE_AA, 0, 0.25);
        cv::putText(res.annotated, "x", px + cv::Point(4, 4), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        cv::putText(res.annotated, "y", py + cv::Point(4, 4), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
    }
    drawCone(res.annotated, res.neon,   cv::Scalar(255, 0, 0), "neon");
    drawCone(res.annotated, res.orange, cv::Scalar(0, 0, 255), "orange");

    return res;
}

const char* whiteSideName(int side)
{
    return side > 0 ? "up" : (side < 0 ? "down" : "on line");
}

const char* blueSideName(int side)
{
    return side > 0 ? "right" : (side < 0 ? "left" : "on line");
}

const char* posName(Pos p)
{
    switch (p) {
    case Pos::None:       return "None";
    case Pos::Unknown:    return "Unknown";
    case Pos::Q1:         return "Q1";
    case Pos::Q2:         return "Q2";
    case Pos::Q3:         return "Q3";
    case Pos::Q4:         return "Q4";
    case Pos::WhitePlus:  return "White-Right";
    case Pos::WhiteMinus: return "White-Left";
    case Pos::BluePlus:   return "Blue-Up";
    case Pos::BlueMinus:  return "Blue-Down";
    case Pos::Center:     return "Center";
    }
    return "?";
}

const char* posNameKo(Pos p)
{
    switch (p) {
    case Pos::None:       return "없음";
    case Pos::Unknown:    return "판별 불가 (기준선 미검출)";
    case Pos::Q1:         return "1사분면";
    case Pos::Q2:         return "2사분면";
    case Pos::Q3:         return "3사분면";
    case Pos::Q4:         return "4사분면";
    case Pos::WhitePlus:  return "하얀선 위 (파란선 오른쪽)";
    case Pos::WhiteMinus: return "하얀선 위 (파란선 왼쪽)";
    case Pos::BluePlus:   return "파란선 위 (하얀선 위쪽)";
    case Pos::BlueMinus:  return "파란선 위 (하얀선 아래쪽)";
    case Pos::Center:     return "중앙";
    }
    return "?";
}

}  // namespace vision