#pragma once
// Qt에 의존하지 않는 영상 처리 모듈 (나중에 ROS 노드에서도 그대로 재사용 가능)
#include <opencv2/core.hpp>

namespace vision {

enum Target { WHITE = 0, BLUE = 1, NEON = 2, ORANGE = 3, TARGET_COUNT = 4 };

// OpenCV HSV 범위: H 0~179, S/V 0~255
struct HsvRange {
    int hLow, hHigh;
    int sLow, sHigh;
    int vLow, vHigh;
};

// 콘 위치 10가지 (사분면은 수학 좌표계: Q1=오른쪽 위, Q2=왼쪽 위, Q3=왼쪽 아래, Q4=오른쪽 아래)
enum class Pos {
    None,        // 콘이 없음
    Unknown,     // 콘은 있는데 기준선(흰선/파란선)이 검출되지 않음
    Q1, Q2, Q3, Q4,
    WhiteUp,     // 위쪽 하얀선 위
    WhiteDown,   // 아래쪽 하얀선 위
    BlueLeft,    // 왼쪽 파란선 위
    BlueRight,   // 오른쪽 파란선 위
    Center       // 중앙(두 선이 만나는 곳)
};

struct ConeResult {
    bool found = false;
    cv::Rect box;          // 바운딩박스
    cv::Point center;      // 바운딩박스 중심
    Pos pos = Pos::None;
    cv::Mat roi;           // 바운딩박스 내부만 잘라낸 새 이미지
};

struct Result {
    cv::Mat annotated;              // 선과 박스를 그린 결과 영상
    cv::Mat masks[TARGET_COUNT];    // 색상별 이진 마스크
    bool hasWhiteLine = false;
    bool hasBlueLine = false;
    ConeResult neon;
    ConeResult orange;
};

Result process(const cv::Mat& bgr, const HsvRange ranges[TARGET_COUNT]);

const char* posName(Pos p);     // 영어 (영상 위에 글자를 그릴 때 사용)
const char* posNameKo(Pos p);   // 한글 (UTF-8, QLabel 표시용)

}  // namespace vision