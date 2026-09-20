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

// 콘 위치: 흰선 = x축, 파란선 = y축 (선이 가로든 세로든 상관없음)
//  +x 방향: 흰선이 가로에 가까우면 오른쪽, 세로에 가까우면 위쪽
//  +y 방향: +x에서 반시계로 90° (수학 좌표계와 같은 방향)
//  Q1 = (+x,+y), Q2 = (-x,+y), Q3 = (-x,-y), Q4 = (+x,-y)
enum class Pos {
    None,        // 콘이 없음
    Unknown,     // 콘은 있는데 기준선 2개가 검출되지 않음
    Q1, Q2, Q3, Q4,
    WhitePlus,   // 흰선 위에 있고, 파란선의 오른쪽(+x쪽)
    WhiteMinus,  // 흰선 위에 있고, 파란선의 왼쪽(-x쪽)
    BluePlus,    // 파란선 위에 있고, 흰선의 위쪽(+y쪽)
    BlueMinus,   // 파란선 위에 있고, 흰선의 아래쪽(-y쪽)
    Center       // 원점(두 선이 만나는 곳)
};

struct ConeResult {
    bool found = false;
    cv::Rect box;          // 바운딩박스
    cv::Point center;      // 바운딩박스 중심
    Pos pos = Pos::None;
    int whiteSide = 0;     // 흰선 기준: +1 = up(+y쪽), -1 = down(-y쪽), 0 = 흰선 위
    int blueSide  = 0;     // 파란선 기준: +1 = right(+x쪽), -1 = left(-x쪽), 0 = 파란선 위
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

const char* whiteSideName(int side);   // "up" / "down" / "on line"
const char* blueSideName(int side);    // "right" / "left" / "on line"
const char* posName(Pos p);     // 영어 (영상 위에 글자를 그릴 때 사용)
const char* posNameKo(Pos p);   // 한글 (UTF-8, QLabel 표시용)

}  // namespace vision