#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <QTimer>
#include <QImage>
#include <QLabel>
#include <QSlider>
#include <QMainWindow>

#include "vision.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private:
    bool openCamera(int index);
    void processFrame();                          // 영상 처리 + 라벨 갱신
    void selectTarget(vision::Target t);          // 라디오 선택 -> 슬라이더를 그 색의 값으로
    void syncSliders();
    void showMat(QLabel *label, const cv::Mat &m);

    Ui::MainWindow *ui;
    cv::VideoCapture cap;
    QTimer *timer = nullptr;

    cv::Mat lastFrame;                            // 마지막 카메라 프레임
    vision::HsvRange ranges[vision::TARGET_COUNT];
    vision::Target current = vision::WHITE;       // 지금 슬라이더로 조절 중인 색

    // 슬라이더/값 라벨 (순서: hLow, hHigh, sLow, sHigh, vLow, vHigh)
    QSlider *sl[6]  = {};
    QLabel  *val[6] = {};
    QLabel  *roiNeon   = nullptr;                 // 콘 바운딩박스 ROI 창
    QLabel  *roiOrange = nullptr;

private slots:
    void updateFrame();
    void onSliderChanged();
};
#endif // MAINWINDOW_H