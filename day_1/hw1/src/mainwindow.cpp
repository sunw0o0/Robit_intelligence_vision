#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "hw1.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <QDebug>
#include <QDir>
#include <QFileInfo>
#include <QMessageBox>
#include <QStatusBar>

namespace {
constexpr int kKernel = 9;   // 가우시안 커널 크기 (홀수). 5, 9, 15 로 바꿔서 비교해 보기
}

MainWindow::MainWindow(const QString &imagePath, QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    const cv::Mat img = cv::imread(imagePath.toStdString());
    if (img.empty()) {
        QMessageBox::warning(this, "이미지 오류", "이미지를 열 수 없습니다:\n" + imagePath);
        return;
    }

    const hw1::Result r = hw1::run(img, kKernel);

    // 왼쪽: 가우시안 O (블러 후 inRange), 오른쪽: 가우시안 X (블러 없이 inRange)
    showMat(ui->gaussia_O,  r.compositeBlur);   // Designer에서 이름을 gaussian_O 로 고쳤다면 여기도 gaussian_O
    showMat(ui->gaussian_X, r.compositeRaw);

    // 결과 저장 (이미지가 있는 폴더 아래 hw1_out)
    const QString outDir = QFileInfo(imagePath).absoluteDir().filePath("hw1_out");
    const bool saved = hw1::saveAll(img, r, outDir.toStdString(), kKernel);

    // 잡음 점 개수를 3색 합계로 요약
    int tinyRaw = 0, tinyBlur = 0;
    for (int i = 0; i < hw1::BALL_COUNT; ++i) {
        tinyRaw += r.statsRaw[i].tinyBlobs;   tinyBlur += r.statsBlur[i].tinyBlobs;
        qInfo().noquote() << QString("%1: 작은 덩어리 %2 -> %3 (블러 전 -> 후)")
                                 .arg(hw1::ballName(i))
                                 .arg(r.statsRaw[i].tinyBlobs).arg(r.statsBlur[i].tinyBlobs);
    }
    statusBar()->showMessage(
        QString("커널 %1x%1 | 잡음 점 %2 -> %3개 (블러 전 -> 후)%4")
            .arg(kKernel).arg(tinyRaw).arg(tinyBlur)
            .arg(saved ? QString("  |  저장: %1").arg(outDir) : QString("  |  저장 실패")));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::showMat(QLabel *label, const cv::Mat &m)
{
    if (m.empty()) { label->clear(); return; }

    QImage img;
    if (m.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(m, rgb, cv::COLOR_BGR2RGB);   // OpenCV는 BGR, Qt는 RGB
        img = QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                     QImage::Format_RGB888).copy();
    } else if (m.type() == CV_8UC1) {
        img = QImage(m.data, m.cols, m.rows, static_cast<int>(m.step),
                     QImage::Format_Grayscale8).copy();
    } else {
        return;
    }
    label->setPixmap(QPixmap::fromImage(img).scaled(
        label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}
