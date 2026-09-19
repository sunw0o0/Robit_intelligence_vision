#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDebug>
#include <QMessageBox>
#include <QSignalBlocker>
#include <QStatusBar>

bool MainWindow::openCamera(int index)
{
    cap.open(index, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        qWarning() << "camera open failed at index" << index;
        return false;
    }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat test;
    if (!cap.read(test) || test.empty()) {
        qWarning() << "camera opened but no frame at index" << index;
        cap.release();
        return false;
    }
    qDebug() << "camera OK, index" << index << "size" << test.cols << "x" << test.rows;
    return true;
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    timer = nullptr;

    // 기본 HSV 범위 (환경에 맞게 슬라이더로 조절해서 사용)
    //                        hLow hHigh sLow sHigh vLow vHigh
    ranges[vision::WHITE]  = {   0, 179,    0,   15, 225, 255 };
    ranges[vision::BLUE]   = {  95, 130,  100,  255,  50, 255 };
    ranges[vision::NEON]   = {  30,  65,   80,  255, 100, 255 };
    ranges[vision::ORANGE] = {   0,  20,  120,  255, 120, 255 };

    // 라디오 버튼: 어떤 색을 조절할지 선택
    connect(ui->radioWhite,  &QRadioButton::toggled, this, [this](bool on) { if (on) selectTarget(vision::WHITE); });
    connect(ui->radioBlue,   &QRadioButton::toggled, this, [this](bool on) { if (on) selectTarget(vision::BLUE); });
    connect(ui->radioNeon,   &QRadioButton::toggled, this, [this](bool on) { if (on) selectTarget(vision::NEON); });
    connect(ui->radioOrange, &QRadioButton::toggled, this, [this](bool on) { if (on) selectTarget(vision::ORANGE); });

    // 슬라이더 6개 매핑 (Designer의 기본 이름 -> 의미)
    sl[0] = ui->horizontalSlider_2; val[0] = ui->label_12;   // H Low
    sl[1] = ui->horizontalSlider;   val[1] = ui->label_11;   // H High
    sl[2] = ui->horizontalSlider_4; val[2] = ui->label_14;   // S Low
    sl[3] = ui->horizontalSlider_3; val[3] = ui->label_13;   // S High
    sl[4] = ui->horizontalSlider_6; val[4] = ui->label_16;   // V Low
    sl[5] = ui->horizontalSlider_5; val[5] = ui->label_15;   // V High

    const int maxv[6] = { 179, 179, 255, 255, 255, 255 };
    for (int i = 0; i < 6; ++i) {
        sl[i]->setRange(0, maxv[i]);
        connect(sl[i], &QSlider::valueChanged, this, &MainWindow::onSliderChanged);
    }

    // ROI 창 (UI에 자리가 없으니 별도 창으로)
    auto makeRoi = [this](const char *title) {
        auto *l = new QLabel(this, Qt::Window);
        l->setWindowTitle(title);
        l->setFixedSize(200, 200);
        l->setAlignment(Qt::AlignCenter);
        return l;
    };
    roiNeon   = makeRoi("Neon ROI");
    roiOrange = makeRoi("Orange ROI");

    ui->radioWhite->setChecked(true);
    syncSliders();

    if (!openCamera(32)) {          // v4l2-ctl로 확인한 Spedal 번호
        QMessageBox::warning(this, "카메라 오류",
                             "카메라를 열 수 없습니다.\n터미널의 메시지를 확인하세요.");
        return;
    }

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::updateFrame);
    timer->start(30);
}

MainWindow::~MainWindow()
{
    if (timer) timer->stop();
    if (cap.isOpened()) cap.release();
    delete ui;
}

void MainWindow::updateFrame()
{
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) return;

    lastFrame = frame;
    processFrame();
}

// 영상 처리(vision::process) 결과를 라벨에 표시
void MainWindow::processFrame()
{
    if (lastFrame.empty()) return;

    const vision::Result r = vision::process(lastFrame, ranges);

    showMat(ui->Usb_Cam,     lastFrame);                       // 원본
    showMat(ui->Find_object, r.annotated);                     // 선/콘 검출 결과
    showMat(ui->labelNeon,   r.masks[vision::NEON]);
    showMat(ui->labelWhite,  r.masks[vision::WHITE]);
    showMat(ui->labelBlue,   r.masks[vision::BLUE]);
    showMat(ui->labelOrange, r.masks[vision::ORANGE]);

    // ROI: 콘이 있을 때만 창을 띄움
    auto roi = [this](QLabel *l, const cv::Mat &m) {
        if (m.empty()) { l->hide(); return; }
        showMat(l, m);
        if (l->isHidden()) l->show();
    };
    roi(roiNeon,   r.neon.roi);
    roi(roiOrange, r.orange.roi);

    // 위치 표시: 좁은 라벨엔 영어, 상태바엔 한글
    ui->label_7->setText(QString("neon: %1").arg(vision::posName(r.neon.pos)));
    ui->label_8->setText(QString("orange: %1").arg(vision::posName(r.orange.pos)));
    statusBar()->showMessage(
        QString("Neon: %1   |   Orange: %2")
            .arg(QString::fromUtf8(vision::posNameKo(r.neon.pos)),
                 QString::fromUtf8(vision::posNameKo(r.orange.pos))));
}

// ---------------------------------------------------------------- HSV 슬라이더

void MainWindow::selectTarget(vision::Target t)
{
    current = t;
    syncSliders();
}

// 선택된 색의 저장값을 슬라이더/값 라벨에 반영 (신호를 막아서 불필요한 재처리 방지)
void MainWindow::syncSliders()
{
    const vision::HsvRange &r = ranges[current];
    const int v[6] = { r.hLow, r.hHigh, r.sLow, r.sHigh, r.vLow, r.vHigh };
    for (int i = 0; i < 6; ++i) {
        const QSignalBlocker blocker(sl[i]);
        sl[i]->setValue(v[i]);
        val[i]->setNum(v[i]);
    }
}

void MainWindow::onSliderChanged()
{
    vision::HsvRange &r = ranges[current];
    int *f[6] = { &r.hLow, &r.hHigh, &r.sLow, &r.sHigh, &r.vLow, &r.vHigh };
    for (int i = 0; i < 6; ++i) {
        *f[i] = sl[i]->value();
        val[i]->setNum(*f[i]);
    }
    processFrame();   // 슬라이더를 움직이면 마지막 프레임에 바로 반영
}

// ---------------------------------------------------------------- 표시

void MainWindow::showMat(QLabel *label, const cv::Mat &m)
{
    if (m.empty()) { label->clear(); return; }

    QImage img;
    if (m.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(m, rgb, cv::COLOR_BGR2RGB);   // OpenCV는 BGR, Qt는 RGB
        img = QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                     QImage::Format_RGB888).copy();
    } else if (m.type() == CV_8UC1) {              // 이진 마스크
        img = QImage(m.data, m.cols, m.rows, static_cast<int>(m.step),
                     QImage::Format_Grayscale8).copy();
    } else {
        return;
    }
    label->setPixmap(QPixmap::fromImage(img).scaled(
        label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}