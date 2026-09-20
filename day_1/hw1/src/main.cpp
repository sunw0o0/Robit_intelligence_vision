#include "mainwindow.h"

#include <QApplication>
#include <QDir>
#include <QFileDialog>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // 사용법: ros2 run vision_day1_hw1 vision_day1_hw1 [이미지경로]
    // 경로를 안 주면 파일 선택 창이 뜬다.
    QString path = a.arguments().value(1);
    if (path.isEmpty()) {
        path = QFileDialog::getOpenFileName(nullptr, "공 사진 선택", QDir::homePath(),
                                            "Images (*.png *.jpg *.jpeg *.bmp)");
        if (path.isEmpty()) return 0;
    }

    MainWindow w(path);
    w.show();
    return a.exec();
}
