#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <opencv2/core.hpp>
#include <QLabel>
#include <QMainWindow>
#include <QString>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(const QString &imagePath, QWidget *parent = nullptr);
    ~MainWindow() override;

private:
    void showMat(QLabel *label, const cv::Mat &m);

    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
