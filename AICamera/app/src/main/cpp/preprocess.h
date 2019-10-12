#ifndef AICAMERA_PREPROCESS_H
#define AICAMERA_PREPROCESS_H

#include <jni.h>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"


std::vector<float> readImg(std::string path){
    assert(access( path.c_str(), F_OK ) != -1);
    cv::Mat img = cv::imread(path);
    assert(img.elemSize() > 0);
    cv::resize(img,img,cv::Size(300,300));
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
    std::vector<float> data(1 * 3 * 300 * 300);

    int predHeight = img.rows;
    int predWidth = img.cols;
    int size = predHeight * predWidth;
    // 注意imread读入的图像格式是unsigned char，如果你的网络输入要求是float的话，下面的操作就不对了。
    for (auto i=0; i<predHeight; i++) {
        //printf("+\n");
        for (auto j=0; j<predWidth; j++) {
            data[i * predWidth + j + 0*size] = (float)img.data[(i*predWidth + j) * 3 + 0];
            data[i * predWidth + j + 1*size] = (float)img.data[(i*predWidth + j) * 3 + 1];
            data[i * predWidth + j + 2*size] = (float)img.data[(i*predWidth + j) * 3 + 2];
        }
    }
    return data;
}

std::vector<float> readImg1(cv::Mat& img){
    cv::resize(img,img,cv::Size(300,300));
    cv::cvtColor(img,img,cv::COLOR_RGBA2RGB);
    cv::subtract(img,127,img);
    cv::divide(128,img,img);
    std::vector<float> data(1 * 3 * 300 * 300);

    int predHeight = img.rows;
    int predWidth = img.cols;
    int size = predHeight * predWidth;
    // 注意imread读入的图像格式是unsigned char，如果你的网络输入要求是float的话，下面的操作就不对了。
    for (auto i=0; i<predHeight; i++) {
        //printf("+\n");
        for (auto j=0; j<predWidth; j++) {
            data[i * predWidth + j + 0*size] = (float)img.data[(i*predWidth + j) * 3 + 0];
            data[i * predWidth + j + 1*size] = (float)img.data[(i*predWidth + j) * 3 + 1];
            data[i * predWidth + j + 2*size] = (float)img.data[(i*predWidth + j) * 3 + 2];
        }
    }
    return data;
}

#endif //AICAMERA_PREPROCESS_H
