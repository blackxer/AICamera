#include <jni.h>
#include <string>
#include <algorithm>
#include "nms.h"

#define PROTOBUF_USE_DLLS 1
#define CAFFE2_USE_LITE_PROTO 1
#include <caffe2/predictor/predictor.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/timer.h>

#include "caffe2/core/init.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <opencv2/core/base.hpp>
#include <opencv2/opencv.hpp>
//#include <altivec.h>

#include "classes.h"
#include "libyuv.h"
#include "preprocess.h"

#define IMG_H 300
#define IMG_W 300
#define IMG_C 3
#define MAX_DATA_SIZE IMG_H * IMG_W * IMG_C
#define alog(...) __android_log_print(ANDROID_LOG_ERROR, "AICamera", __VA_ARGS__);
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "error", __VA_ARGS__))
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, "debug", __VA_ARGS__))

static caffe2::NetDef _initNet, _predictNet;
static caffe2::Predictor *_predictor;
static float input_data[MAX_DATA_SIZE];
static caffe2::Workspace ws;
using namespace cv;

// A function to load the NetDefs from protobufs.
void loadToNetDef(AAssetManager* mgr, caffe2::NetDef* net, const char *filename) {
    AAsset* asset = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
    assert(asset != nullptr);
    const void *data = AAsset_getBuffer(asset);
    assert(data != nullptr);
    off_t len = AAsset_getLength(asset);
    assert(len != 0);
    if (!net->ParseFromArray(data, len)) {
        alog("Couldn't parse net from data.\n");
    }
    AAsset_close(asset);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_ufo_aicamera_MainActivity_initCaffe2(JNIEnv *env, jobject /* this */, jobject assetManager) {

//    jclass envcls = env->FindClass("android/os/Environment"); //获得类引用
////    if (envcls == nullptr) return 0;
//
//    //找到对应的类，该类是静态的返回值是File
//    jmethodID id = env->GetStaticMethodID(envcls, "getExternalStorageDirectory", "()Ljava/io/File;");
//
//    //调用上述id获得的方法，返回对象即File file=Enviroment.getExternalStorageDirectory()
//    //其实就是通过Enviroment调用 getExternalStorageDirectory()
//    jobject fileObj = env->CallStaticObjectMethod(envcls,id,"");
//
//    //通过上述方法返回的对象创建一个引用即File对象
//    jclass flieClass = env->GetObjectClass(fileObj); //或得类引用
//    //在调用File对象的getPath()方法获取该方法的ID，返回值为String 参数为空
//    jmethodID getpathId = env->GetMethodID(flieClass, "getPath", "()Ljava/lang/String;");
//    //调用该方法及最终获得存储卡的根目录
//    jstring pathStr = (jstring)env->CallObjectMethod(fileObj,getpathId,"");
//
//    const char* path = env->GetStringUTFChars(pathStr,NULL);
//    alog("path:%s",path);

    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    alog("Attempting to load protobuf netdefs...");
    loadToNetDef(mgr, &_initNet,   "vgg16-ssd_init_net.pb");
    loadToNetDef(mgr, &_predictNet,"vgg16-ssd_predict_net.pb");
    alog("done.");
    alog("Instantiating predictor...");
    _predictor = new caffe2::Predictor(_initNet, _predictNet);
    alog("done.")

}

float avg_fps = 0.0;
float total_fps = 0.0;
int iters_fps = 10;

//extern "C"
//JNIEXPORT jstring JNICALL
//Java_com_ufo_aicamera_MainActivity_predFromCaffe2(
//        JNIEnv *env,
//        jobject /* this */,
//        jbyteArray y, jbyteArray u, jbyteArray v,
//        jint width, jint height, jint y_row_stride,
//        jint uv_row_stride, jint uv_pixel_stride,
//        jint scale_width, jint scale_height, jint degree) {
//
//    if (!_predictor) {
//        return env->NewStringUTF("Loading...");
//    }
//
//    caffe2::TensorCPU input = caffe2::Tensor(1,caffe2::DeviceType::CPU);
//
//    input.Resize(std::vector<int>({1, IMG_C, IMG_H, IMG_W}));
//
//    alog("Reading data...");
//    std::vector<float> data = readImg("/sdcard/demo/image_00082.jpg");
//    alog("data size %d", data.size());
//    memcpy(input.mutable_data<float>(), data.data(), IMG_H * IMG_W * IMG_C * sizeof(float));
//    alog("done...");
//    caffe2::Predictor::TensorList input_vec{input};
//    caffe2::Predictor::TensorList output_vec;
//    caffe2::Timer t;
//    t.Start();
//    alog("Predicting...");
//    _predictor->operator()(input_vec, &output_vec);
//    float fps = 1000/t.MilliSeconds();
//    total_fps += fps;
//    avg_fps = total_fps / iters_fps;
//    total_fps -= avg_fps;
//    alog("done...");
//
//    caffe2::Timer t_p;
//    t_p.Start();
//    caffe2::TensorCPU scores = output_vec[0]; // 1*8732*3 ：  索引0为背景， 类别数为2
//    caffe2::TensorCPU boxes = output_vec[1];  // 1*8732*4 ：  x1,y1,x2,y2, 左上角和右下角
//    std::vector<float> score_1;  // 保存类别1的scores
//    std::vector<float> score_2;  // 保存类别2的scores
//    std::vector<Bbox> bbox_1;    // 保存类别1的bbox和score
//    std::vector<Bbox> bbox_2;    // 保存类别2的bbox和score
//    for(int i=0; i<scores.size(1); i++){
//        Bbox temp;
//        temp.x1 = boxes.template data<float>()[i*4];
//        temp.y1 = boxes.template data<float>()[i*4+1];
//        temp.x2 = boxes.template data<float>()[i*4+2];
//        temp.y2 = boxes.template data<float>()[i*4+3];
//        temp.score = scores.template data<float>()[i*3+1];
//        temp.area = (temp.x2 - temp.x1) * (temp.y2 - temp.y1);
//        bbox_1.push_back(temp);
//        temp.score = scores.template data<float>()[i*3+2];
//        bbox_2.push_back(temp);
//    }
//    alog("Start nms...");
//    alog("bbox_1 size: %d", bbox_1.size());
//    nms(bbox_1, 0.5, 0.5);
//    nms(bbox_2, 0.5, 0.5);
//    alog("done...");
//    float elapse = t_p.MilliSeconds()/1000;
//    alog("elapse: %f", elapse);
//
//    alog("Post processing");
//    cv::Mat ori_img = cv::imread("/sdcard/demo/image_00082.jpg");
//    int ori_width = ori_img.cols;
//    int ori_height = ori_img.rows;
//    for(int i=0; i<bbox_1.size(); i++){
//        int x1 = bbox_1[i].x1 * ori_width;
//        int y1 = bbox_1[i].y1 * ori_height;
//        int x2 = bbox_1[i].x2 * ori_width;
//        int y2 = bbox_1[i].y2 * ori_height;
//        float score = bbox_1[i].score;
//
//        cv::Point p1(x1, y1);
//        cv::Point p2(x2, y2);
//        cv::Scalar color(255,0,0);
//        cv::rectangle(ori_img,p1,p2,color,5);
//    }
//    for(int i=0; i<bbox_2.size(); i++){
//        int x1 = bbox_2[i].x1 * ori_width;
//        int y1 = bbox_2[i].y1 * ori_height;
//        int x2 = bbox_2[i].x2 * ori_width;
//        int y2 = bbox_2[i].y2 * ori_height;
//        float score = bbox_2[i].score;
//
//        cv::Point p1(x1, y1);
//        cv::Point p2(x2, y2);
//        cv::Scalar color(0,255,0);
//        cv::rectangle(ori_img,p1,p2,color,5);
//    }
//    cv::imwrite("/sdcard/demo/2.jpg", ori_img);
//    alog("done");
//    std::ostringstream stringStream;
//    stringStream << avg_fps << " FPS\n";
//
//
//    return env->NewStringUTF(stringStream.str().c_str());
//}





void BitmapToMat2(JNIEnv *env, jobject& bitmap, cv::Mat& mat, jboolean needUnPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void *pixels = 0;
    cv::Mat &dst = mat;

    try {
        LOGD("nBitmapToMat");
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                  info.format == ANDROID_BITMAP_FORMAT_RGB_565);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);
        dst.create(info.height, info.width, CV_8UC4);
        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOGD("nBitmapToMat: RGBA_8888 -> CV_8UC4");
            cv::Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (needUnPremultiplyAlpha) cv::cvtColor(tmp, dst, cv::COLOR_mRGBA2RGBA);
            else tmp.copyTo(dst);
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            LOGD("nBitmapToMat: RGB_565 -> CV_8UC4");
            cv::Mat tmp(info.height, info.width, CV_8UC2, pixels);
            cv::cvtColor(tmp, dst, cv::COLOR_BGR5652RGBA);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch (const cv::Exception &e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nBitmapToMat catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if (!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nBitmapToMat catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nBitmapToMat}");
        return;
    }
}

void BitmapToMat(JNIEnv *env, jobject& bitmap, cv::Mat& mat) {
    BitmapToMat2(env, bitmap, mat, false);
}

void MatToBitmap2
        (JNIEnv *env, cv::Mat& mat, jobject& bitmap, jboolean needPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void *pixels = 0;
    Mat &src = mat;

    try {
        LOGD("nMatToBitmap");
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                  info.format == ANDROID_BITMAP_FORMAT_RGB_565);
        CV_Assert(src.dims == 2 && info.height == (uint32_t) src.rows &&
                  info.width == (uint32_t) src.cols);
        CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);
        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (src.type() == CV_8UC1) {
                LOGD("nMatToBitmap: CV_8UC1 -> RGBA_8888");
                cvtColor(src, tmp, COLOR_GRAY2RGBA);
            } else if (src.type() == CV_8UC3) {
                LOGD("nMatToBitmap: CV_8UC3 -> RGBA_8888");
                cvtColor(src, tmp, COLOR_RGB2RGBA);
            } else if (src.type() == CV_8UC4) {
                LOGD("nMatToBitmap: CV_8UC4 -> RGBA_8888");
                if (needPremultiplyAlpha)
                    cvtColor(src, tmp, COLOR_RGBA2mRGBA);
                else
                    src.copyTo(tmp);
            }
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            if (src.type() == CV_8UC1) {
                LOGD("nMatToBitmap: CV_8UC1 -> RGB_565");
                cvtColor(src, tmp, COLOR_GRAY2BGR565);
            } else if (src.type() == CV_8UC3) {
                LOGD("nMatToBitmap: CV_8UC3 -> RGB_565");
                cvtColor(src, tmp, COLOR_RGB2BGR565);
            } else if (src.type() == CV_8UC4) {
                LOGD("nMatToBitmap: CV_8UC4 -> RGB_565");
                cvtColor(src, tmp, COLOR_RGBA2BGR565);
            }
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch (const cv::Exception &e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nMatToBitmap catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if (!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        LOGE("nMatToBitmap catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nMatToBitmap}");
        return;
    }
}

void MatToBitmap(JNIEnv *env, Mat& mat, jobject& bitmap) {
    MatToBitmap2(env, mat, bitmap, false);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_ufo_aicamera_MainActivity_predFromCaffe2(
        JNIEnv *env,
        jobject /* this */,
        jobject srcBitmap
       ) {

    if (!_predictor) {
        return env->NewStringUTF("Loading...");
    }

    caffe2::TensorCPU input = caffe2::Tensor(1,caffe2::DeviceType::CPU);

    input.Resize(std::vector<int>({1, IMG_C, IMG_H, IMG_W}));

    alog("Reading data...");
    Mat mat_image_src ;
    BitmapToMat(env,srcBitmap,mat_image_src);//图片转化成mat
    Mat temp_src_img = mat_image_src.clone();
    std::vector<float> data = readImg1(temp_src_img);
    alog("data size %d", data.size());
    memcpy(input.mutable_data<float>(), data.data(), IMG_H * IMG_W * IMG_C * sizeof(float));
    alog("done...");
    caffe2::Predictor::TensorList input_vec{input};
    caffe2::Predictor::TensorList output_vec;
    caffe2::Timer t;
    t.Start();
    alog("Predicting...");
    _predictor->operator()(input_vec, &output_vec);
    float fps = 1000/t.MilliSeconds();
    total_fps += fps;
    avg_fps = total_fps / iters_fps;
    total_fps -= avg_fps;
    alog("done...");

    caffe2::Timer t_p;
    t_p.Start();
    caffe2::TensorCPU scores = output_vec[0]; // 1*8732*3 ：  索引0为背景， 类别数为2
    caffe2::TensorCPU boxes = output_vec[1];  // 1*8732*4 ：  x1,y1,x2,y2, 左上角和右下角
    std::vector<float> score_1;  // 保存类别1的scores
    std::vector<float> score_2;  // 保存类别2的scores
    std::vector<Bbox> bbox_1;    // 保存类别1的bbox和score
    std::vector<Bbox> bbox_2;    // 保存类别2的bbox和score
    for(int i=0; i<scores.size(1); i++){
        Bbox temp;
        temp.x1 = boxes.template data<float>()[i*4];
        temp.y1 = boxes.template data<float>()[i*4+1];
        temp.x2 = boxes.template data<float>()[i*4+2];
        temp.y2 = boxes.template data<float>()[i*4+3];
        temp.score = scores.template data<float>()[i*3+1];
        temp.area = (temp.x2 - temp.x1) * (temp.y2 - temp.y1);
        bbox_1.push_back(temp);
        temp.score = scores.template data<float>()[i*3+2];
        bbox_2.push_back(temp);
    }
    alog("Start nms...");
    alog("bbox_1 size: %d", bbox_1.size());
    nms(bbox_1, 0.5, 0.5);
    nms(bbox_2, 0.5, 0.5);
    alog("done...");
    float elapse = t_p.MilliSeconds()/1000;
    alog("elapse: %f", elapse);

    alog("Post processing");
    int ori_width = mat_image_src.cols;
    int ori_height = mat_image_src.rows;
    for(int i=0; i<bbox_1.size(); i++){
        int x1 = bbox_1[i].x1 * ori_width;
        int y1 = bbox_1[i].y1 * ori_height;
        int x2 = bbox_1[i].x2 * ori_width;
        int y2 = bbox_1[i].y2 * ori_height;
        float score = bbox_1[i].score;

        cv::Point p1(x1, y1);
        cv::Point p2(x2, y2);
        cv::Scalar color(255,0,0);
        cv::rectangle(mat_image_src,p1,p2,color,5);
    }
    for(int i=0; i<bbox_2.size(); i++){
        int x1 = bbox_2[i].x1 * ori_width;
        int y1 = bbox_2[i].y1 * ori_height;
        int x2 = bbox_2[i].x2 * ori_width;
        int y2 = bbox_2[i].y2 * ori_height;
        float score = bbox_2[i].score;

        cv::Point p1(x1, y1);
        cv::Point p2(x2, y2);
        cv::Scalar color(0,255,0);
        cv::rectangle(mat_image_src,p1,p2,color,5);
    }
//    cv::imwrite("/sdcard/demo/2.jpg", mat_image_src);
    //第四步：转成java数组->更新
    MatToBitmap(env,mat_image_src,srcBitmap);//mat转成化图片
    alog("done");
    std::ostringstream stringStream;
    stringStream << avg_fps << " FPS\n";


    return env->NewStringUTF(stringStream.str().c_str());
}