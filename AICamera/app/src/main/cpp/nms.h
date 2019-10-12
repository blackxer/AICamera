#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#define alog(...) __android_log_print(ANDROID_LOG_ERROR, "AICamera", __VA_ARGS__);
using namespace std;

typedef struct Bbox {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float area;
} Bbox;

//升序排列
bool cmpScore(Bbox lsh, Bbox rsh) {
    if (lsh.score < rsh.score)
        return true;
    else
        return false;
}

void nms(vector<Bbox> &boundingBox_, const float overlap_threshold, const float confidence, string modelname = "Union"){
    alog("boxes size: %d", boundingBox_.size());
    if(boundingBox_.empty()){
        return;
    }
    //对各个候选框根据score的大小进行升序排列
    sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
    //删除得分小于阈值的候选框
    for(auto itr = boundingBox_.cbegin(); itr != boundingBox_.cend();){
        if(confidence > itr.base()->score){
            boundingBox_.erase(itr);
        }else{
            break;
        }
    }
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    vector<int> vPick;
    int nPick = 0;
    multimap<float, int> vScores;   //存放升序排列后的score和对应的序号
    const int num_boxes = boundingBox_.size();
    vPick.resize(num_boxes);
    for (int i = 0; i < num_boxes; ++i){
        vScores.insert(pair<float, int>(boundingBox_[i].score, i));
    }
    alog("confidence size: %d", vScores.size());
    while(vScores.size() > 0){
        int last = vScores.rbegin()->second;  //反向迭代器，获得vScores序列的最后那个序列号
        vPick[nPick] = last;
        nPick += 1;
        alog("%d %d", nPick, vScores.size());
        for (multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();){
            int it_idx = it->second;
            maxX = max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
            maxY = max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
            minX = min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
            minY = min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
            //转换成了两个边界框相交区域的边长
            maxX = ((minX-maxX)>0)? (minX-maxX) : 0;
            maxY = ((minY-maxY)>0)? (minY-maxY) : 0;
            //求交并比IOU

            IOU = (maxX * maxY)/(boundingBox_.at(it_idx).area + boundingBox_.at(last).area - maxX * maxY + 0.000001);
            alog("iou: %f", IOU);
            if(IOU > overlap_threshold){
                it = vScores.erase(it);    //删除交并比大于阈值的候选框,erase返回删除元素的下一个元素
            }else{
                it++;
            }
        }
    }
    vPick.resize(nPick);
    vector<Bbox> tmp_;
    tmp_.resize(nPick);
    for(int i = 0; i < nPick; i++){
        tmp_[i] = boundingBox_[vPick[i]];
    }
    boundingBox_ = tmp_;
}


template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {
    // 初始化索引向量
    std::vector<size_t> idx(v.size());
    //使用iota对向量赋0~？的连续值
    std::iota(idx.begin(), idx.end(), 0);
    // 通过比较v的值对索引idx进行排序
    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    return idx;
}

//原文链接：https://blog.csdn.net/qq_34719188/article/details/89672787
//原文链接：https://blog.csdn.net/qq_32582681/article/details/81352758



////!
////! \brief Performs non maximum suppression on final bounding boxes
////!
//std::vector<int> nonMaximumSuppression(std::vector<std::pair<float, int>>& scoreIndex, float* bbox,
//                                       const int classNum, const int numClasses, const float nmsThreshold)
//{
//    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
//        if (x1min > x2min)
//        {
//            std::swap(x1min, x2min);
//            std::swap(x1max, x2max);
//        }
//        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
//    };
//
//    auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
//        float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
//        float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
//        float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
//        float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
//        float overlap2D = overlapX * overlapY;
//        float u = area1 + area2 - overlap2D;
//        return u == 0 ? 0 : overlap2D / u;
//    };
//
//    std::vector<int> indices;
//    for (auto i : scoreIndex)
//    {
//        const int idx = i.second;
//        bool keep = true;
//        for (unsigned k = 0; k < indices.size(); ++k)
//        {
//            if (keep)
//            {
//                const int kept_idx = indices[k];
//                float overlap = computeIoU(
//                        &bbox[(idx * numClasses + classNum) * 4], &bbox[(kept_idx * numClasses + classNum) * 4]);
//                keep = overlap <= nmsThreshold;
//            }
//            else
//            {
//                break;
//            }
//        }
//        if (keep)
//        {
//            indices.push_back(idx);
//        }
//    }
//    return indices;
//}
////原文链接：https://blog.csdn.net/xiangxianghehe/article/details/97244449