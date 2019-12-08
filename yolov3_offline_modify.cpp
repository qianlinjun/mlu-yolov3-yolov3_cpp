/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other cv::Materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#if defined(USE_MLU) && defined(USE_OPENCV)

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "cnrt.h" // NOLINT
#include "blocking_queue.hpp"
#include "common_functions.hpp"



#include<stdlib.h>
#include <iostream>
#include <fstream>       //提供文件头文件
// #include <iomanip>       //C++输出精度控制需要
// #include <io.h>
// using namespace std;
// using namespace cv;
// #include <dirent.h> 
// #include <unistd.h>



using std::queue;
using std::string;
using std::stringstream;
using std::vector;

//batchnorm
//DEFINE_string(offlinemodel, "/home/Cambricon-Test/caffe/examples/offline/yolov3/whu-model/no-maxpool/new-head.cambricon",
//              "The prototxt file used to find net configuration");
DEFINE_string(offlinemodel, "/home/Cambricon-Test/caffe/yolov3-hd-sl_150000.cambricon",
              "The prototxt file used to find net configuration");
DEFINE_string(meanfile, "",
              "The mean file used to subtract from the input image.");
DEFINE_string(meanvalue, "",
            "If specified, can be one value or can be same as image channels"
            " - would subtract from the corresponding channel). Separated by ','."
            "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(images, "/home/Cambricon-Test/data", "The input file list");
DEFINE_string(labels, "/home/Cambricon-Test/caffe/examples/offline/yolov3/label.txt", "infocv::Mation about mapping from label to name");
DEFINE_string(outputdir, "/home/Cambricon-Test/result", "The directoy used to save output images");
DEFINE_string(txtoutputdir, "./txt", "The directoy used to save output images");
DEFINE_string(imgoutputdir, "./images", "The directoy used to save output images");
DEFINE_int32(slice_size,1000, "slice");
DEFINE_int32(overlap, 20, "plane bridge tank ship playground harbor");
DEFINE_int32(classnum, 6, "plane bridge tank ship playground harbor");
DEFINE_int32(netinputdim, 416, "net input size");

DEFINE_int32(anchornum, 3, "anchors per layer");
DEFINE_double(confidence, 0.001, "Only keep detections with scores  equal "
                                         "to or higher than the confidence.");
DEFINE_double(nmsthresh, 0.45, "Identify the optimal cell among all candidates "
                               " when the object lies in multiple cells of a grid");



DEFINE_int32(mludevice, 0, "set using mlu device number, default: 0");
DEFINE_int32(fix8, 0, "FP16 or FIX8, fix8 mode, default: 0");
DEFINE_string(logdir, "",
              "path to dump log file, to terminal "
              "stderr by default");
// DEFINE_int32(dump, 0, "0 or 1, dump output images or not.");

// DEFINE_string(bboxanchordir, "./bbox_anchor/", "The directoy used to read"
//                              " anchor_tensors and x_y_offset");


class Detector {
  public:
  Detector(const string& modelFile, const string& meanFile,
           const string& meanValues);
  ~Detector();

  vector<vector<vector<float>>> detect(
    const vector<cv::Mat>& images,
    const vector<vector<vector<float>>>& x_y_offsets,
    const vector<vector<vector<float>>>& anchor_tensors);
  int getBatchSize() { return batchSize; }
  int inputDim() { return inputShape[2]; }
  float mluTime()  { return mlu_time; }
  void readImages(queue<string>* imagesQueue, int inputNumber,
                  vector<cv::Mat>* images, vector<string>* imageNames);

  private:
  void setMean(const string& meanFile, const string& meanValues);
  void wrapInputLayer(vector<vector<cv::Mat>>* inputImages);
  void preProcess(const vector<cv::Mat>& images,
                  vector<vector<cv::Mat>>* inputImages);

  private:
  cnrtModel_t model;
  cv::Size inputGeometry;
  int batchSize;
  int numberChannels;
  cv::Mat meanValue;
  void** inputCpuPtrS;
  void** outputCpuPtrS;
  void** inputMluPtrS;
  void** outputMluPtrS;
  void** param;
  vector<float*> yolov3_input_cpu;
  vector<float*> yolov3_output_cpu;
  cnrtDataDescArray_t inputDescS, outputDescS;
  cnrtStream_t stream;
  int inputNum, outputNum;
  cnrtFunction_t function;

  vector<int> inputShape;
  vector<vector<int>> outputShape;
  int inputCount;
  int outputCount;
  float mlu_time;
  vector<int> outputCounts;
  vector<vector<vector<float>>> tensors;
};

Detector::Detector(const string& modelFile, const string& meanFile,
                   const string& meanValues) {
  // offline model
  // 1. init runtime_lib and device
  unsigned devNum;
  cnrtGetDeviceCount(&devNum);
  if (FLAGS_mludevice >= 0) {
    CHECK_NE(devNum, 0) << "No device found";
    CHECK_LT(FLAGS_mludevice, devNum) << "valid device count: " << devNum;
  } else {
    LOG(FATAL) << "Invalid device number";
  }
  cnrtDev_t dev;
  cnrtGetDeviceHandle(&dev, FLAGS_mludevice);
  cnrtSetCurrentDevice(dev);
  // 2. load model and get function
  std::cout << "load file: " << modelFile << std::endl;
  cnrtLoadModel(&model, modelFile.c_str());
  string name = "subnet0";
  cnrtCreateFunction(&function);
  cnrtExtractFunction(&function, model, name.c_str());
  // initializa function memory
  cnrtInitFuncParam_t initFuncParam;
  bool muta = false;
  int data_parallel = 1;//1
  unsigned int affinity = 0x01;
  initFuncParam.muta = &muta;
  initFuncParam.affinity = &affinity;
  initFuncParam.data_parallelism = &data_parallel;
  initFuncParam.end = CNRT_PARAM_END;
  cnrtInitFunctionMemory_V2(function, &initFuncParam);
  // 3. get function's I/O DataDesc
  cnrtGetInputDataDesc(&inputDescS, &inputNum, function);
  cnrtGetOutputDataDesc(&outputDescS, &outputNum, function);
#if !defined(CROSS_COMPILE) && !defined(CROSS_COMPILE_ARM64)
  uint64_t stack_size;
  cnrtQueryModelStackSize(model, &stack_size);
  unsigned int current_device_size;
  cnrtGetStackMem(&current_device_size);
  if (stack_size > current_device_size) {
    cnrtSetStackMem(stack_size + 50);
  }
#endif  // CROSS_COMPILE && CROSS_COMPILE_ARM64
  // 4. allocate I/O data space on CPU memory and prepare Input data
  inputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
  outputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));

  /* input shape : 1, 3, 416, 416 */
  // inputNum batch
//std::cout<<"inputNum "<<inputNum<<std::endl;
//exit(0);  
for (int i = 0; i < inputNum; i++) {
    unsigned int inNum, inChannel, inHeight, inWidth;
    cnrtDataDesc_t inputDesc = inputDescS[i];
    cnrtGetHostDataCount(inputDesc, &inputCount);
    inputCpuPtrS[i] =
        reinterpret_cast<void*>(malloc(sizeof(float) * inputCount));
    cnrtSetHostDataLayout(inputDesc, CNRT_FLOAT32, CNRT_NCHW);
    cnrtGetDataShape(inputDesc, &inNum, &inChannel, &inHeight, &inWidth);
    float* inputData = reinterpret_cast<float*>(inputCpuPtrS[i]);
    yolov3_input_cpu.push_back(inputData);
    batchSize = inNum;
    numberChannels = inChannel;
    inputGeometry = cv::Size(inWidth, inHeight);
    inputShape.push_back(inNum);
    inputShape.push_back(inChannel);
    inputShape.push_back(inHeight);
    inputShape.push_back(inWidth);
    LOG(INFO) << "shape " << inNum;
    LOG(INFO) << "shape " << inChannel;
    LOG(INFO) << "shape " << inHeight;
    LOG(INFO) << "shape " << inWidth;
  }

  /* output shape1 : 1, 255, 13, 13  */
  /* output shape2 : 1, 255, 26, 26  */
  /* output shape3 : 1, 255, 52, 52  */
  for (int i = 0; i < outputNum; i++) {
    vector<int> shape;
    unsigned int outNum, outChannel, outHeight, outWidth;
    cnrtDataDesc_t outputDesc = outputDescS[i];
    cnrtSetHostDataLayout(outputDesc, CNRT_FLOAT32, CNRT_NCHW);
    cnrtGetHostDataCount(outputDesc, &outputCount);
    cnrtGetDataShape(outputDesc, &outNum, &outChannel, &outHeight, &outWidth);
    outputCpuPtrS[i] =
        reinterpret_cast<void*>(malloc(sizeof(float) * outputCount));
    float* outputData = reinterpret_cast<float*>(outputCpuPtrS[i]);
    yolov3_output_cpu.push_back(outputData);
    shape.push_back(outNum);
    shape.push_back(outChannel);
    shape.push_back(outHeight);
    shape.push_back(outWidth);
    outputShape.push_back(shape);
    outputCounts.push_back(outputCount);
    LOG(INFO) << "output shape " << outNum;
    LOG(INFO) << "output shape " << outChannel;
    LOG(INFO) << "output shape " << outHeight;
    LOG(INFO) << "output shape " << outWidth;
  }

  // 5. allocate I/O data space on MLU memory and copy Input data
  cnrtMallocBatchByDescArray(&inputMluPtrS, inputDescS, inputNum, 1);
  cnrtMallocBatchByDescArray(&outputMluPtrS, outputDescS, outputNum, 1);
  cnrtCreateStream(&stream);
  setMean(meanFile, meanValues);
}

Detector::~Detector() {
  if (inputCpuPtrS != NULL) {
    for (int i = 0; i < inputNum; i++) {
      if (inputCpuPtrS[i] != NULL) free(inputCpuPtrS[i]);
    }
  }
  if (outputCpuPtrS != NULL) {
    for (int i = 0; i < outputNum; i++) {
      if (outputCpuPtrS[i] != NULL) free(outputCpuPtrS[i]);
    }
    free(outputCpuPtrS);
  }
  cnrtDestroyStream(stream);
  cnrtDestroyFunction(function);
  // unload model
  cnrtUnloadModel(model);
}

// obtain 1x255x13x13 blob data
// return 155 * 169
vector<vector<float>> get_blob_data(
    const vector<int>& yolov3_shape, const float* result_buffer,
    int batch) {
  // batch 64 mini_batch=8
  int batchs = yolov3_shape[0] / batch;
  int channels = yolov3_shape[1];
  int width = yolov3_shape[2];
  int height = yolov3_shape[3];
  vector<vector<float>> output_data(channels, vector<float>(width * height));
  for (int n = 0; n < batchs; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          output_data[c][h * width + w] =
              result_buffer[c * height * width + h * width + w];
        }
      }
    }
  }
  return output_data;
}

// transpose two vector  255*169->169*255
void transpose(const vector<vector<float>>& A, vector<vector<float>>* B) {
  if (A.size() == 0) {
    LOG(FATAL) << "input vector is equal to 0" << std::endl;
  }
  vector<vector<float>> C(A[0].size());
  for (int i = 0; i < C.size(); i++) C[i].resize(A.size());
  for (int i = 0; i < A[0].size(); i++) {
    for (int j = 0; j < A.size(); j++) {
      C[i][j] = A[j][i];
    }
  }
  *B = std::move(C);
}

// reshape two vector
void matrixReshape(vector<vector<float>> A, vector<vector<float>>* B,
                   int r, int c) {
  int m = A.size(), n = A[0].size();
  if (m * n != r * c) {
    *B = std::move(A);
  } else {
    vector<vector<float>> C(r);
    for (int x = 0; x < C.size(); x++) C[x].resize(c);
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
        int k = i * c + j;
        C[i][j] = static_cast<float>(A[k / n][k % n]);
      }
    }
    *B = std::move(C);
  }
}

// sigmoid two vector
void sigmoid(vector<vector<float>>* B, int col) {
  vector<vector<float>>::iterator it_begin = B->begin();
  for (; it_begin != B->end(); ++it_begin) {
    for (int i = col; i < col + 1; i++) {
      (*it_begin)[i] = 1 / (1 + std::exp(-(*it_begin)[i]));
    }
  }
}

void matrixAdd(vector<vector<float>>* a, const vector<vector<float>>& b,
               int numberOfRows, int selectColsLeft, int selectColsRight) {
  for (int i = 0; i < numberOfRows; i++) {
    for (int j = selectColsLeft; j < selectColsRight; j++) {
      (*a)[i][j] = (*a)[i][j] + b[i][j];
    }
  }
}

// sigmoid two vector
void sigmoid(vector<vector<float>>* B, int colsLeft, int colsRight) {
  vector<vector<float>>::iterator it_begin = B->begin();
  for (; it_begin != B->end(); ++it_begin) {
    for (int i = colsLeft; i < colsRight; i++) {
      (*it_begin)[i] = 1 / (1 + std::exp(-(*it_begin)[i]));
    }
  }
}

void matrixMulti(vector<vector<float>>* a, const vector<vector<float>>& b,
                 int numberOfRows, int selectColsLeft, int selectColsRight) {
  for (int i = 0; i < numberOfRows; i++)
    for (int j = selectColsLeft; j < selectColsRight; j++)
      (*a)[i][j] = std::exp((*a)[i][j]) * b[i][j - selectColsLeft];
}

void matrixMulti(vector<vector<float>>* a, int b, int numberOfRows,
                 int selectColsLeft, int selectColsRight) {
  for (int i = 0; i < numberOfRows; i++)
    for (int j = selectColsLeft; j < selectColsRight; j++)
      (*a)[i][j] = (*a)[i][j] * b;
}

void transform_tensor(vector<vector<float>> tensor_output,
                      vector<vector<float>>* tensor_data,
                      int num_classes,
                      vector<vector<float>> x_y_offset,
                      vector<vector<float>> anchor_tensor) {
  int input_dim = FLAGS_netinputdim;
  // 下采样步长   gride_size->stride 13->32 26->16 52->8
  int stride = input_dim / std::sqrt(tensor_output[0].size());  // 32
  //std::cout<<"stride:"<<stride<<std::endl;
  int gride_size = input_dim / stride;  // 13
  int bbox_attrs = 5 + num_classes;  // len(xywhc)+6
  int anchor_num = FLAGS_anchornum;

  vector<vector<float>> tensor_trans;
  transpose(tensor_output, &tensor_trans);  // 255*169( (80+5) *3  13*13)->169*255

  matrixReshape(tensor_trans, tensor_data, gride_size * gride_size * anchor_num,
                bbox_attrs);  // 169*255->507*85
// int i = col; i < col + 1; i++
  sigmoid(tensor_data, 0);//x
  sigmoid(tensor_data, 1);//y

  sigmoid(tensor_data, 4);//conf

//exp(px) + x0 
  matrixAdd(tensor_data, x_y_offset, tensor_data->size(), 0, 2);

  // exp(pw)*aw 得到特征图坐标
  matrixMulti(tensor_data, anchor_tensor, tensor_data->size(), 2, 4);

  sigmoid(tensor_data, 5, bbox_attrs);
  // 得到netinput输入坐标
  matrixMulti(tensor_data, stride, tensor_data->size(), 0, 4);
}

void concatenate(vector<vector<float>>* all_boxes,
                 const vector<vector<float>>& boxes_13,
                 const vector<vector<float>>& boxes_26,
                 const vector<vector<float>>& boxes_52) {
  vector<vector<float>> temp(
      (boxes_13.size() + boxes_26.size() + boxes_52.size()),
      vector<float>(boxes_13[0].size(), 0));
  for (int j = 0; j < temp.size(); j++) {
    for (int k = 0; k < temp[0].size(); k++) {
      if (j < boxes_13.size()) {
        temp[j][k] = boxes_13[j][k];
      } else {
        if (j >= boxes_13.size() && j < (boxes_13.size() + boxes_26.size())) {
          temp[j][k] = boxes_26[j - boxes_13.size()][k];
        } else {
          temp[j][k] = boxes_52[j - (boxes_13.size() + boxes_26.size())][k];
        }
      }
    }
  }
  *all_boxes = std::move(temp);
}

void fill_zeros(vector<vector<float>>* all_boxes, int cols, float confidence) {
  for (int i = 0; i < all_boxes->size(); i++) {
    if ((*all_boxes)[i][cols] > confidence)
    { 
       (*all_boxes)[i][cols] = 1;
           continue;
    }
    else
      (*all_boxes)[i][cols] = 0;
  }
}

vector<vector<float>> filter_boxes(vector<vector<float>>* all_boxes,
                                   vector<float>* max_class_score,
                                   vector<float>* max_class_idx) {
  vector<vector<float>> temp(all_boxes->size(), vector<float>(5 + 2, 0));
  for (int i = 0; i < all_boxes->size(); i++) {
    for (int j = 0; j < 7; j++) {
      if (j < 5)
        temp[i][j] = (*all_boxes)[i][j];
      else if (j == 5)
        temp[i][j] = (*max_class_score)[i];
      else if (j == 6)
        temp[i][j] = (*max_class_idx)[i];
      else
        LOG(FATAL) << " filter_boxes index error ";
    }
  }
  vector<vector<float>> vec;
  for (int m = 0; m < temp.size(); m++) {
    if (temp[m][4] == 0)
      continue;
    else
      vec.push_back(temp[m]);
  }
  return vec;
}

void unique_vector(vector<vector<float>>* input_vector,
                   vector<float>* output_vector) {
  for (int i = 0; i < input_vector->size(); i++) {
    (*output_vector).push_back((*input_vector)[i][6]);
  }
  sort((*output_vector).begin(), (*output_vector).end());
  auto new_end = unique((*output_vector).begin(), (*output_vector).end());
  (*output_vector).erase(new_end, (*output_vector).end());
}

float findMax(vector<float> vec) {
  float max = -999;
  for (auto v : vec) {
    if (max < v) max = v;
  }
  return max;
}

int getPositionOfMax(vector<float> vec, float max) {
  auto distance = find(vec.begin(), vec.end(), max);
  return distance - vec.begin();
}

void nms_by_classes(vector<vector<float>> sort_boxes, vector<float>* ious,
                    int start) {
  for (int i = start + 1; i < sort_boxes.size(); i++) {
    float first_x1 = sort_boxes[start][0];
    float first_y1 = sort_boxes[start][1];
    float first_x2 = sort_boxes[start][2];
    float first_y2 = sort_boxes[start][3];

    float next_x1 = sort_boxes[i][0];
    float next_y1 = sort_boxes[i][1];
    float next_x2 = sort_boxes[i][2];
    float next_y2 = sort_boxes[i][3];

    float inter_x1 = std::max(first_x1, next_x1);
    float inter_y1 = std::max(first_y1, next_y1);
    float inter_x2 = std::min(first_x2, next_x2);
    float inter_y2 = std::min(first_y2, next_y2);
    float inter_area, first_area, next_area, union_area, iou;
    if ((inter_x2 - inter_x1 + 1 > 0) && (inter_y2 - inter_y1 + 1 > 0))
      inter_area = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1);
    else
      inter_area = 0;
    first_area = (first_x2 - first_x1 + 1) * (first_y2 - first_y1 + 1);
    next_area = (next_x2 - next_x1 + 1) * (next_y2 - next_y1 + 1);
    union_area = first_area + next_area - inter_area;
    iou = inter_area / union_area;
    (*ious).push_back(iou);
  }
}

void nms(vector<vector<float>> boxes_cleaned,
        vector<vector<float>>* final_boxes, float nms_thresh){
  // 得到这张图片包含哪些类别
  vector<float> unique_classes;
  unique_vector(&boxes_cleaned, &unique_classes);
  vector<vector<float>> curr_classes;
  for (auto v : unique_classes) {
    // nms 对于每个类别中的每个box
    for (int m = 0; m < boxes_cleaned.size(); m++) {
      if (boxes_cleaned[m][6] == v)
        curr_classes.push_back(boxes_cleaned[m]);
      else
        continue;
    }
    vector<float> object_score;
    for (int n = 0; n < curr_classes.size(); n++) {
      object_score.push_back(curr_classes[n][4]);
    }
    vector<float> sort_score, sort_idx;
    for (int i = 0; i < object_score.size(); i++) {
      float maxNumber = findMax(object_score);
      sort_score.push_back(maxNumber);
      int maxIndex = getPositionOfMax(object_score, maxNumber);
      sort_idx.push_back(maxIndex);
      object_score[maxIndex] = -999;
    }
    vector<vector<float>> sort_boxes;
    for (int j = 0; j < sort_idx.size(); j++) {
      sort_boxes.push_back(curr_classes[sort_idx[j]]);
    }
    vector<float> ious;
    for (int k = 0; k < sort_boxes.size(); k++) {
      // 对于一个类别的某个高置信度的框计算他与每个框的iou 删除大于iou阈值的框
      ious.clear();
      nms_by_classes(sort_boxes, &ious, k);
      int dele_number = 0;

      for (int s = 0; s < ious.size(); s++) {
        if (ious[s] > nms_thresh) {
          sort_boxes.erase(sort_boxes.begin() + k - dele_number + 1 + s);
          dele_number = dele_number + 1;
        } else {
          continue;
        }
      }
    }
    for (int t = 0; t < sort_boxes.size(); t++) {
      (*final_boxes).push_back(sort_boxes[t]);
    }
    // 清楚内容
    curr_classes.clear();
    object_score.clear();
    sort_score.clear();
    sort_idx.clear();
    sort_boxes.clear();
  }
}

void get_detection(vector<vector<float>> all_boxes,
                   vector<vector<float>>* final_boxes, int num_classes,
                   float confidence, float nms_thresh) {
  // 对一张图的结果进行处理s
  // 过滤低置信度的目标 低于阈值设为0
  fill_zeros(&all_boxes, 4, confidence);

  vector<vector<float>> boxes_copy;
  boxes_copy = all_boxes;
  for (int i = 0; i < all_boxes.size(); i++) {
    // xywh -> x1y1x2y2
    all_boxes[i][0] = boxes_copy[i][0] - boxes_copy[i][2] / 2;
    all_boxes[i][1] = boxes_copy[i][1] - boxes_copy[i][3] / 2;
    all_boxes[i][2] = boxes_copy[i][0] + boxes_copy[i][2] / 2;
    all_boxes[i][3] = boxes_copy[i][1] + boxes_copy[i][3] / 2;
  }

  // 获得每个目标的最大置信度和对应类别
  vector<float> max_class_idx;
  vector<float> max_class_score;
  for (int j = 0; j < all_boxes.size(); j++) {
    vector<float>::iterator biggest =
        std::max_element(std::begin(all_boxes[j]) + 5, std::end(all_boxes[j]));
    max_class_score.push_back(*biggest);
    max_class_idx.push_back(
        std::distance(std::begin(all_boxes[j]) + 5, biggest));
  }

  vector<vector<float>> boxes_cleaned;
  // all_box -> xyxy  conf cls_score cls 过滤 0 conf
  boxes_cleaned = filter_boxes(&all_boxes, &max_class_score, &max_class_idx);
  nms(boxes_cleaned, final_boxes, nms_thresh);
}



// core
vector<vector<vector<float>>> Detector::detect(
    const vector<cv::Mat>& images,
    const vector<vector<vector<float>>>& x_y_offsets,
    const vector<vector<vector<float>>>& anchor_tensors) {
  vector<vector<cv::Mat>> inputImages;
  // 预填充空�
  wrapInputLayer(&inputImages);
  // 预处理 图像大小是512 512
  // 单通道uint16 -> uint8 自适应均衡
  preProcess(images, &inputImages);
  cnrtEvent_t eventStart, eventEnd;
  cnrtCreateEvent(&eventStart);
  cnrtCreateEvent(&eventEnd);
  float eventTimeUse;
  cnrtMemcpyBatchByDescArray(inputMluPtrS, inputCpuPtrS, inputDescS, inputNum,
                             1, CNRT_MEM_TRANS_DIR_HOST2DEV);
  param =reinterpret_cast<void**>(malloc(sizeof(void*) * (inputNum + outputNum)));
  for (int i = 0; i < inputNum; i++) param[i] = inputMluPtrS[i];
  for (int i = 0; i < outputNum; i++) param[i + inputNum] = outputMluPtrS[i];
  cnrtInvokeFuncParam_t invokeFuncParam;
  cnrtDim3_t dim = {1, 1, 1};
  unsigned int affinity = 0x01;
  int dp = 1;

  invokeFuncParam.data_parallelism = &dp;
  invokeFuncParam.affinity = &affinity;
  invokeFuncParam.end = CNRT_PARAM_END;
  cnrtFunctionType_t funcType = (cnrtFunctionType_t)0;

cnrtPlaceEvent(eventStart, stream);

  CNRT_CHECK(cnrtInvokeFunction(function, dim, param, funcType, stream,
                                &invokeFuncParam));
  cnrtPlaceEvent(eventEnd, stream);

  if (cnrtSyncStream(stream) == CNRT_RET_SUCCESS) {
    cnrtEventElapsedTime(eventStart, eventEnd, &eventTimeUse);
    std::cout << " execution time: " << eventTimeUse << std::endl;
    mlu_time += eventTimeUse;
  } else {
    std::cout << " SyncStream error " << std::endl;
  }

  cnrtMemcpyBatchByDescArray(outputCpuPtrS, outputMluPtrS, outputDescS,
                             outputNum, 1, CNRT_MEM_TRANS_DIR_DEV2HOST);
 
 
 /* copy the output layer to a vector*/
  /* 255 * 169  13 13* 255=(80+5)*3/ 
  /* 255 * 676  26 26*/
  /* 255 * 2704 54 54*/
  // batch n 7
  vector<vector<vector<float>>> final_boxes;

  // layer1 [im1 im2 imn]
  // layer2 [im1 im2 imn]
  // layer3 [im1 im2 imn]
  // batch
  for (int m = 0; m < inputShape[0]; m++) {
    
    // tensors.clear();
    vector<vector<float>> tensor;
    vector<vector<float>> transform_boxes;

    vector<vector<float>> all_boxes;
    // 2(tiny) pr 3(v3)layers 确定是不是从小特征图到大特征图
    for (int i = 0; i < outputNum; i++) {
      // 每一层每张图结果长度outputCounts/batch
      int singleCount = outputCounts[i] / inputShape[0];
      float* outputData = reinterpret_cast<float*>(outputCpuPtrS[i]);

      // const vector<int>& yolov3_shape, const float* result_buffer, int batch
      tensor = get_blob_data(outputShape[i], outputData + m * singleCount, inputShape[0]);

      transform_boxes.clear();
      // 13 26 52 -> 169 676 2704
      transform_tensor(tensor, &transform_boxes, FLAGS_classnum, x_y_offsets[i],
                       anchor_tensors[i]);
      std::cout<<" layer:"<<i<<" size:"<<transform_boxes.size()<<std::endl;
      all_boxes.insert(all_boxes.end(), transform_boxes.begin(), transform_boxes.end());
      // tensors.push_back(tensor);// tensors包含一张图片的三层结果
    }

    vector<vector<float>> tmp_boxes;
    // 10647*85
    // concatenate(&all_boxes, three_boxes[0], three_boxes[1], three_boxes[2]);

    // 过滤以及 nms
    get_detection(all_boxes, &tmp_boxes, FLAGS_classnum, FLAGS_confidence, FLAGS_nmsthresh);
    final_boxes.push_back(tmp_boxes);


    // modify
    //   fill_zeros(&all_boxes, 4, FLAGS_confidence);

    // vector<vector<float>> boxes_copy;
    // boxes_copy = all_boxes;
    // for (int i = 0; i < all_boxes.size(); i++) {
    //   // 左上角和右下角点
    //   all_boxes[i][0] = boxes_copy[i][0] - boxes_copy[i][2] / 2;
    //   all_boxes[i][1] = boxes_copy[i][1] - boxes_copy[i][3] / 2;
    //   all_boxes[i][2] = boxes_copy[i][0] + boxes_copy[i][2] / 2;
    //   all_boxes[i][3] = boxes_copy[i][1] + boxes_copy[i][3] / 2;
    // }

    // final_boxes.push_back(boxes_copy);

  }//for sbatch





// ori
  // /* copy the output layer to a vector*/
  // /* 255 * 169  13 13* 255=(80+5)*3/ 
  // /* 255 * 676  26 26*/
  // /* 255 * 2704 54 54*/
  // // batch n 7
  // vector<vector<vector<float>>> final_boxes;

  // layer1 [im1 im2 imn]
  // layer2 [im1 im2 imn]
  // layer3 [im1 im2 imn]
  // batch
  // for (int m = 0; m < inputShape[0]; m++) {
  //   tensors.clear();
  //   vector<vector<float>> tensor;
  //   // 3layers
  //   for (int i = 0; i < outputNum; i++) {
  //     // outputCounts/batch
  //     int singleCount = outputCounts[i] / inputShape[0];
  //     float* outputData = reinterpret_cast<float*>(outputCpuPtrS[i]);

  //     // const vector<int>& yolov3_shape, const float* result_buffer, int batch
  //     tensor = get_blob_data(outputShape[i], outputData + m * singleCount, inputShape[0]);
      
  //     tensors.push_back(tensor);
  //   }

  //   // tensors包含一张图片的三层结果

  //   vector<vector<vector<float>>> three_boxes;
  //   three_boxes.resize(3);
  //   for (int i = 0; i < 3; i++) {
  //     transform_tensor(tensors[i], &three_boxes[i], FLAGS_classnum, x_y_offsets[i],
  //                      anchor_tensors[i]);
  //   }
  //   // ori
  //   vector<vector<float>> all_boxes, tmp_boxes;
  //   // 10647*85
  //   concatenate(&all_boxes, three_boxes[0], three_boxes[1], three_boxes[2]);
  //   // 过滤以及 nms
  //   get_detection(all_boxes, &tmp_boxes, FLAGS_classnum, FLAGS_confidence, FLAGS_nmsthresh);
  //   final_boxes.push_back(tmp_boxes);


  //   // modify
  //   //   fill_zeros(&all_boxes, 4, FLAGS_confidence);

  //   // vector<vector<float>> boxes_copy;
  //   // boxes_copy = all_boxes;
  //   // for (int i = 0; i < all_boxes.size(); i++) {
  //   //   // 左上角和右下角点
  //   //   all_boxes[i][0] = boxes_copy[i][0] - boxes_copy[i][2] / 2;
  //   //   all_boxes[i][1] = boxes_copy[i][1] - boxes_copy[i][3] / 2;
  //   //   all_boxes[i][2] = boxes_copy[i][0] + boxes_copy[i][2] / 2;
  //   //   all_boxes[i][3] = boxes_copy[i][1] + boxes_copy[i][3] / 2;
  //   // }

  //   // final_boxes.push_back(boxes_copy);

  // }//for sbatch
  return final_boxes;
}

/* Load the mean file in binaryproto forcv::Mat. */
void Detector::setMean(const string& meanFile, const string& meanValues) {
  cv::Scalar channelMean;
  if (!meanValues.empty()) {
    if (!meanFile.empty()) {
      std::cout << "Cannot specify mean file";
      std::cout << " and mean value at the same time; " << std::endl;
      std::cout << "Mean value will be specified " << std::endl;
    }
    stringstream ss(meanValues);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == numberChannels)
        << "Specify either 1 mean_value or as many as channels: "
        << numberChannels;
    vector<cv::Mat> channels;
    for (int i = 0; i < numberChannels; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(inputGeometry.height, inputGeometry.width, CV_32FC1,
                      cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, meanValue);
  } else {
    LOG(INFO) << "Cannot support mean file";
  }
}

void Detector::wrapInputLayer(vector<vector<cv::Mat>>* inputImages) {
  int width = inputGeometry.width;
  int height = inputGeometry.height;
  float* inputData = reinterpret_cast<float*>(inputCpuPtrS[0]);
  for (int i = 0; i < batchSize; ++i) {
    (*inputImages).push_back(vector<cv::Mat>());
    for (int j = 0; j < numberChannels; ++j) {
      cv::Mat channel(height, width, CV_32FC1, inputData);
      (*inputImages)[i].push_back(channel);
      inputData += width * height;
    }
  }
}

void Detector::preProcess(const vector<cv::Mat>& images,
                          vector<vector<cv::Mat>>* inputImages) {
  CHECK(images.size() == inputImages->size())
      << "Size of imgs and input_imgs doesn't cv::Match";
  for (int i = 0; i < images.size(); ++i) {
    cv::Mat sample;
    int num_channels_ = inputShape[1];
    cv::Size input_geometry;
    input_geometry = cv::Size(inputShape[2], inputShape[3]);  // 416*416
    if (images[i].channels() == 3 && num_channels_ == 1)
      cv::cvtColor(images[i], sample, cv::COLOR_BGR2GRAY);
    else if (images[i].channels() == 4 && num_channels_ == 1)
      cv::cvtColor(images[i], sample, cv::COLOR_BGRA2GRAY);
    else if (images[i].channels() == 4 && num_channels_ == 3)
      cv::cvtColor(images[i], sample, cv::COLOR_BGRA2BGR);
    else if (images[i].channels() == 1 && num_channels_ == 3)
      cv::cvtColor(images[i], sample, cv::COLOR_GRAY2BGR);
    else
      sample = images[i];

    // 2.resize the image evelop
    cv::Mat sample_temp;
    int input_dim = inputShape[2];
    cv::Mat sample_resized(input_dim, input_dim, CV_8UC3,
                           cv::Scalar(128, 128, 128));
    if (sample.size() != input_geometry) {
      // resize
      float img_w = sample.cols;
      float img_h = sample.rows;
      int new_w = static_cast<int>(
          img_w * std::min(static_cast<float>(input_dim) / img_w,
                           static_cast<float>(input_dim) / img_h));
      int new_h = static_cast<int>(
          img_h * std::min(static_cast<float>(input_dim) / img_w,
                           static_cast<float>(input_dim) / img_h));
      cv::resize(sample, sample_temp, cv::Size(new_w, new_h), CV_INTER_CUBIC);
      sample_temp.copyTo(sample_resized(
          cv::Range((static_cast<float>(input_dim) - new_h) / 2,
                    (static_cast<float>(input_dim) - new_h) / 2 + new_h),
          cv::Range((static_cast<float>(input_dim) - new_w) / 2,
                    (static_cast<float>(input_dim) - new_w) / 2 + new_w)));
    } else {
      sample_resized = sample;
    }

    // 3.BGR->RGB
    cv::Mat sample_rgb;
    //cv::cvtColor(sample_resized, sample_rgb, cv::COLOR_BGR2RGB);
    // 4.convert to float
    cv::Mat sample_float;
    if (num_channels_ == 3)
      // 1/255.0
      //std::cout<<"1/255."<<std::endl;
      //sample_rgb.convertTo(sample_float, CV_32FC3, 1/255.);
sample_resized.convertTo(sample_float, CV_32FC3, 1/255.);
   else
      sample_rgb.convertTo(sample_float, CV_32FC1, 1./255.);

    cv::Mat sampleNormalized;
    if (FLAGS_fix8 || (FLAGS_meanvalue.empty() && FLAGS_meanfile.empty()))
      sampleNormalized = sample_float;
    else
      cv::subtract(sample_float, meanValue, sampleNormalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sampleNormalized, (*inputImages)[i]);
  }
}



//ori
// void Detector::readImages(queue<string>* imagesQueue, int inputNumber,
//                           vector<cv::Mat>* images, vector<string>* imageNames) {
//   int leftNumber = imagesQueue->size();
//   string file = imagesQueue->front();
//   for (int i = 0; i < inputNumber; i++) {
//     if (i < leftNumber) {
//       file = imagesQueue->front();
//       imageNames->push_back(file);
//       imagesQueue->pop();
//       if (file.find(" ") != string::npos) file = file.substr(0, file.find(" "));
//       cv::Mat image = cv::imread(file, -1);
//       images->push_back(image);
//     } else {
//       cv::Mat image = cv::imread(file, -1);
//       images->push_back(image);
//       imageNames->push_back("null");
//     }
//   }
// }

// void readtxt(vector<vector<vector<float>>>* x_y_offsets,
//              vector<vector<vector<float>>>* anchor_tensors,
//              vector<int> output_stride, vector<vector<float>> anchor_strs) {
// //  vector<int> size_row = {507, 2028, 8112};
//   // x_y_offsets.push_back(vector<vector<float>>(507, vector<float>(2)));
//   // x_y_offsets.push_back(vector<vector<float>>(2028, vector<float>(2)));
//   // x_y_offsets.push_back(vector<vector<float>>(8112, vector<float>(2)));
//   // anchor_tensors.push_back(vector<vector<float>>(507, vector<float>(2)));
//   // anchor_tensors.push_back(vector<vector<float>>(2028, vector<float>(2)));
//   // anchor_tensors.push_back(vector<vector<float>>(8112, vector<float>(2)));
  
//   // ori
//   int size = x_y_offsets->size();
//   for (int i = 0; i < size; i++) {
//     std::ifstream infile_1;
//     string filename_1 = FLAGS_bboxanchordir + "/x_y_offset_"
//         + anchor_strs[i] + ".txt";
//     infile_1.open(filename_1);
//     for (int m = 0; m < size_rows[i]; m++) {
//       for (int n = 0; n < 2; n++) {
//         infile_1 >> (*x_y_offsets)[i][m][n];
//       }
//     }
//     infile_1.close();
//     std::ifstream infile_2;
//     string filename_2 = FLAGS_bboxanchordir +  "/anchors_tensor_"
//         + anchor_strs[i] + ".txt";
//     infile_2.open(filename_2);
//     for (int m = 0; m < size_rows[i]; m++) {
//       for (int n = 0; n < 2; n++) {
//         infile_2 >> (*anchor_tensors)[i][m][n];
//       }
//     }
//   }
// }

void readtxt(vector<vector<vector<float>>>* x_y_offsets,
             vector<vector<vector<float>>>* anchor_tensors,
             vector<int> output_stride, vector<vector<float>> anchors) {

  
  int layers = x_y_offsets->size();
for (int l = 0; l < layers; l++) {
    for (int row=0;row<output_stride[l];++row)
    {
      for (int col=0;col<output_stride[l];++col)
      {
          for (int a=0;a<FLAGS_anchornum;++a)
          {
            // int layer_num = output_stride[l]*output_stride[l];
            // 3anchor
            (*anchor_tensors)[l][FLAGS_anchornum*(row*output_stride[l] + col)+a][0] = anchors[FLAGS_anchornum*l+a][0]*1.0/FLAGS_netinputdim*output_stride[l];
            (*anchor_tensors)[l][FLAGS_anchornum*(row*output_stride[l] + col)+a][1] = anchors[FLAGS_anchornum*l+a][1]*1.0/FLAGS_netinputdim*output_stride[l];
            (*x_y_offsets)[l][FLAGS_anchornum*(row*output_stride[l] + col) + a][0] = col;
            (*x_y_offsets)[l][FLAGS_anchornum*(row*output_stride[l] + col) + a][1] = row;
          }//for anchor(default 3)
      }//for col
    }//for row
  }//for layer
}


// add
vector<string> split(const string &s, const string &seperator){
  vector<string> result;
  typedef string::size_type string_size;
  string_size i = 0;
  
  while(i != s.size()){
    //找到字符串中首个不等于分隔符的字母；
    int flag = 0;
    while(i != s.size() && flag == 0){
      flag = 1;
      for(string_size x = 0; x < seperator.size(); ++x)
        if(s[i] == seperator[x]){
          ++i;
          flag = 0;
        break;
      }
    }
    
    //找到又一个分隔符，将两个分隔符之间的字符串取出；
    flag = 0;
    string_size j = i;
    while(j != s.size() && flag == 0){
     for(string_size x = 0; x < seperator.size(); ++x)
        if(s[j] == seperator[x]){
        flag = 1;
        break;
        }
      if(flag == 0) 
      ++j;
    }
    if(i != j){
      result.push_back(s.substr(i, j-i));
      i = j;
    }
  }
  return result;
}
//add
void fileNameFromPath(const string full_path, string & dir_name, string & file_name){
    //1.获取不带路径的文件名
	string::size_type iPos = full_path.find_last_of('/') + 1;
	string filename = full_path.substr(iPos, full_path.length() - iPos);
	// // std::cout << filename << std::endl;
 
	// //2.获取不带后缀的文件名
	string name = filename.substr(0, filename.rfind("."));
	// std::cout << name << std::endl;

    // ../main_dir/pic_all_n/pic_cut/*.tif
	vector<string> sliceInfoV = split(full_path, "/");
    string pic_all_n = sliceInfoV[sliceInfoV.size() - 3];
    // std::cout<<pic_all_n<<" "<<name<<std::endl;
    dir_name = pic_all_n;
    file_name = name;
}

void sliceInfo_from_FullPath(const string full_path, string* srcName, int* x0, int* y0, int*sliceim_w, int*sliceim_h, int*bigim_w=NULL, int*bigim_h=NULL){
    
    //1.获取不带路径的文件名
    string::size_type startPos, endPos;
    startPos = full_path.rfind('/') + 1;
    // %s|%s_%s_%s_%s_%s_%s.txt
    string sliceName = full_path.substr(startPos);
    
    //2解析文件名
    endPos = full_path.find('|');
    // std::cout<<"slice_name"<<sliceName<<" "<<startPos<<" "<<endPos<<std::endl;
    
    *srcName = full_path.substr(startPos, endPos - startPos);
    string sliceInfo = full_path.substr(endPos + 1);

    vector<string> sliceInfoV = split(sliceInfo, "_");
    
    // std::cout<<sliceInfoV[0]<<" "<<sliceInfoV[1]<<std::endl;
    // %s_%s|  %d_%d_%d_%d_%d_%d.txt", outdir, dir_name, file_name,   y0,x0,FLAGS_slice_size, FLAGS_slice_size, img_w, img_h
    stringstream ss;
    ss<<sliceInfoV[0];
    ss>>*y0;
    ss.clear();
    ss<<sliceInfoV[1];
    ss>>*x0;
    
    ss.clear();
    ss<<sliceInfoV[2];
    ss>>*sliceim_h;
    ss.clear();
    ss<<sliceInfoV[3];
    ss>>*sliceim_w;

    ss.clear();
    ss<<sliceInfoV[4];
    ss>>*bigim_w;
    ss.clear();
    ss<<sliceInfoV[5];
    ss>>*bigim_h;
}


static void get_batch_global_coord(
    const vector<cv::Mat>& images,
    const vector<vector<vector<float>>> detections,
    vector<vector<float>>& batch_results,
    const vector<string>& labelToDisplayName,
    const vector<string>& imageNames,
    int input_dim,
    bool visuallize=false) {
  // Retrieve detections.
  const int imageNumber = imageNames.size();

  for (int i = 0; i < imageNumber; ++i) {
    if (imageNames[i] == "null") continue;
    // '''xywh conf cls_score cls'''
    vector<vector<float>> result = detections[i];
    std::string name = imageNames[i];
    // '''要改成根据裁切图像名得到原图文件名 然后进行保存 最好进行nms'''
    // 得到输入图片文件名
    // 反向查找
    // int positionMap = imageNames[i].rfind("/");
    // if (positionMap > 0 && positionMap < imageNames[i].size()) {
    //   name = name.substr(positionMap + 1);
    // }
    // positionMap = name.rfind(".");
    // if (positionMap > 0 && positionMap < name.size()) {
    //   name = name.substr(0, positionMap);
    // }
    // name 不包括后缀裁切后的图像名称

    // 找到原图名称和裁切信息 得到全局坐标
    // srcName %s|%s_%s_%s_%s_%s_%s.%s
    std::string srcName;
    int x0,y0,sliceim_w,sliceim_h, bigim_w,bigim_h;
    // imageNames "%s/%s_%s|%d_%d_%d_%d_%d_%d.png", FLAGS_outputdir, dir_name, file_name,y0,x0,FLAGS_slice_size, FLAGS_slice_size, img_w, img_h
    sliceInfo_from_FullPath(name, &srcName, &x0, &y0, &sliceim_w, &sliceim_h, &bigim_w, &bigim_h);


    float scaling_factors = std::min(
        static_cast<float>(input_dim) / static_cast<float>(sliceim_w),
        static_cast<float>(input_dim) / static_cast<float>(sliceim_h));

    for (int j = 0; j < result.size(); j++) {
      // ''' 得到网络处理尺寸的真实坐标'''
      result[j][0] =
          result[j][0] -
          static_cast<float>(input_dim - scaling_factors * sliceim_w) /2.0;
      result[j][2] =
          result[j][2] -
          static_cast<float>(input_dim - scaling_factors * sliceim_w) /2.0;
      result[j][1] =
          result[j][1] -
          static_cast<float>(input_dim - scaling_factors * sliceim_h) /2.0;
      result[j][3] =
          result[j][3] -
          static_cast<float>(input_dim - scaling_factors * sliceim_h) /2.0;
      // '''缩放到原图坐标'''
      // for (int k = 0; k < 4; k++) {
      //   result[j][k] = result[j][k] / scaling_factors;
      // }
        // result[j][0] = result[j][0] / scaling_factors + x0;
        // result[j][1] = result[j][1] / scaling_factors + y0;
        // result[j][2] = result[j][2] / scaling_factors + x0;
        // result[j][3] = result[j][3] / scaling_factors + y0;
        
        result[j][0] = result[j][0] / scaling_factors;
        result[j][1] = result[j][1] / scaling_factors;
        result[j][2] = result[j][2] / scaling_factors;
        result[j][3] = result[j][3] / scaling_factors;

    }

    // // '''检查边界'''
    // for (int j = 0; j < result.size(); j++) {
    //   // bounding boxes bundary check
    //   result[j][0] = result[j][0] < 0 ? 0 : result[j][0];
    //   result[j][2] = result[j][2] < 0 ? 0 : result[j][2];
    //   result[j][1] = result[j][1] < 0 ? 0 : result[j][1];
    //   result[j][3] = result[j][3] < 0 ? 0 : result[j][3];

    //   result[j][0] =
    //       result[j][0] > bigim_w ? bigim_w : result[j][0];
    //   result[j][2] =
    //       result[j][2] > bigim_w ? bigim_w : result[j][2];
    //   result[j][1] =
    //       result[j][1] > bigim_h ? bigim_h : result[j][1];
    //   result[j][3] =
    //       result[j][3] > bigim_h ? bigim_h : result[j][3];
    // }

    // batch_results.insert(batch_results.end(), result.begin(), result.end());

    // '''绘制结果'''
    if (visuallize==true)
    {
      // dir_name_fiename
      name = FLAGS_outputdir + "/" + srcName + "_temp.txt";
std::cout<<name<<std::endl;    
//   '''add模式'''
      std::ofstream fileMap(name);

      cv::Mat image = images[i];
      for (int j = 0; j < result.size(); j++) {
cv::Point p1(static_cast<int>(result[j][0]),
                    static_cast<int>(result[j][1]));
        cv::Point p2(static_cast<int>(result[j][2]),
                    static_cast<int>(result[j][3]));
        cv::rectangle(image, p1, p2, cv::Scalar(0, 0, 255), 1, 1, 0);
        
cv::Point p3(static_cast<int>(result[j][0]),
                    static_cast<int>(result[j][1]) - 20);
        cv::Point p4(static_cast<int>(result[j][0]) + 100,
                    static_cast<int>(result[j][1]));
        cv::rectangle(image, p3, p4, cv::Scalar(255, 0, 0), -1, 4);
stringstream ss;
        ss << round(result[j][4] * 1000) / 1000.0;
std::string str =
          labelToDisplayName[static_cast<int>(result[j][6])] + ":" + ss.str();
cv::Point p5(static_cast<int>(result[j][0]),
                    static_cast<int>(result[j][1]) - 1);
cv::putText(image, str, p5, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 215, 0), 1);
        // fileMap << labelToDisplayName[static_cast<int>(result[j][6])] << " "
        //         << ss.str() << " " << static_cast<float>(p1.x) / image.cols << " "
        //         << static_cast<float>(p1.y) / image.rows << " "
        //         << static_cast<float>(p2.x) / image.cols << " "
        //         << static_cast<float>(p2.y) / image.rows << " " << image.cols
        //         << " " << image.rows << std::endl;
        // fileMap << ss_cls.str() << " "<< ss_conf.str() << " " << static_cast<float>(p1.x) / image.cols << " "
        //         << static_cast<float>(p1.y) / image.rows << " "
        //         << static_cast<float>(p2.x) / image.cols << " "
        //         << static_cast<float>(p2.y) / image.rows << " " << image.cols
        //         << " " << image.rows << std::endl;
        stringstream ss_conf, ss_cls;
        // int str
        // ss_cls << result[j][6]
        // 
        ss_conf << round(result[j][4] * 1000) / 1000.0;
        ss_cls  << round(result[j][5] * 1000) / 1000.0;

        // x1 y1 x2 y2 conf cls_score cls_id
        fileMap << static_cast<int>(result[j][0])  << " "
                << static_cast<int>(result[j][1])  << " "
                << static_cast<int>(result[j][2])  << " "
                << static_cast<int>(result[j][3])  << " "
                << ss_conf.str()   << " "
                << ss_cls.str()  << " "
                << static_cast<int>(result[j][6])  << " "<< std::endl;
      }
      fileMap.close();
      stringstream ss;
      string outFile;
      string path = FLAGS_outputdir + "/";
      ss << path << srcName<<y0<<x0<<".png";
      ss >> outFile;
      cv::imwrite(outFile.c_str(), image);
      //exit(0)
      // cv::imwrite((FLAGS_outputdir + "/" + imageNames[i].c_str()), image);
    }



    // 转换到全局坐标 并检查范围
    for (int j = 0; j < result.size(); j++) {
      // bounding boxes bundary check
      result[j][0] = result[j][0]  + x0;
      result[j][1] = result[j][1]  + y0;
      result[j][2] = result[j][2]  + x0;
      result[j][3] = result[j][3]  + y0;
      
      result[j][0] = result[j][0] < 0 ? 0 : result[j][0];
      result[j][2] = result[j][2] < 0 ? 0 : result[j][2];
      result[j][1] = result[j][1] < 0 ? 0 : result[j][1];
      result[j][3] = result[j][3] < 0 ? 0 : result[j][3];

      result[j][0] =
          result[j][0] > bigim_w ? bigim_w : result[j][0];
      result[j][2] =
          result[j][2] > bigim_w ? bigim_w : result[j][2];
      result[j][1] =
          result[j][1] > bigim_h ? bigim_h : result[j][1];
      result[j][3] =
          result[j][3] > bigim_h ? bigim_h : result[j][3];
    }

   batch_results.insert(batch_results.end(), result.begin(), result.end());
    
  }//for batch
}


static void WriteVisualizeBBox_offline(
    const vector<cv::Mat>& images,
    const vector<vector<vector<float>>> detections,
    const vector<string>& labelToDisplayName,
    const vector<string>& imageNames,
    int input_dim) {
  // Retrieve detections.
  const int imageNumber = images.size();

  for (int i = 0; i < imageNumber; ++i) {
    if (imageNames[i] == "null") continue;
    vector<vector<float>> result = detections[i];
    cv::Mat image = images[i];
    std::string name = imageNames[i];
    
    // '''要改成根据裁切图像名得到原图文件名 然后进行保存 最好进行nms'''
    // 得到输入图片文件名
    // 反向查找
    int positionMap = imageNames[i].rfind("/");
    if (positionMap > 0 && positionMap < imageNames[i].size()) {
      name = name.substr(positionMap + 1);
    }
    positionMap = name.rfind(".");
    if (positionMap > 0 && positionMap < name.size()) {
      name = name.substr(0, positionMap);
    }
    name = FLAGS_txtoutputdir + "/" + name + ".txt";
    std::ofstream fileMap(name);

    float scaling_factors = std::min(
        static_cast<float>(input_dim) / static_cast<float>(images[i].cols),
        static_cast<float>(input_dim) / static_cast<float>(images[i].rows));

    for (int j = 0; j < result.size(); j++) {
      // ''' 得到在原图上的真实坐标'''
      result[j][0] =
          result[j][0] -
          static_cast<float>(input_dim - scaling_factors * images[i].cols) /2.0;
      result[j][2] =
          result[j][2] -
          static_cast<float>(input_dim - scaling_factors * images[i].cols) /
              2.0;
      result[j][1] =
          result[j][1] -
          static_cast<float>(input_dim - scaling_factors * images[i].rows) /
              2.0;
      result[j][3] =
          result[j][3] -
          static_cast<float>(input_dim - scaling_factors * images[i].rows) /
              2.0;
      // '''回归到原图坐标'''
      for (int k = 0; k < 4; k++) {
        result[j][k] = result[j][k] / scaling_factors;
      }
    }
    // 
    for (int j = 0; j < result.size(); j++) {
      // bounding boxes bundary check
      result[j][0] = result[j][0] < 0 ? 0 : result[j][0];
      result[j][2] = result[j][2] < 0 ? 0 : result[j][2];
      result[j][1] = result[j][1] < 0 ? 0 : result[j][1];
      result[j][3] = result[j][3] < 0 ? 0 : result[j][3];

      result[j][0] =
          result[j][0] > images[i].cols ? images[i].cols : result[j][0];
      result[j][2] =
          result[j][2] > images[i].cols ? images[i].cols : result[j][2];
      result[j][1] =
          result[j][1] > images[i].rows ? images[i].rows : result[j][1];
      result[j][3] =
          result[j][3] > images[i].rows ? images[i].rows : result[j][3];
    }
    for (int j = 0; j < result.size(); j++) {
      cv::Point p1(static_cast<int>(result[j][0]),
                   static_cast<int>(result[j][1]));
      cv::Point p2(static_cast<int>(result[j][2]),
                   static_cast<int>(result[j][3]));
      cv::rectangle(image, p1, p2, cv::Scalar(0, 0, 255), 1, 1, 0);
      cv::Point p3(static_cast<int>(result[j][0]),
                   static_cast<int>(result[j][1]) - 20);
      cv::Point p4(static_cast<int>(result[j][0]) + 100,
                   static_cast<int>(result[j][1]));
      cv::rectangle(image, p3, p4, cv::Scalar(255, 0, 0), -1, 4);
      stringstream ss;
      ss << round(result[j][4] * 1000) / 1000.0;
      std::string str =
          labelToDisplayName[static_cast<int>(result[j][6])] + ":" + ss.str();
      cv::Point p5(static_cast<int>(result[j][0]),
                   static_cast<int>(result[j][1]) - 1);
      cv::putText(image, str, p5, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(255, 215, 0), 1);
      
      // x1 y1 x2 y2 conf cls_score cls_id  
      fileMap << labelToDisplayName[static_cast<int>(result[j][6])] << " "
              << ss.str() << " " << static_cast<float>(p1.x) / image.cols << " "
              << static_cast<float>(p1.y) / image.rows << " "
              << static_cast<float>(p2.x) / image.cols << " "
              << static_cast<float>(p2.y) / image.rows << " " << image.cols
              << " " << image.rows << std::endl;
    }
    fileMap.close();
    stringstream ss;
    string outFile;
    int position = imageNames[i].find_last_of('/');
    string fileName(imageNames[i].substr(position + 1));
    string path = FLAGS_outputdir + "/" + "yolov3_";
    ss << path << fileName;
    ss >> outFile;

    std::cout<<"imwrite"<<std::endl;
    cv::imwrite(outFile.c_str(), image);
    // cv::imwrite((FLAGS_outputdir + "/" + imageNames[i].c_str()), image);
  }
}

cv::Mat convertTo3Channels(const cv::Mat& binImg)
{
    cv::Mat three_channel = cv::Mat::zeros(binImg.rows,binImg.cols,CV_8UC3);
    vector<cv::Mat> channels;
    for (int i=0;i<3;i++)
    {
        channels.push_back(binImg);
    }
    cv::merge(channels,three_channel);
    return three_channel;
}




int main(int argc, char** argv) {
{
  const char * env = getenv("log_prefix");
  if (!env || strcmp(env, "true") != 0)
    FLAGS_log_prefix = false;
}
  ::google::InitGoogleLogging(argv[0]);
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage(
      "Do detection using yolov3 mode.\n"
      "Usage:\n"
      "    yolov3_offline [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(argv[0],
                                       "examples/yolo_v3/yolov3_offline");
    return 1;
  }
  if (FLAGS_logdir != "") {
    FLAGS_log_dir = FLAGS_logdir;
  } else {
    //  log to terminal's stderr if no log path specified
    FLAGS_alsologtostderr = 1;
  }

  cnrtInit(0);
  /* Create Detector class */
  Detector* detector =
      new Detector(FLAGS_offlinemodel, FLAGS_meanfile, FLAGS_meanvalue);
  /* Load labels. */
std::vector<string> labels = {"飞机", "船只", "储蓄罐", "操场", "码头", "桥梁"};
//std::vector<string> labels = {"plane", "ship", "oil-tank", "court", "harbor", "bridge"};
//std::vector<string> labels;   
//std::ifstream labelsHandler(FLAGS_labels.c_str());
//   CHECK(labelsHandler) << "Unable to open labels file " << FLAGS_labels;
//  string line;
//   while (std::getline(labelsHandler, line)){ 
//std::cout<<"label:"<<line<<std::endl;     
//labels.push_back(line);
//}
//std::cout<<"label:"<<labels[0]<<labels[1]<<labels[2]<<std::endl;
//exit(0);
  // labelsHandler.close();

  // add 目前来看最简单的方法是提前切好
  // gdal 裁切 然后做自适应直方图均衡



// ori
  /* Load image files */
  // queue<string> imageListQueue;
  // int figuresNumber = 0;
  // string lineTemp;
  // std::ifstream filesHandler(FLAGS_images.c_str(), std::ios::in);
  // CHECK(!filesHandler.fail()) << "Image file is invalid!";
   
  // while (getline(filesHandler, lineTemp)) {
  //   imageListQueue.push(lineTemp);
  //   figuresNumber++;
  // }
  // filesHandler.close();
  // 读图片路径

//modify
//,"34th","42th"
vector<string> test_dir = {"12th","34th","42th"};//,"pic_all_3"

// opencv glob
vector<string> image_paths;
for (auto dir: test_dir)
{
    //modify
    string path = FLAGS_images+ "/" + dir + "/Tile/*.tif";//*.txt
    vector<cv::String> glob_list;
    
    cv::glob(path, glob_list, false);
    for (int i = 0; i < glob_list.size(); i++)
    {
        // images.push_back(imread(fn[i]));
        // imshow("pic", images[i]);
       image_paths.push_back(glob_list[i]);
        std::cout<<glob_list[i]<<std::endl;
    }
}
int figuresNumber = image_paths.size();

  std::cout << "there are " << figuresNumber << " figures in " << FLAGS_images << std::endl;
  

  // modify
  // yolov3
  // vector<int> output_stride = {13, 26, 52};   
  /* vector<vector<float>> anchor_predefine = {{116,90},{156,198},{373,326},
{30,61},{62,45},{59,119},                                             
{10,13},{16,30},{33,23}};*/
    // vector<vector<float>> anchor_predefine = {{373,326},{116,90},{156,198},
                                              //  {33,23},{10,13},{16,30},
 //                                           {59,119},{30,61},{62,45}};
// tiny head
vector<int> output_stride = {26, 13}; 
// 原始大小的百分比 乘以在相应特征图尺寸
vector<vector<float>> anchor_predefine = {{10,14}, {23,27},  {37,58}, {81,82}, {135,169}, {344,319}};

//add
//vector<int> output_stride = {13,26};
// ▒~N~_▒~K大▒~O▒~Z~D▒~Y▒▒~H~F▒~T ▒~X以▒~\▒▒~[▒▒~T▒~I▒▒~A▒~[▒尺寸
//vector<vector<float>> anchor_predefine = {{81,82}, {135,169}, {344,319}, {10,14}, {23,27},  {37,58}};

vector<vector<vector<float>>> x_y_offsets;
  vector<vector<vector<float>>> anchor_tensors;
  for(int i =0;i<output_stride.size(); i++)
  {
        int outputNum_curlayer = output_stride[i]*output_stride[i] * FLAGS_anchornum;
        x_y_offsets.push_back(vector<vector<float>>(outputNum_curlayer, vector<float>(2)));
        anchor_tensors.push_back(vector<vector<float>>(outputNum_curlayer, vector<float>(2)));
  }
// 填充值
  readtxt(&x_y_offsets, &anchor_tensors, output_stride, anchor_predefine);


  /* Detecting images */
  float timeUse;
  float totalTime = 0;
  struct timeval tpStart, tpEnd;

  // int batchesNumber = ceil(static_cast<float>(figuresNumber) / detector->getBatchSize());
  
  
  // '''detect by batch'''
  // for (int i = 0; i < batchesNumber; i++) {
  //   gettimeofday(&tpStart, NULL);
  //   vector<cv::Mat> images;
  //   vector<string> imageNames;
  //   // 先对每张裁切图进行识别
  //   detector->readImages(&imageListQueue, detector->getBatchSize(), &images,
  //                        &imageNames);
  //   //前向过程 
  //   // 对每张裁切图进行了nms
  //   vector<vector<vector<float>>> detections =
  //       detector->detect(images, x_y_offsets, anchor_tensors);

  //   // 保存结果 可视化
  //   if (FLAGS_dump) {
  //     if (!FLAGS_outputdir.empty()) {
  //       // WriteVisualizeBBox_offline(images, detections, labels, imageNames,
  //       //         detector->inputDim());
  //       // 保存中间结果
  //       WriteTempBBox_offline(detections, labels, imageNames,detector->inputDim())
  //     }
  //   }

  //   gettimeofday(&tpEnd, NULL);
  //   timeUse = 1000000 * (tpEnd.tv_sec - tpStart.tv_sec) + tpEnd.tv_usec -
  //             tpStart.tv_usec;
  //   totalTime += timeUse;
  //   std::cout << "Detecting execution time: " << timeUse << " us" << std::endl;
  //   for (int num = 0; num < detector->getBatchSize(); num++) {
  //     std::cout << "object is : " << detections[num].size() << std::endl;
  //   }
  //   images.clear();
  //   imageNames.clear();
  // }



// 保存所有图片的所有结果
std::ofstream out_file(FLAGS_outputdir+"/result.txt");
  
int batch_size = detector->getBatchSize();
//std::cout<<"batch_size "<<batch_size<<std::endl;
//exit(0);
bool visualize = false;

// 每个线程要切的图片id
    for(int i = 0 ; i< figuresNumber; ++i)
    {   std::cout<<i<<"/"<<figuresNumber<<std::endl;
        vector<vector<float>> all_results;

        const string img_path = image_paths[i];
        string dir_name, file_name;
         fileNameFromPath(img_path, dir_name, file_name);
        // const string img_name = fileNameFromPath(img_path).c_str();
        std::cout<<"dirname"<<dir_name<<" "<<file_name<<std::endl;

        cv::Mat src_img = cv::imread(img_path, -1);//ori depth
        double Min = 0, Max = 0; 
        cv::minMaxLoc(src_img, &Min, &Max);
        //std::cout<<"minmax"<<Min<<" "<<Max<<" "<<src_img.depth()<<std::endl;
        cv::Mat Mat_8;

        if (Min!=Max)
            src_img.convertTo(Mat_8, CV_8U, 255.0/(Max-Min),-255.0*Min/(Max-Min));//0
        
        //直方图均衡化
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));
        cv::Mat clahe_dst, merge_1c;
	      clahe->apply(Mat_8, clahe_dst);
        
        merge_1c =0.6*Mat_8 + 0.4*clahe_dst;//cv_8u
        cv::Mat merge = convertTo3Channels(merge_1c);//cv_8uc3

        const int img_h = src_img.rows;
        const int img_w = src_img.cols;
        const int dx = int(FLAGS_slice_size - FLAGS_overlap);
        const int dy = int(FLAGS_slice_size - FLAGS_overlap);
        
        const int h_step = (img_h - FLAGS_slice_size - 1)/dy+1;
        const int h_limit = h_step * dy ;
        const int w_step = (img_w - FLAGS_slice_size -1 )/dx+1;
        const int w_limit = w_step * dx;

        int total = (h_step+1)*(w_step+1);
        std::cout<<"total"<<total<<std::endl;
        vector<cv::Mat> images_batch;
        vector<string> imageNames;

        int count = 0;
        for(int y0=0; y0 <= h_limit ; y0 +=dy)
        {
            for(int x0=0; x0 <=w_limit ; x0 +=dx)
            {   
                std::cout<<"4096 4096 count: "<<count<<"/"<<total<<std::endl;
                cv::Mat dst_img;
                if ( FLAGS_slice_size > img_h || FLAGS_slice_size > img_w)
                {
                   //图部分替换(可用于补边)
                    cv::Mat matZeros,mat_bg;
                    mat_bg = cv::Mat::zeros(FLAGS_slice_size, FLAGS_slice_size, src_img.type());
                    // cv::Mat img(500, 500, CV_8U, Scalar(0));

                    // xywh
                    mat_bg = mat_bg(cv::Rect(0,0, src_img.cols, src_img.rows));
                    merge.copyTo(mat_bg);
                    mat_bg.copyTo(dst_img);

                }else{
                    int x,y;
                    // 如果最后图片只剩一小部分　两种处理方法
                    // 1裁切部分图片补边　2回退裁切完整图片(2)
                    // make sure we don't have a tiny image on the edge
                    if (y0+FLAGS_slice_size > img_h || x0 +FLAGS_slice_size > img_w)
                        y = img_h - FLAGS_slice_size;
                    else
                        y = y0;
                    if(x0+FLAGS_slice_size > img_w)
                        x = img_w - FLAGS_slice_size;
                    else
                        x = x0;
                    // std::cout<<x<<" "<<y << FLAGS_slice_size <<" "<< FLAGS_slice_size<<std::endl;
                    // 1,1,src.cols,src.rows   x y  w h
                    merge(cv::Rect(x, y, FLAGS_slice_size, FLAGS_slice_size)).copyTo(dst_img);
                }
                
                // '''''FLAGS_outputdir labels按顺序'''''
                char save_path[512];
                // std::cout<<"before snprintf:"<<std::endl;
                // FLAGS_outputdir.c_str()
                snprintf(save_path, 512, \
                "%s/%s_%s|%d_%d_%d_%d_%d_%d.png", FLAGS_outputdir.c_str(), dir_name.c_str(), file_name.c_str(),y0,x0,FLAGS_slice_size, FLAGS_slice_size, img_w, img_h);
                // imwrite(save_path, dst_img);
                
                images_batch.push_back(dst_img);
                imageNames.push_back(save_path);//保存完整路径为了写的时候恢复全局坐标
                ++count;
                // std::cout<<"count"<<count<<std::endl;

                // 每batch次检测
                if(count% batch_size==0 || count==total)
                {
                    if(images_batch.size() < batch_size)
                    {
                        for(int b=images_batch.size(); b< batch_size; ++b){
                            images_batch.push_back(images_batch[0]);
                            imageNames.push_back("null");
                        }
                    }

                    // process
                    //前向过程 
                    // 对每张裁切图进行了nms
                    //std::cout<<" start detect and nms----------"<<std::endl;
                    vector<vector<vector<float>>> detections = detector->detect(images_batch, x_y_offsets, anchor_tensors);

                    //std::cout<<" finish detect and nms"<<std::endl;
                    


                    // 中间结果 可视化
                    // WriteVisualizeBBox_offline(images, detections, labels, imageNames,
                    //         detector->inputDim());
                    // 得到原图坐标 global coord
                    // WriteTempBBox_offline(detections, labels, imageNames, detector->inputDim())
                    get_batch_global_coord(images_batch, 
                                           detections, 
                                           all_results, 
                                           labels, 
                                           imageNames,
                                           detector->inputDim(), 
                                           false);
                    


                    // gettimeofday(&tpEnd, NULL);
                    // timeUse = 1000000 * (tpEnd.tv_sec - tpStart.tv_sec) + tpEnd.tv_usec -
                    //           tpStart.tv_usec;
                    // totalTime += timeUse;
                    // std::cout << "Detecting execution time: " << timeUse << " us" << std::endl;
                    for (int num = 0; num < detector->getBatchSize(); num++) {
                      std::cout << "object is : " << detections[num].size() << std::endl;
                    }
                    images_batch.clear();
                    imageNames.clear();
                }//detect for each batch images
            }//for x
        }//for y

        std::cout<<"start nms"<<std::endl;
        // nms然后保存一张4096的结果到result.txt
        vector<vector<float>> global_nms_boxes;
        //global_nms_boxes = all_results;
        nms(all_results, &global_nms_boxes, FLAGS_nmsthresh);
        // file
        out_file <<dir_name<<"_"<<file_name<<std::endl;
        for (int j = 0; j < global_nms_boxes.size(); j++)
        {
                 if(global_nms_boxes[j][2] - global_nms_boxes[j][0] < 20 || global_nms_boxes[j][3]-global_nms_boxes[j][1]<20)
                  continue;
                // id cls_name conf  x1 y1 x2 y2
                stringstream ss_conf, ss_cls;
                // int str
                // ss_cls << result[j][6]
                ss_conf << round(global_nms_boxes[j][4] * 1000) / 1000.0;
                // ss_cls  << round(result[j][5] * 1000) / 1000.0;
                int id = j+1;
                out_file << id <<" "
                  << labels[static_cast<int>(global_nms_boxes[j][6])]<<" "
                  << ss_conf.str()   << " ("
                  << static_cast<int>(global_nms_boxes[j][0])  << ","
                  << static_cast<int>(global_nms_boxes[j][1])  << ") ("
                  << static_cast<int>(global_nms_boxes[j][2])  << ","
                  << static_cast<int>(global_nms_boxes[j][3])  << ")"<< std::endl;
        }


    // '''绘制大图结果'''
    if (visualize==true)
    {
      // dir_name_fiename
    //   name = FLAGS_outputdir + "/" + srcName + "_temp.txt";
    // //   '''add模式'''
    //   std::ofstream fileMap(name);
    std::cout<<"write result to big pic"<<std::endl;
      for (int j = 0; j < global_nms_boxes.size(); j++) {
        cv::Point p1(static_cast<int>(global_nms_boxes[j][0]),
                    static_cast<int>(global_nms_boxes[j][1]));
        cv::Point p2(static_cast<int>(global_nms_boxes[j][2]),
                    static_cast<int>(global_nms_boxes[j][3]));
        cv::rectangle(merge, p1, p2, cv::Scalar(0, 0, 255), 1, 1, 0);
        cv::Point p3(static_cast<int>(global_nms_boxes[j][0]),
                    static_cast<int>(global_nms_boxes[j][1]) - 20);
        cv::Point p4(static_cast<int>(global_nms_boxes[j][0]) + 100,
                    static_cast<int>(global_nms_boxes[j][1]));
        cv::rectangle(merge, p3, p4, cv::Scalar(255, 0, 0), -1, 4);
        stringstream ss;
        ss << round(global_nms_boxes[j][4] * 1000) / 1000.0;
        std::string str =
          labels[static_cast<int>(global_nms_boxes[j][6])] + ":" + ss.str();
        cv::Point p5(static_cast<int>(global_nms_boxes[j][0]),
                    static_cast<int>(global_nms_boxes[j][1]) - 1);
        cv::putText(merge, str, p5, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 215, 0), 1);


        // stringstream ss_conf, ss_cls;
        // // int str
        // // ss_cls << result[j][6]
        // // 
        // ss_conf << round(result[j][4] * 1000) / 1000.0;
        // ss_cls  << round(result[j][5] * 1000) / 1000.0;

        // // x1 y1 x2 y2 conf cls_score cls_id
        // fileMap << static_cast<int>(result[j][0])  << " "
        //         << static_cast<int>(result[j][1])  << " "
        //         << static_cast<int>(result[j][2])  << " "
        //         << static_cast<int>(result[j][3])  << " "
        //         << ss_conf.str()   << " "
        //         << ss_cls.str()  << " "
        //         << static_cast<int>(result[j][6])  << " "<< std::endl;
      }
      // fileMap.close();
      stringstream ss;
      string outFile;

      string path = FLAGS_outputdir + "/";
      ss << path << dir_name<<"_"<<file_name<<".png";
      ss >> outFile;
      cv::imwrite(outFile.c_str(), merge);
      // cv::imwrite((FLAGS_outputdir + "/" + imageNames[i].c_str()), merge);
    }
    std::cout<<"------------------------------------------------------------"<<std::endl;


    }//for imgs

out_file.close();


// std::cout<<" start nms"<<std::endl;
// '''post process for all result'''
//   // 对每张图的检测结果进行综合nms
//   vector<String> temp_result_lists;
//   cv::glob(FLAGS_txtoutputdir+'/*.txt', temp_result_lists, false);// 获取所有结果文件的路径
//    // 所有结果保存在一个txt下面
//   std::ofstream out_file(FLAGS_outputdir+'/result.txt');
  
//   for(int i=0;i<temp_result_lists.size(); ++i)
//   {
//     // 对每一个txt进行循环
//     string txt_name = temp_result_lists[i];//FLAGS_outputdir
    
//     std::string srcName
//     int x0,y0;
//     sliceInfo_from_FullPath(txt_name, &srcName, &x0, &y0);


//     std::ifstream filesHandler(txt_name.c_str(), std::ios::in);
//     CHECK(!filesHandler.fail()) << "Image file is invalid!";
//     // x1 y1 x2 y2 conf cls_score cls_id 
//     vector<vector<float>> slice_boxes;
//     string line;
//     stringstream ss;
//     // 对行进行循环
//     while (getline(filesHandler, line)) {/
//       ss.clear();
//       ss.str(line);
//       float temp;
//       vector<float> box;
//       // 对列进行循环
//       while(ss>>temp)
//       {
//         box.push_back(temp)
//       }
//       //将a的所有元素插入到b中
//       slice_boxes.push_back(box);
//     }
//     filesHandler.close();

// // 对一张图进行nms
//     vector<vector<float>> tmp_boxes;
//     // x1 y1 x2 y2 conf cls_score cls
//     nms(slice_boxes, tmp_boxes, nms_thresh)
    

//     for (int j = 0; j < result.size(); j++) {
//       // x1 y1 x2 y2 conf cls_score cls_id
//       fileMap << static_cast<int>(result[j][0])  << " "
//               << static_cast<int>(result[j][1])  << " "
//               << static_cast<int>(result[j][2])  << " "
//               << static_cast<int>(result[j][3])  << " "
//               << ss_conf.str()   << " "
//               << ss_cls.str()  << " "
//               << static_cast<int>(result[j][6])  << " "<< std::endl;
//     }


//   }//for every txt
  
  

  std::cout << "Total execution time: " << totalTime << " us"<< std::endl;

#if !defined(CROSS_COMPILE) && !defined(CROSS_COMPILE_ARM64)
  std::cout << "Hardware fps: " << figuresNumber / detector->mluTime() * 1e6
            << std::endl;
#endif  // CROSS_COMPILE && CROSS_COMPILE_ARM64
  std::cout << "End2end throughput fps: " << figuresNumber / totalTime * 1e6
            << std::endl;
  delete detector;
  cnrtDestroy();
  return 0;
}
#else
#include "caffe/common.hpp"
int main(int argc, char* argv[]) {
  LOG(FATAL) << "This program should be compiled with USE_MLU!";
  return 0;
}
#endif  // USE_MLU  && USE OPENCV
