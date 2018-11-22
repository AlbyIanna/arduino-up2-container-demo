/*

  // Copyright (c) 2018 Intel Corporation
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //      http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.
*/

/**
  \brief The entry point for the Inference Engine interactive_face_detection sample application
  \file object_detection_sample_ssd/main.cpp
  \example object_detection_sample_ssd/main.cpp
*/
#include "Arduino.h"
#undef min
#undef max
#undef round
#undef DEFAULT

#include <ArduinoOpenVINO.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>

#include <inference_engine.hpp>

#include <common.hpp>

#include "interactive_face_detection.hpp"
#include "mkldnn/mkldnn_extension_ptr.hpp"
#include <extension/ext_list.hpp>

#include <opencv2/opencv.hpp>
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"

using namespace InferenceEngine;

/*
  Please note that in order to run this sketch, you should have
  correctly installed the OpenVINO toolkit with its dependencies.
  Before running, make sure you have set the *user* variable with
  the name of the user currently running on your OS, otherwise
  the display will not show any output.
*/

// CHANGE THE user VARIABLE ACCORDING TO THE USER RUNNING ON YOUR OS 
String user = "upsquared";

std::string FLAGS_i = "/home/upsquared/web-cam-shot.jpg";

int FLAGS_n_ag = 16;
int FLAGS_n_hp = 16;
std::string FLAGS_m = "/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP16/face-detection-retail-0004.xml";
std::string FLAGS_m_ag = "/opt/intel/computer_vision_sdk/deployment_tools/intel_models/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml";
std::string FLAGS_m_hp = "/opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml";
std::string FLAGS_d = "GPU";
std::string FLAGS_d_ag = "GPU";
std::string FLAGS_d_hp = "GPU";
std::string FLAGS_l = "";
std::string FLAGS_c = "";
bool FLAGS_pc = false;
bool FLAGS_r = true;
bool FLAGS_no_show = true;
bool FLAGS_no_wait = false;
double FLAGS_t = 0.5;
bool isFound = false;


template <typename T>
void matU8ToBlob(const cv::Mat& orig_image, Blob::Ptr& blob, float scaleFactor = 1.0, int batchIndex = 0) {
  SizeVector blobSize = blob.get()->dims();
  const size_t width = blobSize[0];
  const size_t height = blobSize[1];
  const size_t channels = blobSize[2];
  T* blob_data = blob->buffer().as<T*>();

  cv::Mat resized_image(orig_image);
  if (width != orig_image.size().width || height != orig_image.size().height) {
    cv::resize(orig_image, resized_image, cv::Size(width, height));
  }

  int batchOffset = batchIndex * width * height * channels;

  for (size_t c = 0; c < channels; c++) {
    for (size_t  h = 0; h < height; h++) {
      for (size_t w = 0; w < width; w++) {
        blob_data[batchOffset + c * width * height + h * width + w] =
          resized_image.at<cv::Vec3b>(h, w)[c] * scaleFactor;
      }
    }
  }
}

// -------------------------Generic routines for detection networks-------------------------------------------------

struct BaseDetection {
  ExecutableNetwork net;
  InferenceEngine::InferencePlugin * plugin;
  InferRequest::Ptr request;
  std::string & commandLineFlag;
  std::string topoName;
  const int maxBatch;

  BaseDetection(std::string &commandLineFlag, std::string topoName, int maxBatch)
    : commandLineFlag(commandLineFlag), topoName(topoName), maxBatch(maxBatch) {}

  virtual ~BaseDetection() {}

  ExecutableNetwork* operator ->() {
    return &net;
  }
  virtual InferenceEngine::CNNNetwork read()  = 0;

  virtual void submitRequest() {
    if (!enabled() || request == nullptr) return;
    request->StartAsync();
  }

  virtual void wait() {
    if (!enabled() || !request) return;
    request->Wait(IInferRequest::WaitMode::RESULT_READY);
  }
  mutable bool enablingChecked = false;
  mutable bool _enabled = false;

  bool enabled() const  {
    if (!enablingChecked) {
      _enabled = !commandLineFlag.empty();
      if (!_enabled) {
        std::cout << "[ INFO ] " << topoName << " DISABLED" << std::endl;
      }
      enablingChecked = true;
    }
    return _enabled;
  }
  void printPerformanceCounts() {
    if (!enabled()) {
      return;
    }
    std::cout << "[ INFO ] Performance counts for " << topoName << std::endl << std::endl;
    ::printPerformanceCounts(request->GetPerformanceCounts(), std::cout, false);
  }
};

struct FaceDetectionClass : BaseDetection {
  std::string input;
  std::string output;
  int maxProposalCount;
  int objectSize;
  int enquedFrames = 0;
  float width = 0;
  float height = 0;
  bool resultsFetched = false;
  std::vector<std::string> labels;
  using BaseDetection::operator=;

  struct Result {
    int label;
    float confidence;
    cv::Rect location;
  };

  std::vector<Result> results;

  void submitRequest() override {
    if (!enquedFrames) return;
    enquedFrames = 0;
    resultsFetched = false;
    results.clear();
    BaseDetection::submitRequest();
  }

  void enqueue(const cv::Mat &frame) {
    if (!enabled()) return;

    if (!request) {
      request = net.CreateInferRequestPtr();
    }

    width = frame.cols;
    height = frame.rows;

    auto  inputBlob = request->GetBlob(input);

    matU8ToBlob<uint8_t >(frame, inputBlob);

    enquedFrames = 1;
  }


  FaceDetectionClass() : BaseDetection(FLAGS_m, "Face Detection", 1) {}
  InferenceEngine::CNNNetwork read() override {
    std::cout << "[ INFO ] Loading network files for Face Detection" << std::endl;
    InferenceEngine::CNNNetReader netReader;
    /** Read network model **/
    netReader.ReadNetwork(FLAGS_m);
    /** Set batch size to 1 **/
    std::cout << "[ INFO ] Batch size is set to  " << maxBatch << std::endl;
    netReader.getNetwork().setBatchSize(maxBatch);
    /** Extract model name and load it's weights **/
    std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
    netReader.ReadWeights(binFileName);
    /** Read labels (if any)**/
    std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";

    std::ifstream inputFile(labelFileName);
    std::copy(std::istream_iterator<std::string>(inputFile),
              std::istream_iterator<std::string>(),
              std::back_inserter(labels));
    // -----------------------------------------------------------------------------------------------------

    /** SSD-based network should have one input and one output **/
    // ---------------------------Check inputs ------------------------------------------------------
    std::cout << "[ INFO ] Checking Face Detection inputs" << std::endl;
    InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
    if (inputInfo.size() != 1) {
      throw std::logic_error("Face Detection network should have only one input");
    }
    auto& inputInfoFirst = inputInfo.begin()->second;
    inputInfoFirst->setPrecision(Precision::U8);
    inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
    // -----------------------------------------------------------------------------------------------------

    // ---------------------------Check outputs ------------------------------------------------------
    std::cout << "[ INFO ] Checking Face Detection outputs" << std::endl;
    InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
    if (outputInfo.size() != 1) {
      throw std::logic_error("Face Detection network should have only one output");
    }
    auto& _output = outputInfo.begin()->second;
    output = outputInfo.begin()->first;

    const auto outputLayer = netReader.getNetwork().getLayerByName(output.c_str());
    if (outputLayer->type != "DetectionOutput") {
      throw std::logic_error("Face Detection network output layer(" + outputLayer->name +
                             ") should be DetectionOutput, but was " +  outputLayer->type);
    }

    if (outputLayer->params.find("num_classes") == outputLayer->params.end()) {
      throw std::logic_error("Face Detection network output layer (" +
                             output + ") should have num_classes integer attribute");
    }

    const int num_classes = outputLayer->GetParamAsInt("num_classes");
    if (labels.size() != num_classes) {
      if (labels.size() == (num_classes - 1))  // if network assumes default "background" class, having no label
        labels.insert(labels.begin(), "fake");
      else
        labels.clear();
    }
    const InferenceEngine::SizeVector outputDims = _output->dims;
    maxProposalCount = outputDims[1];
    objectSize = outputDims[0];
    if (objectSize != 7) {
      throw std::logic_error("Face Detection network output layer should have 7 as a last dimension");
    }
    if (outputDims.size() != 4) {
      throw std::logic_error("Face Detection network output dimensions not compatible shoulld be 4, but was " +
                             std::to_string(outputDims.size()));
    }
    _output->setPrecision(Precision::FP32);
    _output->setLayout(Layout::NCHW);

    std::cout << "[ INFO ] Loading Face Detection model to the " << FLAGS_d << " plugin" << std::endl;
    input = inputInfo.begin()->first;
    return netReader.getNetwork();
  }

  void fetchResults() {
    isFound = false;
    if (!enabled()) return;
    results.clear();
    if (resultsFetched) return;
    resultsFetched = true;
    const float *detections = request->GetBlob(output)->buffer().as<float *>();

    for (int i = 0; i < maxProposalCount; i++) {
      float image_id = detections[i * objectSize + 0];
      Result r;
      r.label = static_cast<int>(detections[i * objectSize + 1]);
      r.confidence = detections[i * objectSize + 2];
      if (r.confidence <= FLAGS_t) {
        continue;
      }

      r.location.x = detections[i * objectSize + 3] * width;
      r.location.y = detections[i * objectSize + 4] * height;
      r.location.width = detections[i * objectSize + 5] * width - r.location.x;
      r.location.height = detections[i * objectSize + 6] * height - r.location.y;

      if (image_id < 0) {
        break;
      }
      if (FLAGS_r) {
        if (r.confidence > FLAGS_t) {
          isFound = true;
        }
        std::cout << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
                  "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
                  << r.location.height << ")"
                  << ((r.confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
      }

      results.push_back(r);
    }
  }
};

struct AgeGenderDetection : BaseDetection {
  std::string input;
  std::string outputAge;
  std::string outputGender;
  int enquedFaces = 0;


  using BaseDetection::operator=;
  AgeGenderDetection() : BaseDetection(FLAGS_m_ag, "Age Gender", FLAGS_n_ag) {}

  void submitRequest() override {
    if (!enquedFaces) return;
    BaseDetection::submitRequest();
    enquedFaces = 0;
  }

  void enqueue(const cv::Mat &face) {
    if (!enabled()) {
      return;
    }
    if (enquedFaces == maxBatch) {
      std::cout << "[ WARNING ] Number of detected faces more than maximum(" << maxBatch << ") processed by Age Gender detector" << std::endl;
      return;
    }
    if (!request) {
      request = net.CreateInferRequestPtr();
    }

    auto  inputBlob = request->GetBlob(input);

    matU8ToBlob<float>(face, inputBlob, 1.0f, enquedFaces);

    enquedFaces++;
  }

  struct Result {
    float age;
    float maleProb;
  };
  Result operator[] (int idx) const {
    auto  genderBlob = request->GetBlob(outputGender);
    auto  ageBlob    = request->GetBlob(outputAge);

    return {ageBlob->buffer().as<float*>()[idx] * 100,
            genderBlob->buffer().as<float*>()[idx * 2 + 1]};
  }

  CNNNetwork read() override {
    std::cout << "[ INFO ] Loading network files for AgeGender" << std::endl;
    InferenceEngine::CNNNetReader netReader;
    /** Read network model **/
    netReader.ReadNetwork(FLAGS_m_ag);

    /** Set batch size to 16 **/
    netReader.getNetwork().setBatchSize(maxBatch);
    std::cout << "[ INFO ] Batch size is set to " << netReader.getNetwork().getBatchSize() << " for Age Gender" << std::endl;


    /** Extract model name and load it's weights **/
    std::string binFileName = fileNameNoExt(FLAGS_m_ag) + ".bin";
    netReader.ReadWeights(binFileName);

    // -----------------------------------------------------------------------------------------------------

    /** Age Gender network should have one input two outputs **/
    // ---------------------------Check inputs ------------------------------------------------------
    std::cout << "[ INFO ] Checking Age Gender inputs" << std::endl;
    InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
    if (inputInfo.size() != 1) {
      throw std::logic_error("Age gender topology should have only one input");
    }
    auto& inputInfoFirst = inputInfo.begin()->second;
    inputInfoFirst->setPrecision(Precision::FP32);
    inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
    input = inputInfo.begin()->first;
    // -----------------------------------------------------------------------------------------------------

    // ---------------------------Check outputs ------------------------------------------------------
    std::cout << "[ INFO ] Checking Age Gender outputs" << std::endl;
    InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
    if (outputInfo.size() != 2) {
      throw std::logic_error("Age Gender network should have two output layers");
    }
    auto it = outputInfo.begin();
    auto ageOutput = (it++)->second;
    auto genderOutput = (it++)->second;

    // if gender output is convolution, it can be swapped with age
    if (genderOutput->getCreatorLayer().lock()->type == "Convolution") {
      std::swap(ageOutput, genderOutput);
    }

    if (ageOutput->getCreatorLayer().lock()->type != "Convolution") {
      throw std::logic_error("In Age Gender network, age layer (" + ageOutput->getCreatorLayer().lock()->name +
                             ") should be a Convolution, but was: " + ageOutput->getCreatorLayer().lock()->type);
    }

    if (genderOutput->getCreatorLayer().lock()->type != "SoftMax") {
      throw std::logic_error("In Age Gender network, gender layer (" + genderOutput->getCreatorLayer().lock()->name +
                             ") should be a SoftMax, but was: " + genderOutput->getCreatorLayer().lock()->type);
    }
    std::cout << "[ INFO ] Age layer: " << ageOutput->getCreatorLayer().lock()->name << std::endl;
    std::cout << "[ INFO ] Gender layer: " << genderOutput->getCreatorLayer().lock()->name << std::endl;

    outputAge = ageOutput->name;
    outputGender = genderOutput->name;

    std::cout << "[ INFO ] Loading Age Gender model to the " << FLAGS_d_ag << " plugin" << std::endl;
    _enabled = true;
    return netReader.getNetwork();
  }
};

struct HeadPoseDetection : BaseDetection {
  std::string input;
  std::string outputAngleR = "angle_r_fc";
  std::string outputAngleP = "angle_p_fc";
  std::string outputAngleY = "angle_y_fc";
  int enquedFaces = 0;
  cv::Mat cameraMatrix;
  HeadPoseDetection() : BaseDetection(FLAGS_m_hp, "Head Pose", FLAGS_n_hp) {}

  void submitRequest() override {
    if (!enquedFaces) return;
    BaseDetection::submitRequest();
    enquedFaces = 0;
  }

  void enqueue(const cv::Mat &face) {
    if (!enabled()) {
      return;
    }
    if (enquedFaces == maxBatch) {
      std::cout << "[ WARNING ] Number of detected faces more than maximum(" << maxBatch << ") processed by Head Pose detector" << std::endl;
      return;
    }
    if (!request) {
      request = net.CreateInferRequestPtr();
    }

    auto  inputBlob = request->GetBlob(input);

    matU8ToBlob<float>(face, inputBlob, 1.0f, enquedFaces);

    enquedFaces++;
  }

  struct Results {
    float angle_r;
    float angle_p;
    float angle_y;
  };

  Results operator[] (int idx) const {
    auto  angleR = request->GetBlob(outputAngleR);
    auto  angleP = request->GetBlob(outputAngleP);
    auto  angleY = request->GetBlob(outputAngleY);

    return {angleR->buffer().as<float*>()[idx],
            angleP->buffer().as<float*>()[idx],
            angleY->buffer().as<float*>()[idx]};
  }

  CNNNetwork read() override {
    std::cout << "[ INFO ] Loading network files for Head Pose detection " << std::endl;
    InferenceEngine::CNNNetReader netReader;
    /** Read network model **/
    netReader.ReadNetwork(FLAGS_m_hp);
    /** Set batch size to maximum currently set to one provided from command line **/
    netReader.getNetwork().setBatchSize(maxBatch);
    netReader.getNetwork().setBatchSize(maxBatch);
    std::cout << "[ INFO ] Batch size is sey to  " << netReader.getNetwork().getBatchSize() << " for Head Pose Network" << std::endl;
    /** Extract model name and load it's weights **/
    std::string binFileName = fileNameNoExt(FLAGS_m_hp) + ".bin";
    netReader.ReadWeights(binFileName);


    /** Age Gender network should have one input two outputs **/
    // ---------------------------Check inputs ------------------------------------------------------
    std::cout << "[ INFO ]  Checking Head Pose Network inputs" << std::endl;
    InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
    if (inputInfo.size() != 1) {
      throw std::logic_error("Head Pose topology should have only one input");
    }
    auto& inputInfoFirst = inputInfo.begin()->second;
    inputInfoFirst->setPrecision(Precision::FP32);
    inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
    input = inputInfo.begin()->first;
    // -----------------------------------------------------------------------------------------------------

    // ---------------------------Check outputs ------------------------------------------------------
    std::cout << "[ INFO ]  Checking Head Pose network outputs" << std::endl;
    InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
    if (outputInfo.size() != 3) {
      throw std::logic_error("Head Pose network should have 3 outputs");
    }
    std::map<std::string, bool> layerNames = {
      {outputAngleR, false},
      {outputAngleP, false},
      {outputAngleY, false}
    };

    for (auto && output : outputInfo) {
      auto layer = output.second->getCreatorLayer().lock();
      if (layerNames.find(layer->name) == layerNames.end()) {
        throw std::logic_error("Head Pose network output layer unknown: " + layer->name + ", should be " +
                               outputAngleR + " or " + outputAngleP + " or " + outputAngleY);
      }
      if (layer->type != "FullyConnected") {
        throw std::logic_error("Head Pose network output layer (" + layer->name + ") has invalid type: " +
                               layer->type + ", should be FullyConnected");
      }
      auto fc = dynamic_cast<FullyConnectedLayer*>(layer.get());
      if (fc->_out_num != 1) {
        throw std::logic_error("Head Pose network output layer (" + layer->name + ") has invalid out-size=" +
                               std::to_string(fc->_out_num) + ", should be 1");
      }
      layerNames[layer->name] = true;
    }

    std::cout << "[ INFO ]  Loading Head Pose model to the " << FLAGS_d_hp << " plugin" << std::endl;

    _enabled = true;
    return netReader.getNetwork();
  }

  void buildCameraMatrix(int cx, int cy, float focalLength) {
    if (!cameraMatrix.empty()) return;
    cameraMatrix = cv::Mat::zeros(3, 3, CV_32F);
    cameraMatrix.at<float>(0) = focalLength;
    cameraMatrix.at<float>(2) = static_cast<float>(cx);
    cameraMatrix.at<float>(4) = focalLength;
    cameraMatrix.at<float>(5) = static_cast<float>(cy);
    cameraMatrix.at<float>(8) = 1;
  }

  void drawAxes(cv::Mat& frame, cv::Point3f cpoint, Results headPose, float scale) {
    double yaw   = headPose.angle_y;
    double pitch = headPose.angle_p;
    double roll  = headPose.angle_r;

    if (FLAGS_r) {
      std::cout << "Head pose results: yaw, pitch, roll = " << yaw << ";" << pitch << ";" << roll << std::endl;
    }

    pitch *= CV_PI / 180.0;
    yaw   *= CV_PI / 180.0;
    roll  *= CV_PI / 180.0;

    cv::Matx33f        Rx(1,           0,            0,
                          0,  cos(pitch),  -sin(pitch),
                          0,  sin(pitch),  cos(pitch));
    cv::Matx33f Ry(cos(yaw),           0,    -sin(yaw),
                   0,           1,            0,
                   sin(yaw),           0,    cos(yaw));
    cv::Matx33f Rz(cos(roll), -sin(roll),            0,
                   sin(roll),  cos(roll),            0,
                   0,           0,            1);


    auto r = cv::Mat(Rz * Ry * Rx);
    buildCameraMatrix(frame.cols / 2, frame.rows / 2, 950.0);

    cv::Mat xAxis(3, 1, CV_32F), yAxis(3, 1, CV_32F), zAxis(3, 1, CV_32F), zAxis1(3, 1, CV_32F);

    xAxis.at<float>(0) = 1 * scale;
    xAxis.at<float>(1) = 0;
    xAxis.at<float>(2) = 0;

    yAxis.at<float>(0) = 0;
    yAxis.at<float>(1) = -1 * scale;
    yAxis.at<float>(2) = 0;

    zAxis.at<float>(0) = 0;
    zAxis.at<float>(1) = 0;
    zAxis.at<float>(2) = -1 * scale;

    zAxis1.at<float>(0) = 0;
    zAxis1.at<float>(1) = 0;
    zAxis1.at<float>(2) = 1 * scale;

    cv::Mat o(3, 1, CV_32F, cv::Scalar(0));
    o.at<float>(2) = cameraMatrix.at<float>(0);

    xAxis = r * xAxis + o;
    yAxis = r * yAxis + o;
    zAxis = r * zAxis + o;
    zAxis1 = r * zAxis1 + o;

    cv::Point p1, p2;

    p2.x = static_cast<int>((xAxis.at<float>(0) / xAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
    p2.y = static_cast<int>((xAxis.at<float>(1) / xAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
    cv::line(frame, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 0, 255), 2);

    p2.x = static_cast<int>((yAxis.at<float>(0) / yAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
    p2.y = static_cast<int>((yAxis.at<float>(1) / yAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
    cv::line(frame, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 255, 0), 2);

    p1.x = static_cast<int>((zAxis1.at<float>(0) / zAxis1.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
    p1.y = static_cast<int>((zAxis1.at<float>(1) / zAxis1.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);

    p2.x = static_cast<int>((zAxis.at<float>(0) / zAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
    p2.y = static_cast<int>((zAxis.at<float>(1) / zAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
    cv::line(frame, p1, p2, cv::Scalar(255, 0, 0), 2);

    cv::circle(frame, p2, 3, cv::Scalar(255, 0, 0), 2);
  }
};

struct Load {
  BaseDetection& detector;
  explicit Load(BaseDetection& detector) : detector(detector) { }

  void into(InferenceEngine::InferencePlugin & plg) const {
    if (detector.enabled()) {
      detector.net = plg.LoadNetwork(detector.read(), {});
      detector.plugin = &plg;
    }
  }
};

void switchUser(String user) {
  String uid = System.runShellCommand("id -u " + user);
  String gid = System.runShellCommand("id -g " + user);

  setuid(uid.toInt());
  setgid(gid.toInt());
}

int main_function() {
  try {
    switchUser(user);

    /** This sample covers 3 certain topologies and cannot be generalized **/
    std::cout << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;

    // -----------------------------Read input -----------------------------------------------------
    std::cout << "[ INFO ]  Reading input" << std::endl;
    cv::VideoCapture cap;
    const bool isCamera = FLAGS_i == "cam";

    if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
      throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
    }
    const size_t width  = (size_t) cap.get(CV_CAP_PROP_FRAME_WIDTH);
    const size_t height = (size_t) cap.get(CV_CAP_PROP_FRAME_HEIGHT);

    // read input (video) frame
    cv::Mat frame;
    if (!cap.read(frame)) {
      throw std::logic_error("Failed to get frame from cv::VideoCapture");
    }

    // ---------------------Load plugins for inference engine------------------------------------------------
    std::map<std::string, InferencePlugin> pluginsForDevices;
    std::vector<std::pair<std::string, std::string>> cmdOptions = {
      {FLAGS_d, FLAGS_m}, {FLAGS_d_ag, FLAGS_m_ag}, {FLAGS_d_hp, FLAGS_m_hp}
    };

    FaceDetectionClass FaceDetection;
    AgeGenderDetection AgeGender;
    HeadPoseDetection HeadPose;

    std::string ld_path =  std::string(getenv("LD_LIBRARY_PATH"));
    std::vector<std::string> paths;
    std::istringstream f(ld_path);
    std::string s;
    while (std::getline(f, s, ':')) {
      paths.push_back(s);
    }

    std::vector<std::string> pluginPaths;
    for (int i = 0; i < paths.size(); i++) {
      pluginPaths.push_back(paths[i]);
    }

    PluginDispatcher pluginDispatcher = PluginDispatcher(pluginPaths);

    for (auto && option : cmdOptions) {
      auto deviceName = option.first;
      auto networkName = option.second;

      if (deviceName == "" || networkName == "") {
        continue;
      }

      if (pluginsForDevices.find(deviceName) != pluginsForDevices.end()) {
        continue;
      }
      std::cout << "[ INFO ]  Loading plugin " << deviceName << std::endl;
      InferencePlugin plugin = pluginDispatcher.getPluginByDevice(deviceName);

      /** Printing plugin version **/
      printPluginVersion(plugin, std::cout);

      /** Load extensions for the CPU plugin **/
      if ((deviceName.find("CPU") != std::string::npos)) {
        plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

        if (!FLAGS_l.empty()) {
          // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
          auto extension_ptr = make_so_pointer<InferenceEngine::MKLDNNPlugin::IMKLDNNExtension>(FLAGS_l);
          plugin.AddExtension(std::static_pointer_cast<IExtension>(extension_ptr));
        }
      } else if (!FLAGS_c.empty()) {
        // Load Extensions for other plugins not CPU
        plugin.SetConfig({ { PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c } });
      }
      pluginsForDevices[deviceName] = plugin;
    }

    /** Per layer metrics **/
    if (FLAGS_pc) {
      for (auto && plugin : pluginsForDevices) {
        plugin.second.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
      }
    }


    // --------------------Load networks (Generated xml/bin files)-------------------------------------------

    Load(FaceDetection).into(pluginsForDevices[FLAGS_d]);
    Load(AgeGender).into(pluginsForDevices[FLAGS_d_ag]);
    Load(HeadPose).into(pluginsForDevices[FLAGS_d_hp]);


    // ----------------------------Do inference-------------------------------------------------------------
    std::cout << "[ INFO ]  Start inference " << std::endl;
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    auto wallclock = std::chrono::high_resolution_clock::now();

    double ocv_decode_time = 0, ocv_render_time = 0;
    bool firstFrame = true;
    cv::Mat prev_frame, next_frame;
    
    /** Start inference & calc performance **/
    while (true) {
      /** requesting new frame if any*/
      cap.grab();

      auto t0 = std::chrono::high_resolution_clock::now();
      FaceDetection.enqueue(frame);
      auto t1 = std::chrono::high_resolution_clock::now();
      ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

      t0 = std::chrono::high_resolution_clock::now();
      // ----------------------------Run face detection inference------------------------------------------
      FaceDetection.submitRequest();
      FaceDetection.wait();

      t1 = std::chrono::high_resolution_clock::now();
      ms detection = std::chrono::duration_cast<ms>(t1 - t0);

      FaceDetection.fetchResults();

      for (auto && face : FaceDetection.results) {
        
        if (AgeGender.enabled() || HeadPose.enabled()) {
          auto clippedRect = face.location & cv::Rect(0, 0, width, height);
          auto face = frame(clippedRect);
          AgeGender.enqueue(face);
          HeadPose.enqueue(face);
        }
      }
      // ----------------------------Run age-gender, and head pose detection simultaneously----------------
      t0 = std::chrono::high_resolution_clock::now();
      if (AgeGender.enabled() || HeadPose.enabled()) {
        AgeGender.submitRequest();
        HeadPose.submitRequest();

        AgeGender.wait();
        HeadPose.wait();
      }
      t1 = std::chrono::high_resolution_clock::now();
      ms secondDetection = std::chrono::duration_cast<ms>(t1 - t0);

      // ----------------------------Processing outputs-----------------------------------------------------
      std::ostringstream out;
      out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
          << (ocv_decode_time + ocv_render_time) << " ms";
      cv::putText(frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));

      out.str("");
      out << "Face detection time  : " << std::fixed << std::setprecision(2) << detection.count()
          << " ms ("
          << 1000.f / detection.count() << " fps)";
      cv::putText(frame, out.str(), cv::Point2f(0, 45), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                  cv::Scalar(255, 0, 0));

      if (HeadPose.enabled() || AgeGender.enabled()) {
        out.str("");
        out << (AgeGender.enabled() ? "Age Gender"  : "")
            << (AgeGender.enabled() && HeadPose.enabled() ? "+"  : "")
            << (HeadPose.enabled() ? "Head Pose "  : "")
            << "time: " << std::fixed << std::setprecision(2) << secondDetection.count()
            << " ms ";
        if (!FaceDetection.results.empty()) {
          out << "(" << 1000.f / secondDetection.count() << " fps)";
        }
        cv::putText(frame, out.str(), cv::Point2f(0, 65), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));
      }

      int i = 0;
      for (auto & result : FaceDetection.results) {
        cv::Rect rect = result.location;

        out.str("");

        if (AgeGender.enabled() && i < AgeGender.maxBatch) {
          out << (AgeGender[i].maleProb > 0.5 ? "M" : "F");
          out << std::fixed << std::setprecision(0) << "," << AgeGender[i].age;
        } else {
          out << (result.label < FaceDetection.labels.size() ? FaceDetection.labels[result.label] :
                  std::string("label #") + std::to_string(result.label))
              << ": " << std::fixed << std::setprecision(3) << result.confidence;
        }

        cv::putText(frame,
                    out.str(),
                    cv::Point2f(result.location.x, result.location.y - 15),
                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                    1.2,
                    cv::Scalar(0, 0, 255));

        if (FLAGS_r) {
          std::cout << "Predicted gender, age = " << out.str() << std::endl;
        }

        if (HeadPose.enabled() && i < HeadPose.maxBatch) {
          cv::Point3f center(rect.x + rect.width / 2, rect.y + rect.height / 2, 0);
          HeadPose.drawAxes(frame, center, HeadPose[i], 50);
        }

        auto genderColor = (AgeGender.enabled() && (i < AgeGender.maxBatch)) ?
                           ((AgeGender[i].maleProb < 0.5) ? cv::Scalar(147, 20, 255) : cv::Scalar(255, 0, 0)) :
                           cv::Scalar(100, 100, 100);

        cv::rectangle(frame, result.location, genderColor, 3);

        i++;
      }
      
      if (isFound) {
          std::vector<int> compression_params;
          compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
          compression_params.push_back(9);
          try {
              cv::imwrite("/home/upsquared/detected_face.png", frame, compression_params);
          }
          catch (std::runtime_error& ex) {
              fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
              return 1;
          }
      }

      if (-1 != cv::waitKey(1))
        break;

      t0 = std::chrono::high_resolution_clock::now();
      if (!FLAGS_no_show) {
        cv::imshow("Detection results", frame);
      }
      t1 = std::chrono::high_resolution_clock::now();
      ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();

      // end of file, for single frame file, like image we just keep it displayed to let user check what was shown
      if (!cap.retrieve(frame)) {
        if (!FLAGS_no_wait) {
          std::cout << "[ INFO ]  Press any key to exit" << std::endl;
          cv::waitKey(0);
        }
        break;
      }

      prev_frame = frame;
      frame = next_frame;
      next_frame = cv::Mat();
      firstFrame = false;
    }

    // ---------------------------Some perf data--------------------------------------------------
    if (FLAGS_pc) {
      FaceDetection.printPerformanceCounts();
      AgeGender.printPerformanceCounts();
      HeadPose.printPerformanceCounts();
    }
  }
  catch (const std::exception& error) {
    std::cerr << "[ ERROR ] " << error.what() << std::endl;
    return 1;
  }
  catch (...) {
    std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
    return 1;
  }

  std::cout << "[ INFO ] Execution successful" << std::endl;
  return 0;
}


void setup() {
  
  int err = runAsync("echo LD_LIBRARY_PATH:");
  checkError(err, "Failed to run command 1                                                                    .");
  err = runAsync("echo $LD_LIBRARY_PATH > /home/upsquared/a.out");          
  checkError(err, "Failed to run command 2.");
  delay (3000);
  main_function();
}

void loop() {
  // put your main code here, to run repeatedly:
  exit(0);
}

int runAsync(String command) {
  return runAsyncPath(command, "/root");
}

int runAsyncPath(String command, char* path) {
  Process p;
  if(path!=NULL) {
    p.changeDirectory(path);
  }
  p.runShellCommandAsynchronously(command);
  while(p.running()) {
    if(p.available()) {
      DebugSerial.print((char)p.read());
    }
  }
  return p.exitValue();
}

void checkError(int err, String errorMessage) {
  if(err != 0) {
    DebugSerial.println("\n.\n.\n.\nError: " + errorMessage + "\n\tTerminating sketch!\n\n");
    exit(0);
  }
}