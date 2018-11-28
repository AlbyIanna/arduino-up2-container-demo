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

#include "face_detection_photo.h"
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

std::string FLAGS_i = "cam";

std::string FLAGS_m = "/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP16/face-detection-retail-0004.xml";
std::string FLAGS_d = "GPU";
std::string FLAGS_l = "";
std::string FLAGS_c = "";
bool FLAGS_pc = false;
bool FLAGS_r = false;
bool FLAGS_no_show = false;
bool FLAGS_no_wait = false;
double FLAGS_t = 0.9;
bool isFound = false;
int lastFound = 0;
int picsTook = 0;
int firstFoundInSeries = 0;
bool picSeries = false;
cv::Mat framesWithFaces [5];
    
template <typename T>
void matU8ToBlob(const cv::Mat& orig_image, Blob::Ptr& blob, float scaleFactor = 1.0, int batchIndex = 0) {
  SizeVector blobSize = blob.get()->dims();
  const size_t width = blobSize[0];
  const size_t height = blobSize[1];
  const size_t channels = blobSize[2];
  T* blob_data = blob->buffer().as<T*>();

  cv::Mat resized_image(orig_image);
  if (width != orig_image.size().width || height != orig_image.size().height) {
    if (orig_image.empty()) {
      return;
    }else {
      cv::resize(orig_image, resized_image, cv::Size(width, height));
    }
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
    if (resultsFetched) {
      return;
    }
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
      
      isFound = true;
      if (FLAGS_r) {

        std::cout << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
                  "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
                  << r.location.height << ")"
                  << ((r.confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
      }

      results.push_back(r);
    }
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
      {FLAGS_d, FLAGS_m}, {}, {}
    };

    FaceDetectionClass FaceDetection;

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


    // ----------------------------Do inference-------------------------------------------------------------
    std::cout << "[ INFO ]  Start inference " << std::endl;
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    auto wallclock = std::chrono::high_resolution_clock::now();

    double ocv_decode_time = 0, ocv_render_time = 0;
    bool firstFrame = true;
    
    /** Start inference & calc performance **/
    while (true) {
      /** requesting new frame if any*/
      cap.grab();

      auto t0 = std::chrono::high_resolution_clock::now();
      FaceDetection.enqueue(frame);
      auto t1 = std::chrono::high_resolution_clock::now();
      ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();
            FaceDetection.submitRequest();
      FaceDetection.wait();

      FaceDetection.fetchResults();

      // ----------------------------Processing outputs-----------------------------------------------------
      std::ostringstream out;
      out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
          << (ocv_decode_time + ocv_render_time) << " ms";
      cv::putText(frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));

      out.str("");
      cv::putText(frame, out.str(), cv::Point2f(0, 45), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                  cv::Scalar(255, 0, 0));

      int i = 0;
      for (auto & result : FaceDetection.results) {
        cv::Rect rect = result.location;

        out.str("");

        out << (result.label < FaceDetection.labels.size() ? FaceDetection.labels[result.label] :
                std::string("label #") + std::to_string(result.label))
            << ": " << std::fixed << std::setprecision(3) << result.confidence;

        cv::putText(frame,
                    out.str(),
                    cv::Point2f(result.location.x, result.location.y - 15),
                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                    1.2,
                    cv::Scalar(0, 0, 255));

        cv::rectangle(frame, result.location, cv::Scalar(147, 20, 255), 3);

        i++;
      }
      
      if (isFound) {
        if ((millis() - lastFound) > 1000) {
          if (!picSeries) {
            firstFoundInSeries = millis();
            picSeries = true;
          }
          DebugSerial.println("FACE FOUND!");
          framesWithFaces[picsTook++] = frame;
          lastFound = millis();
        }
      }
      
      if (picSeries && (millis() - firstFoundInSeries) > 5000) {
        picSeries = false;
        savePics();
        DebugSerial.println("mail sent");
        System.runShellCommand("curl -s -X POST http://localhost:32786/face-found");
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

      firstFrame = false;
    }

    // ---------------------------Some perf data--------------------------------------------------
    if (FLAGS_pc) {
      FaceDetection.printPerformanceCounts();
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

void savePics() {
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    for (int i = 0; i < picsTook; i++) {
      char fileName[50];
      sprintf(fileName, "/tmp/node/detected_face_%d.png", i);
      try {
        cv::imwrite(fileName, framesWithFaces[i], compression_params);
        DebugSerial.print("file ");
        DebugSerial.print(fileName);
        DebugSerial.println(" saved");
      }
      catch (std::runtime_error& ex) {
        DebugSerial.print("failed to save file ");
        DebugSerial.print(fileName);
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        return;
      }
    }
    picsTook = 0;
    std::memset( framesWithFaces, 0, sizeof( framesWithFaces ) );
}