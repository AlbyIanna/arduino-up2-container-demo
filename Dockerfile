FROM ubuntu:16.04

RUN apt update && apt install -y \
    libnss3 \
    libpulse-mainloop-glib0 \
    libxslt1.1 \
    libqt5sql5-mysql \
    sudo \
    cpio \
    fswebcam \
    lsb-core \
    && rm -rf /var/lib/apt/lists/*
RUN pwd
ENV vino_dir /opt/intel/computer_vision_sdk
COPY l_openvino_toolkit_p_2018.3.343.tgz .
RUN tar -xvzf ./l_openvino_toolkit*.tgz
RUN find ./ -maxdepth 1 -type d  -regex '.*/l_openvino_toolkit.*' -exec mv {} ./l_openvino_toolkit \;
ENV install_dir ./l_openvino_toolkit
WORKDIR ${install_dir}
RUN ./install_cv_sdk_dependencies.sh
RUN sed -i 's/ACCEPT_EULA=decline/ACCEPT_EULA=accept/' ./silent.cfg
RUN ./install.sh --silent ./silent.cfg
RUN /bin/bash -c "source ${vino_dir}/bin/setupvars.sh; ${vino_dir}/install_dependencies/_install_all_dependencies.sh"

# BUILD DEMO
RUN mkdir -p ${vino_dir}/deployment_tools/inference_engine/samples/build/
WORKDIR ${vino_dir}/deployment_tools/inference_engine/samples/build/
COPY main.cpp /
RUN cat /main.cpp
RUN rm -f ../interactive_face_detection_sample/main.cpp \
    && mv /main.cpp ../interactive_face_detection_sample/main.cpp
RUN ls -la ../interactive_face_detection_sample/
RUN cat ../interactive_face_detection_sample/main.cpp
RUN /bin/bash -c "source ${vino_dir}/bin/setupvars.sh; cmake -DCMAKE_BUILD_TYPE=Release .."
RUN /bin/bash -c "source ${vino_dir}/bin/setupvars.sh; make -j8 interactive_face_detection_sample"
WORKDIR ${vino_dir}/deployment_tools/inference_engine/samples/build/intel64/Release
CMD /bin/bash -c "source ${vino_dir}/bin/setupvars.sh; fswebcam -r 640x480 --jpeg 85 -D 1 web-cam-shot.jpg; ./interactive_face_detection_sample -i web-cam-shot.jpg -m /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP32/face-detection-retail-0004.xml -d CPU --no-show -r"
