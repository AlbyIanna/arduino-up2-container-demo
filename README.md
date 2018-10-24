# arduino-up2-container-demo
An example of use of the Arduino docker containers feature with UP² board (UP Squared) and OpenVINO

# Connect Up2
- plug to power and pc via usb cable
- do getting started (install connector and docker)
- go to [Arduino Device Manager](https://create.arduino.cc/devices/) and open up2
- go to network, get ip and ssh into it
  ```bash
  ssh upsquared@<ip_address>
  ```
- if you don't want to use sudo when using docker you have to create a group docker and add your user to the latter:
  ```bash
  sudo groupadd docker
  sudo usermod -aG docker $USER
  ```
  (You may have to log out and log in again to see the results)
- go to http://10.130.22.221:32768
- install open vino:
  ```
  sudo su
  cd
  apt update
  ls -d /opt/intel/computer_vision_sdk*/
  ls ./l_openvino_toolkit*.tgz
  tar -xvzf ./l_openvino_toolkit*.tgz
  find ./ -maxdepth 1 -type d  -regex '.*/l_openvino_toolkit.*' -exec mv {} ./l_openvino_toolkit \;
  cd l_openvino_toolkit 
  ./install_cv_sdk_dependencies.sh
  sed -i 's/ACCEPT_EULA=decline/ACCEPT_EULA=accept/' ./silent.cfg
  ./install.sh --silent ./silent.cfg
  apt install libnss3 libpulse-mainloop-glib0 libxslt1.1 libqt5sql5-mysql -y
  source /opt/intel/computer_vision_sdk/bin/setupvars.sh
  cd  /opt/intel/computer_vision_sdk/install_dependencies/
  ./_install_all_dependencies.sh
  usermod -a -G video upsquared
  ```

# images
- Getting Started landing page, UP^2 Grove Kit selected
![Getting Started landing page, UP^2 Grove Kit selected](https://user-images.githubusercontent.com/6939054/47091288-e48e8080-d224-11e8-8985-71c2611d9c42.png)
- Device Manager landing page 
![Device Manager landing page](https://user-images.githubusercontent.com/6939054/47099868-7e125e00-d236-11e8-9a97-e463d97bcb11.png)

# links
- [Node-RED](https://nodered.org/)
- [Docker](https://www.docker.com/)
- [Arduino Device Manager](https://create.arduino.cc/devices/)
- []()
