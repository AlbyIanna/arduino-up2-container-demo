### tldr;
Smart doorbell that sends a mail when recognizing a face and can open the door via cloud.

### Project
The scope of this project is to demonstrate how to create an IoT setup in order to learn to:
- create [connected Things](https://en.wikipedia.org/wiki/Internet_of_things) on [Arduino IoT Cloud](https://create.arduino.cc/cloud)
- deploy [Docker](https://www.docker.com/resources/what-container) containers on an [UP SQUARED](https://up-board.org/upsquared/specifications/) board from [Arduino Device Manager](https://create.arduino.cc/devices/)
- setup a [Node-RED](https://nodered.org/) container to send a mail when a certain event is triggered
- use the Arduino [OpenVINO](https://software.intel.com/en-us/openvino-toolkit) library to perform face detection

First we are going to create the **door-opening Thing** and control it with a **dashboard** from the browser. Then we'll configure the UP SQUARED and deploy a Node-RED container set to **send a mail** with the file containing the photo taken by the camera **when it detects a face**. And finally we will remotely upload on the UP SQUARED a **sketch that performs face detection** using Arduino OpenVINO.

With all this set up, it will result in a camera that takes a picture when detects a face, which will be sent via mail by the Node-RED container, so that the mail will contain the picture and a link to the Arduino IoT dashboard, where you can push a button to open the door.

### Requirements
- an [Arduino Create](https://create.arduino.cc/) account. If you don't have one, [you can create it now](https://auth.arduino.cc/login/), it's free :)
- Arduino [MKR WiFi 1010](https://store.arduino.cc/arduino-mkr-wifi-1010) or [MKR 1000](https://store.arduino.cc/arduino-mkr1000)
![mkr1010](https://user-images.githubusercontent.com/6939054/50213671-914cc180-037e-11e9-9e34-2f64deddd7ee.jpg)

- [UP Squared AI Vision Developer Kit](https://up-shop.org/home/243-up-squared-ai-vision-dev-kit.html)
![up-squared](https://user-images.githubusercontent.com/6939054/50213781-dec92e80-037e-11e9-889a-69e1aa0f01bb.png)
- [Digital Continuous Rotation (360) Servo](https://store.arduino.cc/digital-continuous-rotation-360-servo) or another servo
![servo_continuous](https://user-images.githubusercontent.com/6939054/50213672-914cc180-037e-11e9-85f3-c2c3356408b8.jpg)

You won't need any prior experience with any of the tool used in this tutorial.

### Setup a door-opening thing
The first thing to do is to create a Thing that allows you to open a door when pressing a button in the browser.
- In order to do this, we have to configure an Arduino MKR1000 or MKR WIFI 1010 to be IoT-enabled following the [Arduino IoT Getting Started](https://create.arduino.cc/getting-started/iotsetup?redirectCloud=1). It's pretty simple and straightforward, just follow the steps, give your board a name (I'll call mine **MyMkr1000**, just cause i'm a very creative person), click on **CONFIGURE** and you're done.
- Now click on the button **BACK TO CLOUD**, and you will be redirected to the Thing's creation page (if you don't see the button, click on [this link](https://create.arduino.cc/cloud/things/new))
- You can now create your Thing, just give it a name (mine is **DoorGuy**) and select the MKR board you just configured from the dropdown (which in my case is **MyMkr1000**)
- A modal will show up asking for **WiFi credentials**. Fill it with the WiFi name and password of the network to which the board will connect.
- Then create a property clicking on the **+** sign on the right. In the name field insert `doorOpenStatus` (you could choose whatever name you like, just remember to modify the sketch accordingly later) and select the type **ON/OFF (Boolean)**, leave the rest as it is and click **CREATE**
- Now click on **EDIT CODE** in order to auto-generate a sketch linked to the DoorGuy thing (the name of the sketch is the same as the name of the Thing followed by the current date). You will also be redirected to the [Arduino Create Editor](https://create.arduino.cc/editor/) with the auto-generated sketch open.
- The auto-generated sketch has a file named `ArduinoCloudSettings.h` that deals with connecting the board to the WiFi and to the cloud and a file named `cloudProperties.h` that initializes the properties you created in Arduino IoT Cloud. Notice that every time you create or delete properties and click on **EDIT CODE** the sketch will be auto-modified accordingly. In the **Secret** tab you can modify the WiFi credentials. In the main file (in my case `DoorGuy_dec14a.ino`) you can see a function called `onDoorOpenStatusChange()`: this one will be called every time you modify the properties' value from the DoorGuy dashboard.
- We are going to use the [Servo library](https://www.arduino.cc/en/Reference/Servo), so let's include it
  ```Arduino
  #include <Servo.h>
  ```
  and declare these variables immediately below
  ```Arduino
  int servo_pin = 9;
  int lastTimeServoMoved;
  Servo servo;
  ```
  At the end of the `setup()` function, initialize the servo motor
  ```Arduino
  servo.attach(servo_pin);
  servo.write(90); // a value of 90 means no movement 
  ```
  Then we need to tell the servo to move when the `doorOpenStatus` value is ON, and to stop it when it's OFF
  ```Arduino
  void onDoorOpenStatusChange() {
    // Check the value of the doorOpenStatus property
    if (doorOpenStatus) { // If it is ON, rotate
      servo.write(180); // A value of 180 means rotate clockwise at full speed
      lastTimeServoMoved = millis();
    } else { // If it is OFF, stop it
      servo.write(90);
    }
  }
  ```
  And finally in the middle of the `loop()` function we need set the `doorOpenStatus` to `false` (OFF) after two seconds since it's been turned ON.
  ```Arduino
  // Set the doorOpenStatus property to false (OFF) after 2000ms (2 seconds) and stop the servo
  if (doorOpenStatus && millis() - lastTimeServoMoved > 2000) {
    doorOpenStatus = false;
    onDoorOpenStatusChange();
  }
  ```

  You can find the entire code of the **.ino** file at [DoorGuy_sketch.ino](DoorGuy/DoorGuy_sketch.ino). If you intend to copy/paste it, just pay attention to the following line
  ```Arduino
  #define THING_ID "93c68534-5b53-4876-bff8-64c562e03aa3"
  ```
  That weird string should is my and only mine Thing's ID. Your sketch have a different string that is your and only yours Thing's ID, and you should never modify that. **If you modify it, the connection to the Cloud won't work**. If for some reason you lose it, you can find it in the URL of your Thing in [Arduino IoT Cloud](https://create.arduino.cc/cloud/things) (for example, the URL of my Thing is https://create.arduino.cc/cloud/things/93c68534-5b53-4876-bff8-64c562e03aa3/dashboard)
- **Wire things up!** Connect the servo motor to pin 9  
- Make sure your MKR 1000 or MKR WIFI 1010 is selected in the dropdown above the sketch, open the **Serial Monitor** clicking on **Monitor** on the left of the Arduino Create Editor, or press `CTRL + M`, and then **upload the sketch** to the MKR board clicking on the **UPLOAD** button above (right arrow symbol), or pressing `CTRL + U`. If you wrote everything correctly, the sketch will be compiled and uploaded to the board, and it should start printing some connection info in the serial monitor on the left. If anything goes wrong, the reason could be:
  - It couldn't connect to the WiFi, for example because you inserted wrong credentials or the modem was too far away from the MKR board. Fix the problem and press the reset button or re-upload the sketch
  - It couldn't connect to Cloud, this could be tricky because there are various explanations. For example, sometimes it could fail to connect when the servo is attached to board and the board is powered by your computer. If this is the case, you can try to detach the servo and reset the board, or you can power your board by a wall socket (in this case you won't be able to see the serial monitor output). 
- At a certain point the board should print `Successfully connected to Arduino Cloud :)` on the serial monitor. This means it's all set and you can play with it! Now go to [Arduino IoT Cloud](https://create.arduino.cc/cloud/things) and open your Thing, then click on the eye icon under the name of the Thing to open the **Dashboard**. From here you can monitor the values of its properties and even change them. If the board got connected you should see an ON/OFF button. C'mon, turn ON that switch and watch the Servo do the job. Cool right?
![cloud-open-door](https://user-images.githubusercontent.com/6939054/50213667-90b42b00-037e-11e9-8ffc-50fb41167f35.png)

### Deploy Node-RED container
Now we are going to setup the UP SQUARED and deploy a Node-RED container
- First thing, connect the UP SQUARED to the power, plug in the ethernet and the HDMI cable and attach it to a monitor.  With UP SQUARED switched on, now you should see the Ubuntu login screen on the monitor. Both the username and the password are set to default to **"upsquared"**.
- Once logged into the UP SQUARED, open a browser and go to the [UP SQUARED AI Vision Kit's Getting Started](https://create.arduino.cc/getting-started/intel-up2-cv-kit) page and follow the instruction to link the board to your Arduino account. At a certain point you will be asked to insert the UP SQUARED credentials, which again are username: **"upsquared"** and password: **"upsquared"**. Go on, give it a name and you'll be set up. Leave the UP SQUARED on and grab your computer again.
- Once you connected the UP SQUARED to your account, you will find it in [Arduino Create Device Manager](https://create.arduino.cc/devices), with the name you just gave it, and the status should be **ONLINE** (if it's not, check whether the board is turned ON and connected to the Internet). Click on it to see further details about the board. From this page, you will see some information about the device.
- Open the **Containers** tab on the left. From here you can see the status of the containers on the board and deploy new ones. Click on **DEPLOY CONTAINER**. Fill the modal like this:
  - Container name: **MyNodeRedContainer** (you can choose another name if you want)
  - Image URL/Name: **nodered/node-red-docker**
  - Volume Flags: **-v /tmp:/data**
  - Port Binding: **-p 1880:32700**
  - Restart Policy: **unless-stopped**
  Click on **DEPLOY** and wait until it's done.
- When it finishes deploying, you should see your container running. Now click on **Network** on the left and look for the **IP Address** of the board. Copy it and paste it in a new tab of the browser, followed by `:32700`, which is the container port (for example, if your IP address is 123.39.0.1, you should type `123.39.0.1:32700`), then press enter. You will see and empty Node-RED flow.
- Click on the menu on the upper-right corner, then **Import > Clipboard**, paste the content of [flows.json](node-red/flows.json) and click **Import**.
- Now you have to configure the flow so that will receive an e-mail at your preferred address. Copy the URL of your Thing's dashboard from [Arduino IoT Cloud](https://create.arduino.cc/cloud), then double-click on the **Compose e-mail** node and paste the copied URL inside the variable `dashboardUrl`. Now double-click on the mail node and insert the recipient address in the **To** field, and the sender address and password in the **Userid** and **Password** fields respectively (you may have to [Let less secure apps access your account](https://support.google.com/accounts/answer/6010255?hl=en))
- Finally click on **Deploy** in the upper-right corner. Now the container will wait until a photo in a certain path will be taken and then it will send a mail to the recipient address.
![node-red-mail-flow](https://user-images.githubusercontent.com/6939054/50214185-e63d0780-037f-11e9-9497-8717c6f6ee57.png)


### Upload the face detection sketch
The last step to take in order for this whole system to work is the core of the project: face detection!
- Go to [Arduino Create Editor](https://create.arduino.cc/editor), and create a **NEW SKETCH** pressing on the button on the top left corner.
- Delete the whole content of the new sketch, then copy/paste the content inside [face_detection_photo.ino](face_detection_photo/face_detection_photo.ino)
- Make sure the **camera** is connected to the UP SQUARED
- Make sure your UP SQUARED is selected in the dropdown above the sketch, then open the **Monitor** pressing `CTRL + M`, and finally upload the sketch pressing `CTRL + U`
- You should see some lines printed on the monitor telling that face detection is starting. When the string `[ INFO ]  Start inference` it means that the application is ready. If you move your face in front of the camera, you should see the string `FACE FOUND!`, and after a while you should receive a mail. Click on the link to the dashboard and the servomotor should rotate.
![create-editor-face-detection-sketch](https://user-images.githubusercontent.com/6939054/50214410-901c9400-0380-11e9-8962-e34cdc048663.png)
![node-red-mail-flow](https://user-images.githubusercontent.com/6939054/50214185-e63d0780-037f-11e9-9497-8717c6f6ee57.png)

### Troubleshooting
So now the camera can detect faces and everything should be set up to send a mail when that happens, so that you can press the button on the Thing's dashboard and open the door. If anything isn't working, make sure that:
- the MKR 1000 or MKR 1010 is connected to the cloud, if not should reset it or re-upload the sketch
- the UP SQUARED is connected to the Internet and you see it online on [Arduino Device Manager](https://create.arduino.cc/devices/)
- the Node-RED container is up and running. You can see its status on the **Containers** tab of your UP SQUARED on [Arduino Device Manager](https://create.arduino.cc/devices/)
![device-manager-node-red-container](https://user-images.githubusercontent.com/6939054/50213669-914cc180-037e-11e9-9ff7-659668073245.png)
- the Node-RED flow is deployed. If not sure, go to `UP_SQUARED_IP_ADDRESS:32700`, make sure every value is set up correctly and click on **Deploy**. Here you can also see if there were errors sending the mail, if the page is up when a face is found.
- the face detection sketch is running. You can check its status on the **Sketches** tab of your UP SQUARED on [Arduino Device Manager](https://create.arduino.cc/devices/)
![device-manager-face-detection-sketch](https://user-images.githubusercontent.com/6939054/50213668-90b42b00-037e-11e9-836a-5d0e4f6f164d.png)
