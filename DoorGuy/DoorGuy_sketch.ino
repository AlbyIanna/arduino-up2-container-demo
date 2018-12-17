/* 
  Sketch generated by the Arduino IoT Cloud Thing "DoorGuy"
  https://create.arduino.cc/cloud/things/93c68534-5b53-4876-bff8-64c562e03aa3 

  Arduino IoT Cloud Properties description

  The following variables are automatically generated and updated when changes are made to the Thing properties

  bool doorOpenStatus;

  Properties which are marked as READ/WRITE in the Cloud Thing will also have functions
  which are called when their values are changed from the Dashboard.
  These functions are generated with the Thing and added at the end of this sketch.
*/

// Your THING_ID
#define THING_ID "93c68534-5b53-4876-bff8-64c562e03aa3"

/*
  The following data fields are filled by the Secret tab.
  Do not modify them here
*/
char ssid[] = SECRET_SSID;        // your network SSID (name)
char pass[] = SECRET_PASS;    // your network password (use for WPA, or use as key for WEP)

// the following include needs data from the Secret tab to define the arrays above
#include "ArduinoCloudSettings.h"
#include <Servo.h>

int servo_pin = 9;
int lastTimeServoMoved;
Servo servo;

void setup() {
  // Initialize serial and wait for port to open:
  Serial.begin(9600);
  // this delay gives the chance to wait for a Serial Monitor without blocking if none is found
  delay(1500); 
  // Defined in cloudProperties.h
  initProperties();
  // Defined in ArduinoCloudSettings.h
  initConnection();
  // set last network check to now, since it is connected
  lastNetworkCheck = millis();
  
  servo.attach(servo_pin);
  servo.write(90); // a value of 90 means no movement 
}

void loop() {
  ArduinoCloud.update();

  // Set the doorOpenStatus property to false (OFF) after 2000ms (2 seconds) and stop the servo
  if (doorOpenStatus && millis() - lastTimeServoMoved > 2000) {
    doorOpenStatus = false;
    onDoorOpenStatusChange();
  }

  // Check if device is connected to the network, if it is do nothing, otherwise reconnect it. Leave this at the end of loop().
  checkNetworkConnection();
}


void onDoorOpenStatusChange() {
  // Check the value of the doorOpenStatus property
  if (doorOpenStatus) { // If it is ON, rotate
    servo.write(180); // A value of 180 means rotate clockwise at full speed
    lastTimeServoMoved = millis();
  } else { // If it is OFF, stop it
    servo.write(90);
  }
}
