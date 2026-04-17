#include <Arduino.h>

int pins[6] = {A0, A1, A2, A3, A4, A5};
int piezoPin = 9;

void setup() {
  // Binding the pins to the input mode
  for (int pin : pins) {
    pinMode(pin, INPUT);
  }
  
  Serial.begin(9600);
  delay(1000);        // Wait for serial to initialize
  Serial.println("A0,A1,A2,A3,A4,A5,hole");

  // Loop through the 30 holes and play a tone for each one, while printing the sensor values
  for (int i = 0; i < 30; i = i + 1) {
    tone(piezoPin, 1000, 500);  // Play a tone on the piezo buzzer
    delay(1000);                // Wait to move flashlight
    for (int j = 0; j < 50; j = j + 1) {
      printValues(i);
    }
  }

  Serial.println("Done!");
}

void printValues(int param) {
  for (int pin : pins) {
    int value = analogRead(pin);
    Serial.print(map(value, 0, 1023, 0, 100));  // Map the value to a percentage
    Serial.print(",");                          // Making a .csv
  }
  Serial.print(param);
  Serial.println(); // New line after printing all sensor values
  delay(100);
}

void loop() {
  // Code does not loop, all operations are done in setup()
}
