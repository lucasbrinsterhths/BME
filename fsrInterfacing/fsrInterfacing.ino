// defining pins and variables
const int fsrPin = A0;
int fsrValue;
float voltage;
String pressureLevel;

void setup() {
  Serial.begin(9600);
}

void loop() {
  fsrValue = analogRead(fsrPin);

  // fsr to voltage
  voltage = fsrValue * (5.0 / 1023.0);
  
  // pressure classification logic
  if (fsrValue < 10) {
    pressureLevel = "no pressure";
  } else if (fsrValue < 200) {
    pressureLevel = "light touch";
  } else if (fsrValue < 500) {
    pressureLevel = "light squeeze";
  } else if (fsrValue < 800) {
    pressureLevel = "medium squeeze";
  } else {
    pressureLevel = "big squeeze";
  }

  // outputting values
  Serial.print("Raw: ");
  Serial.print(fsrValue);
  Serial.print(",     Voltage: ");
  Serial.print(voltage, 2);
  Serial.print(",     Pressure: ");
  Serial.println(pressureLevel);

  delay(200);
}
