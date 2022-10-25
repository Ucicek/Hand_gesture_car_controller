// put your setup code here, to run once:
int motorApin1 = 2;
int motorApin2 = 3;
int motorBpin1 = 4;
int motorBpin2 = 5;
//1 2 3 1 4
String input;
void setup() {
  Serial.begin(9600);


}

void loop() {
  // put your main code here, to run repeatedly:


  while (Serial.available() == 0) {

  }
  input = Serial.readStringUntil('\r');


  if (input == "straightDirection") {

    digitalWrite(motorApin1, LOW);
    digitalWrite(motorApin2, HIGH);
    // straight here
    digitalWrite(motorBpin1, LOW);
    digitalWrite(motorBpin2, HIGH);

    delay(1000);
  }
  else if (input == "fullRightDirection") {
    digitalWrite(motorApin1, LOW);
    digitalWrite(motorBpin1, LOW);
    digitalWrite(motorApin2, HIGH);
    digitalWrite(motorBpin2, LOW);
    // turn right here1
    delay(1000);
  }
  else if (input == "fullLeftDirection") {
    
    digitalWrite(motorApin1, LOW);
    digitalWrite(motorBpin1, LOW);
    digitalWrite(motorApin2, LOW);
    digitalWrite(motorBpin2, HIGH);
    // turn left here
    delay(1000);
  }
  else if (input == "stopDirection") {
    digitalWrite(motorApin1, LOW);
    digitalWrite(motorApin2, LOW);
    digitalWrite(motorBpin1, LOW);
    digitalWrite(motorBpin2, LOW);
    delay(1000);

    //stop here
  }


}
