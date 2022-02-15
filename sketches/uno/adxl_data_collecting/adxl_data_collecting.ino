#include <SPI.h>
#include <ADXL362.h>

#define DELAY   10
 

ADXL362 xl;

int16_t XValue, YValue, ZValue, TValue;

void setup(){
  
  Serial.begin(115200);
  xl.begin(10);                   // Setup SPI protocol, issue device soft reset
  xl.beginMeasure();              // Switch ADXL362 to measure mode    
}

void loop(){
    
  xl.readXYZTData(XValue, YValue, ZValue, TValue);  
  Serial.print('<');
  Serial.print(XValue);  
  Serial.print(',');
  Serial.print(YValue);  
  Serial.print(',');
  Serial.print(ZValue);  
  Serial.print('>');
  Serial.println();
  delay(DELAY);
}
