#include <math.h>
#include <PDM.h>


#define PDM_SOUND_GAIN    255    // sound gain of PDM mic
#define PDM_BUFFER_SIZE   256    // buffer size of PDM mic

#define SAMPLE_THRESHOLD  1000    // RMS threshold to trigger sampling
#define FEATURE_SIZE      32     // sampling size of one voice instance
#define SAMPLE_DELAY      20     // delay time (ms) between sampling

#define TOTAL_SAMPLE      5000     // total number of voice instance


double feature_data[FEATURE_SIZE];
short sample[PDM_BUFFER_SIZE / 2];
volatile double rms;
unsigned int total_counter = 0;


void onPDMdata() {
  rms = -1;
  short sample_buffer[PDM_BUFFER_SIZE];
  int bytes_available = PDM.available();
  PDM.read(sample_buffer, bytes_available);
  unsigned int sum = 0;
  for (unsigned short i = 0; i < (bytes_available / 2); i++) {
    sum += pow(sample_buffer[i], 2);
    sample[i] = sample_buffer[i];
  }
  rms = sqrt(double(sum) / (double(bytes_available) / 2.0));
}

void setup() {

  Serial.begin(115200);
  while (!Serial);

  PDM.onReceive(onPDMdata);
  PDM.setBufferSize(PDM_BUFFER_SIZE);
  PDM.setGain(PDM_SOUND_GAIN);

  if (!PDM.begin(1, 16000)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }

  pinMode(LED_BUILTIN, OUTPUT);
  delay(900);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(100);
  digitalWrite(LED_BUILTIN, LOW);  
}

void loop() {
  while (rms < SAMPLE_THRESHOLD);
  digitalWrite(LED_BUILTIN, HIGH);
  Serial.print('<');
  for (unsigned short i = 0; i < FEATURE_SIZE; i++) {
    while (rms < 0);
    feature_data[i] = rms;
    delay(SAMPLE_DELAY);        
    for (unsigned short j = 0; j < PDM_BUFFER_SIZE / 2; j++) {
      Serial.print(sample[j]);
      if (j < (PDM_BUFFER_SIZE / 2 - 1)) {
        Serial.print(',');
      }
    }
    if (i == (FEATURE_SIZE - 1)) {
      Serial.println();
    } else {
      Serial.print(',');
    }    
  }
  Serial.print('>');  
  digitalWrite(LED_BUILTIN, LOW);

  total_counter++;
  if (total_counter >= TOTAL_SAMPLE) {
    PDM.end();    
  }
}
