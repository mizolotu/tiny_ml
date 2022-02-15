#include <math.h>
#include <PDM.h>


#define PDM_SOUND_GAIN    255   // sound gain of PDM mic
#define PDM_BUFFER_SIZE   256   // buffer size of PDM mic
#define FEATURE_SIZE      125    // sampling size of one voice instance
#define TOTAL_SAMPLE      1000    // total number of voice instance


short sample[PDM_BUFFER_SIZE / 2];
short feature_vector[FEATURE_SIZE * PDM_BUFFER_SIZE / 2];
unsigned int total_counter = 0;


void onPDMdata() {
  short sample_buffer[PDM_BUFFER_SIZE];
  int bytes_available = PDM.available();
  PDM.read(sample_buffer, bytes_available);
  for (unsigned short i = 0; i < (bytes_available / 2); i++) {
    sample[i] = sample_buffer[i];
  }
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
  
  digitalWrite(LED_BUILTIN, HIGH);

  for (unsigned short i = 0; i < FEATURE_SIZE; i++) {
    delay(8);
    for (unsigned short j = 0; j < PDM_BUFFER_SIZE / 2; j++) {    
      feature_vector[i * PDM_BUFFER_SIZE / 2 + j] = sample[j];
    }
  }
  
  digitalWrite(LED_BUILTIN, LOW);
  
  Serial.print('<');
  for (unsigned short i = 0; i < FEATURE_SIZE * PDM_BUFFER_SIZE / 2; i++) {
    Serial.print(feature_vector[i]);     
    if (i == (FEATURE_SIZE * PDM_BUFFER_SIZE / 2 - 1)) {
      Serial.println();
    } else {
      Serial.print(',');
    }    
  }
  Serial.print('>');
  
  total_counter++;
  if (total_counter >= TOTAL_SAMPLE) {
    PDM.end();
    while (1);
  }
}
