#include <math.h>
#include <PDM.h>
#include <fix_fft.h>


#define PDM_SOUND_GAIN    255   // sound gain of PDM mic
#define PDM_BUFFER_SIZE   256   // buffer size of PDM mic
#define FFT_SIZE          17     // number of buffers per fft
#define FFT_N             6     // fft n
#define FFT_FEATURES      33    // fft size 
#define SAMPLE_THRESHOLD  1000  // RMS threshold to trigger sampling
#define FEATURE_SIZE      7    // sampling size of one voice instance
#define TOTAL_SAMPLE      1000    // total number of voice instance


short sample[PDM_BUFFER_SIZE / 2];
short re[PDM_BUFFER_SIZE / 2 * FFT_SIZE];
short im[PDM_BUFFER_SIZE / 2 * FFT_SIZE];
short feature_vector[FEATURE_SIZE * FFT_FEATURES];
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
    for (unsigned short j = 0; j < FFT_SIZE; j++) {
      delay(8);
      for (unsigned short k = 0; k < PDM_BUFFER_SIZE / 2; k++) {
        re[j * PDM_BUFFER_SIZE / 2 + k] = sample[k];
        im[j * PDM_BUFFER_SIZE / 2 + k] = 0;
      }
    }
    fix_fft(re, im, FFT_N, 0);
    for (unsigned short j = 0; j < FFT_FEATURES; j++) {
      feature_vector[i * FFT_FEATURES + j] = (int)(sqrt(re[j] * re[j] + im[j] * im[j]) / 2);
    }
  }

  digitalWrite(LED_BUILTIN, LOW);
  
  Serial.print('<');
  for (unsigned short i = 0; i < FEATURE_SIZE * FFT_FEATURES; i++) {
    Serial.print(feature_vector[i]);     
    if (i == (FEATURE_SIZE * FFT_FEATURES - 1)) {
      Serial.println();
    } else {
      Serial.print(',');
    }    
  }
  Serial.print('>');    
  
  //delay(1000);

  total_counter++;
  if (total_counter >= TOTAL_SAMPLE) {
    PDM.end();
    while (1);
  }
}
