#include <math.h>
#include <PDM.h>
#include <fix_fft.h>


#define PDM_SOUND_GAIN    255    // sound gain of PDM mic
#define PDM_BUFFER_SIZE   256    // buffer size of PDM mic
#define FFT_SIZE          4      // number of buffers per fft
#define SAMPLE_THRESHOLD  1000    // RMS threshold to trigger sampling
#define FEATURE_SIZE      31     // sampling size of one voice instance
#define SAMPLE_DELAY      20     // delay time (ms) between sampling
#define TOTAL_SAMPLE      10     // total number of voice instance


char sample[FFT_SIZE * PDM_BUFFER_SIZE / 2];
char im[FFT_SIZE * PDM_BUFFER_SIZE / 2];
short feature_vector[FEATURE_SIZE];
unsigned int total_counter = 0;
short sample_count = 0;


void onPDMdata() {
  short sample_buffer[PDM_BUFFER_SIZE];
  int bytes_available = PDM.available();
  PDM.read(sample_buffer, bytes_available);
  for (unsigned short i = 0; i < (bytes_available / 2); i++) {
    sample[sample_count * PDM_BUFFER_SIZE / 2 + i] = (sample_buffer[i] / 256);
    im[sample_count * PDM_BUFFER_SIZE / 2 + i] = 0;
  }
  sample_count = (sample_count + 1) % FFT_SIZE;  
  //fix_fft(sample, im, 5, 0);
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
  Serial.print('<');
  for (unsigned short i = 0; i < FEATURE_SIZE; i++) {
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
  delay(1000);

  total_counter++;
  if (total_counter >= TOTAL_SAMPLE) {
    PDM.end();
    while (1);
  }
}
