#include <math.h>
#include <PDM.h>
#include <fix_fft.h>


#define PDM_SOUND_GAIN    255   // sound gain of PDM mic
#define PDM_BUFFER_SIZE   256   // buffer size of PDM mic
#define FFT_SIZE          4     // number of buffers per fft
#define FFT_N             6     // fft n
#define FFT_FEATURES      33    // fft size 
#define SAMPLE_THRESHOLD  1000  // RMS threshold to trigger sampling
#define FEATURE_SIZE      31    // sampling size of one voice instance
#define SAMPLE_DELAY      20    // delay time (ms) between sampling
#define TOTAL_SAMPLE      10    // total number of voice instance


char sample[FFT_SIZE * PDM_BUFFER_SIZE / 2];
char im[FFT_SIZE * PDM_BUFFER_SIZE / 2];
short fft_sample[FFT_FEATURES];
short feature_vector[FEATURE_SIZE * FFT_FEATURES];
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
  
  fix_fft(sample, im, FFT_N, 0);

  for (unsigned short i = 0; i < FFT_FEATURES; i++) {
    fft_sample[i] = (short)sqrt(sample[i] * sample[i] + im[i] * im[i]);
  }

  sample_count = (sample_count + 1) % FFT_SIZE;
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
    delay(32);
    for (unsigned short j = 0; j < FFT_FEATURES; j++) {
      feature_vector[i * FFT_FEATURES + j] = fft_sample[j];
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
  
  delay(1000);

  total_counter++;
  if (total_counter >= TOTAL_SAMPLE) {
    PDM.end();
    while (1);
  }
}
