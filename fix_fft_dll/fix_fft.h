#pragma once

#ifdef FIXFFT_EXPORTS
#define FIXFFT_API __declspec(dllexport)
#else
#define FIXFFT_API __declspec(dllimport)
#endif

extern "C" FIXFFT_API int fix_fft(short fr[], short fi[], short m, short inverse);

extern "C" FIXFFT_API int fix_fftr(short f[], int m, int inverse);