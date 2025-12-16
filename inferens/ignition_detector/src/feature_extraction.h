#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Vælg evt. FFT-backend i build-flags:
//  -DFE_USE_CMSIS   (ARM CMSIS-DSP real FFT)
//  -DFE_USE_KISSFFT (KissFFT real FFT)
// Ellers bruges en langsom, men enkel DFT-fallback.

typedef struct {
    // Samplerate
    int   sample_rate;          // fx 16000

    // YDRE vinduer til modellen: 400 ms med 50% overlap (200 ms hop)
    int   outer_win_ms;         // = 400
    int   outer_hop_ms;         // = 200

    // INDRE MFCC-parametre (skal matche din notebook 1:1)
    int   n_fft;                // = 512
    int   win_length_ms;        // = 25
    int   hop_length_ms;        // = 20
    int   n_mels;               // = 32
    int   n_mfcc;               // = 13
    float pre_emph;             // = 0.98
    float fmin;                 // = 80.0
    float fmax;                 // = sample_rate/2

    // dB-klipning svarende til librosa.power_to_db(..., top_db=80)
    float top_db;               // = 80.0

    // Optional: brugerleveret Hann-vindue for indre STFT (samples = win_length)
    const float* hann_window;   // NULL => genereres
} FE_Config;

typedef struct {
    // Afledte størrelser
    int   outer_win_samples;
    int   outer_hop_samples;
    int   win_length;           // samples (indre)
    int   hop_length;           // samples (indre)
    int   fft_bins;             // n_fft/2 + 1

    // Buffere
    float* x_preemph;           // [outer_win_samples]
    float* stft_frame;          // [win_length]
    float* hann;                // [win_length] (ejes hvis genereret)
    float* fft_in;              // [n_fft]
    float* mag;                 // [fft_bins]
    float* mel_fb;              // [n_mels * fft_bins]
    float* dct;                 // [n_mfcc * n_mels]
    float* mel_tmp;             // [n_mels]
    float* seq_TF;              // [T * n_mfcc] for sidste 400ms-vindue

    // FFT-handle
    void*  fft_handle;

    // meta
    int    T;                   // antal frames i ét 400ms vindue
    bool   owns_hann;
} FE_State;

// Init/Free
bool fe_init(const FE_Config* cfg, FE_State* st);
void fe_free(FE_State* st);

// Processér ÉT 400 ms vindue (int16_t) og få MFCC-sekvensen (T x n_mfcc) med CMVN.
// returnerer true ved succes; out_X peger på intern buffer i st->seq_TF (row-major).
bool fe_mfcc_sequence_from_window(const FE_Config* cfg, FE_State* st,
                                  const int16_t* pcm_window,
                                  const float** out_X, int* out_T, int* out_F);

// Hjælpere til slicing som i Python: slice_to_windows(x, sr, 400ms, 200ms)
int  fe_slice_count(int num_samples, int outer_win_samples, int outer_hop_samples);
void fe_slice_copy(const int16_t* x, int num_samples,
                   int outer_win_samples, int outer_hop_samples,
                   int slice_index, int16_t* dst_window /*len=outer_win_samples*/);

#ifdef __cplusplus
}
#endif

#endif // FEATURE_EXTRACTION_H
