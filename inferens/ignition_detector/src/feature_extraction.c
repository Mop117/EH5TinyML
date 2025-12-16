#include "feature_extraction.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------- Clamp float ----------
static inline float clampf(float x, float lo, float hi){ return x<lo?lo:(x>hi?hi:x); }

// ---------- Hann window ----------
static void make_hann(float* w, int N) {
    for (int n=0; n<N; ++n) {
        w[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * n / (N - 1));
    }
}

// ---------- DFT (real -> |X[k]|, k=0..N/2) ----------
static void dft_real_mag(const float* x, int Nfft, float* mag_out) {
    int K = Nfft/2 + 1;
    for (int k=0; k<K; ++k) {
        float re=0.f, im=0.f;
        for (int n=0; n<Nfft; ++n) {
            float ang = -2.0f*(float)M_PI*k*n/Nfft;
            re += x[n]*cosf(ang);
            im += x[n]*sinf(ang);
        }
        mag_out[k] = sqrtf(re*re + im*im);
    }
}

// ---------- HTK mel <-> Hz ----------
static float hz_to_mel_htk(float f)   { return 2595.0f * log10f(1.0f + f/700.0f); }
static float mel_to_hz_htk(float mel) { return 700.0f * (powf(10.0f, mel/2595.0f) - 1.0f); }

// ---------- Build mel filterbank: HTK, lineært i mel, trekantet i Hz ----------
static bool build_mel_fb(const FE_Config* cfg, FE_State* st) {
    int M = cfg->n_mels;
    int K = st->fft_bins;
    float nyq = cfg->sample_rate * 0.5f;

    float mel_lo = hz_to_mel_htk(cfg->fmin);
    float mel_hi = hz_to_mel_htk(cfg->fmax);
    float mel_step = (mel_hi - mel_lo) / (M + 1);

    float* centers_mel = (float*)malloc((M+2)*sizeof(float));
    if (!centers_mel) return false;
    for (int m=0; m<M+2; ++m) centers_mel[m] = mel_lo + mel_step*m;

    float* centers_hz = (float*)malloc((M+2)*sizeof(float));
    if (!centers_hz) { free(centers_mel); return false; }
    for (int m=0; m<M+2; ++m) centers_hz[m] = mel_to_hz_htk(centers_mel[m]);

    st->mel_fb = (float*)malloc(M * K * sizeof(float));
    if (!st->mel_fb) { free(centers_mel); free(centers_hz); return false; }
    memset(st->mel_fb, 0, M*K*sizeof(float));

    for (int m=0; m<M; ++m) {
        float f_left  = centers_hz[m+0];
        float f_center= centers_hz[m+1];
        float f_right = centers_hz[m+2];

        for (int k=0; k<K; ++k) {
            float f_k = nyq * ((float)k / (K - 1));
            float w=0.f;
            if (f_k >= f_left && f_k <= f_center) {
                w = (f_k - f_left) / (f_center - f_left + 1e-12f);
            } else if (f_k >= f_center && f_k <= f_right) {
                w = (f_right - f_k) / (f_right - f_center + 1e-12f);
            }
            st->mel_fb[m*K + k] = clampf(w, 0.0f, 1.0f);
        }
    }

    free(centers_mel);
    free(centers_hz);
    return true;
}

// ---------- DCT-II (ortho) ----------
static bool build_dct(const FE_Config* cfg, FE_State* st) {
    int K = cfg->n_mfcc;
    int M = cfg->n_mels;
    st->dct = (float*)malloc(K * M * sizeof(float));
    if (!st->dct) return false;

    float scale0 = sqrtf(1.0f / M);
    float scale  = sqrtf(2.0f / M);

    for (int k=0; k<K; ++k) {
        for (int n=0; n<M; ++n) {
            float val = cosf((float)M_PI * (n + 0.5f) * k / M);
            st->dct[k*M + n] = (k==0 ? scale0 : scale) * val;
        }
    }
    return true;
}

// ---------- Init / Free ----------
bool fe_init(const FE_Config* cfg, FE_State* st) {
    if (!cfg || !st) return false;
    memset(st, 0, sizeof(*st));

    st->outer_win_samples = (int)lroundf(cfg->sample_rate * (cfg->outer_win_ms / 1000.0f));
    st->outer_hop_samples = (int)lroundf(cfg->sample_rate * (cfg->outer_hop_ms / 1000.0f));
    st->win_length        = (int)lroundf(cfg->sample_rate * (cfg->win_length_ms / 1000.0f));
    st->hop_length        = (int)lroundf(cfg->sample_rate * (cfg->hop_length_ms / 1000.0f));
    st->fft_bins          = cfg->n_fft/2 + 1;

    if (cfg->fmax <= 0) return false;

    // antal STFT-frames (T) for ét 400ms vindue (center=False):
    // T = floor((N - win)/hop) + 1  for N >= win, ellers 0
    int N = st->outer_win_samples;
    st->T = (N >= st->win_length) ? ((N - st->win_length) / st->hop_length + 1) : 0;
    if (st->T <= 0) return false;

    // Buffere
    st->x_preemph = (float*)malloc(st->outer_win_samples * sizeof(float));
    st->stft_frame= (float*)malloc(st->win_length * sizeof(float));
    st->hann      = (cfg->hann_window) ? (float*)cfg->hann_window
                                       : (float*)malloc(st->win_length * sizeof(float));
    st->fft_in    = (float*)malloc(cfg->n_fft * sizeof(float));
    st->mag       = (float*)malloc(st->fft_bins * sizeof(float));
    st->mel_tmp   = (float*)malloc(cfg->n_mels * sizeof(float));
    st->seq_TF    = (float*)malloc(st->T * cfg->n_mfcc * sizeof(float));
    if (!st->x_preemph || !st->stft_frame || !st->hann || !st->fft_in ||
        !st->mag || !st->mel_tmp || !st->seq_TF) return false;

    st->owns_hann = (cfg->hann_window == NULL);
    if (st->owns_hann) make_hann(st->hann, st->win_length);

    if (!build_mel_fb(cfg, st)) return false;
    if (!build_dct(cfg, st))    return false;


    return true;
}

void fe_free(FE_State* st) {
    if (!st) return;
#if defined(FE_USE_CMSIS) || defined(FE_USE_KISSFFT)
    if (st->fft_handle) free(st->fft_handle);
#endif
    if (st->mel_fb)    free(st->mel_fb);
    if (st->dct)       free(st->dct);
    if (st->x_preemph) free(st->x_preemph);
    if (st->stft_frame)free(st->stft_frame);
    if (st->fft_in)    free(st->fft_in);
    if (st->mag)       free(st->mag);
    if (st->mel_tmp)   free(st->mel_tmp);
    if (st->seq_TF)    free(st->seq_TF);
    if (st->owns_hann && st->hann) free(st->hann);
    memset(st, 0, sizeof(*st));
}

// ---------- FFT magnitude fra tidsramme ----------
static void compute_mag(const FE_Config* cfg, FE_State* st, const float* xwin) {
#if defined(FE_USE_CMSIS)
    // Brug tmp = fft_in som både in/out buffer
    float* tmp = st->fft_in;
    memcpy(tmp, xwin, cfg->n_fft * sizeof(float));
    arm_rfft_fast_instance_f32* h = (arm_rfft_fast_instance_f32*)st->fft_handle;
    arm_rfft_fast_f32(h, tmp, tmp, 0);
    int K = st->fft_bins;
    st->mag[0] = fabsf(tmp[0]); // DC
    for (int k=1; k<K-1; ++k) {
        float re = tmp[2*k+0];
        float im = tmp[2*k+1];
        st->mag[k] = sqrtf(re*re + im*im);
    }
    st->mag[K-1] = fabsf(tmp[1]); // Nyquist
#elif defined(FE_USE_KISSFFT)
    kiss_fftr_cfg h = (kiss_fftr_cfg)st->fft_handle;
    kiss_fft_scalar* rin = (kiss_fft_scalar*)st->fft_in;
    for (int i=0; i<cfg->n_fft; ++i) rin[i] = (kiss_fft_scalar)xwin[i];
    kiss_fft_cpx* cbuf = (kiss_fft_cpx*)malloc(st->fft_bins * sizeof(kiss_fft_cpx));
    kiss_fftr(h, rin, cbuf);
    for (int k=0; k<st->fft_bins; ++k) {
        float re = cbuf[k].r, im = cbuf[k].i;
        st->mag[k] = sqrtf(re*re + im*im);
    }
    free(cbuf);
#else
    dft_real_mag(xwin, cfg->n_fft, st->mag);
#endif
}

// ---------- mel power -> dB (ref=1.0), top_db klip ----------
static void mel_power_to_db_clip(const FE_Config* cfg, const FE_State* st,
                                 float* mel /*len=n_mels*/, float* out_db) {
    int M = cfg->n_mels;
    float max_db = -1e30f;
    // 10*log10(power)
    for (int m=0; m<M; ++m) {
        float v = mel[m] <= 0.f ? 1e-10f : mel[m];
        out_db[m] = 10.0f * log10f(v); // ref=1.0
        if (out_db[m] > max_db) max_db = out_db[m];
    }
    // top_db=80 -> clip til [max_db-80, max_db]
    float floorv = max_db - cfg->top_db;
    for (int m=0; m<M; ++m) {
        if (out_db[m] < floorv) out_db[m] = floorv;
    }
}

// ---------- Én MFCC frame fra en tidsramme (allerede preemph'et) ----------
static void mfcc_from_timeframe(const FE_Config* cfg, FE_State* st,
                                const float* x_frame /*len=win_length*/,
                                float* mfcc_out /*len=n_mfcc*/) {
    // 1) Hann + zero-pad til n_fft
    for (int i=0; i<cfg->n_fft; ++i) {
        float v = (i < st->win_length) ? x_frame[i] * st->hann[i] : 0.f;
        st->fft_in[i] = v;
    }
    // 2) FFT magnitude
    compute_mag(cfg, st, st->fft_in);
    // 3) mel power
    int K = st->fft_bins, M = cfg->n_mels;
    for (int m=0; m<M; ++m) {
        const float* w = &st->mel_fb[m*K];
        float acc = 0.f;
        for (int k=0; k<K; ++k) {
            float p = st->mag[k]*st->mag[k]; // power=2.0
            acc += w[k] * p;
        }
        st->mel_tmp[m] = acc;
    }
    // 4) power_to_db(ref=1.0, top_db=80)
    float* mel_db = st->fft_in; // reuse
    mel_power_to_db_clip(cfg, st, st->mel_tmp, mel_db);
    // 5) DCT-II (ortho) -> MFCC
    // y[K] = DCT[KxM] * mel_db[M]
    for (int k=0; k<cfg->n_mfcc; ++k) {
        const float* row = &st->dct[k*M];
        float acc=0.f;
        for (int n=0; n<M; ++n) acc += row[n]*mel_db[n];
        mfcc_out[k] = acc;
    }
}

// ---------- Python: slice_to_windows (kopi af logik) ----------
int fe_slice_count(int num_samples, int outer_win_samples, int outer_hop_samples) {
    if (num_samples < outer_win_samples) return 1; // padder som i Python
    int count = 0;
    for (int s=0; s + outer_win_samples <= num_samples; s += outer_hop_samples) count++;
    return count;
}

void fe_slice_copy(const int16_t* x, int num_samples,
                   int outer_win_samples, int outer_hop_samples,
                   int slice_index, int16_t* dst_window) {
    int start = slice_index * outer_hop_samples;
    // Pad i slutningen hvis nødvendigt (Python padder kun hvis signal < win, men her følger vi slicing)
    for (int i=0; i<outer_win_samples; ++i) {
        int idx = start + i;
        dst_window[i] = (idx < num_samples) ? x[idx] : 0;
    }
}

// ---------- Hovedfunktion: ét 400ms-vindue -> (T x F) med CMVN ----------
bool fe_mfcc_sequence_from_window(const FE_Config* cfg, FE_State* st,
                                  const int16_t* pcm_window,
                                  const float** out_X, int* out_T, int* out_F) {
    if (!cfg || !st || !pcm_window || !out_X) return false;

    // 0) pre-emphasis på HELE 400ms-vinduet (y[0]=x[0])
    st->x_preemph[0] = (float)pcm_window[0] / 32768.0f;
    for (int i=1; i<st->outer_win_samples; ++i) {
        float x = (float)pcm_window[i]   / 32768.0f;
        float xm= (float)pcm_window[i-1] / 32768.0f;
        st->x_preemph[i] = x - cfg->pre_emph * xm;
    }

    // 1) STFT-framing (center=False), hop=20 ms, win=25 ms
    int T = st->T;
    int F = cfg->n_mfcc;

    for (int t=0; t<T; ++t) {
        int start = t * st->hop_length;
        // kopi indre ramme
        for (int i=0; i<st->win_length; ++i) st->stft_frame[i] = st->x_preemph[start + i];
        // MFCC for denne ramme
        mfcc_from_timeframe(cfg, st, st->stft_frame, &st->seq_TF[t*F]);
    }

    // 2) CMVN over tidsaksen pr. feature (kolonnevis)
    for (int f=0; f<F; ++f) {
        // mean
        float mu=0.f;
        for (int t=0; t<T; ++t) mu += st->seq_TF[t*F + f];
        mu /= (float)T;
        // std
        float var=0.f;
        for (int t=0; t<T; ++t) {
            float d = st->seq_TF[t*F + f] - mu;
            var += d*d;
        }
        var /= (float)T;
        float sd = sqrtf(var) + 1e-6f;
        // normér
        for (int t=0; t<T; ++t) {
            st->seq_TF[t*F + f] = (st->seq_TF[t*F + f] - mu) / sd;
        }
    }

    *out_X = st->seq_TF;
    if (out_T) *out_T = T;
    if (out_F) *out_F = F;
    return true;
}
