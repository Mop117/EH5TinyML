/* ignition_detector.cpp
 * Kontinuerlig optagelse + klassifikation (NORMAL/FAULTY)
 * Kræver:
 *  - Microphone_PDM (SIGNED_16 @ 16 kHz)
 *  - feature_extraction.c/.h 
 *  - mymodel.h (C-inference fra keras2c/TFLM)
 */

#include "Particle.h"
#include "Microphone_PDM.h"
#include "feature_extraction.h"
extern "C" {
  #include "k2c_tensor_include.h"   
  #include "mymodel.h"              
}
SYSTEM_MODE(AUTOMATIC);
SYSTEM_THREAD(ENABLED);

SerialLogHandler logHandler(LOG_LEVEL_INFO);

// ------------------ LED-pins (justér efter dit HW) ------------------
constexpr pin_t PIN_LED_GREEN = D5;
constexpr pin_t PIN_LED_RED   = D6;

// ------------------ MFCC/Model parametre (1:1 med din Python) -------
static const int SR = 16000;           // sample rate (skal matche træning)
static const int OUTER_WIN_MS = 400;   // 400 ms vindue til modellen
static const int OUTER_HOP_MS = 200;   // 50% overlap = 200 ms hop

// Indre MFCC-parametre (EI-match fra din kode)
static const int NFFT          = 512;
static const int WIN_MS        = 25;
static const int HOP_MS        = 20;
static const int N_MELS        = 32;
static const int N_MFCC        = 13;
static const float PRE_EMPH    = 0.98f;
static const float FMIN_HZ     = 80.0f;
static const float TOP_DB      = 80.0f;

// Antal klasser i din model
static const int N_CLASSES     = 2;    // NORMAL, FAULTY (ændr hvis nødvendigt)

// ------------------ Arbejdsbuffere til sliding window ---------------
static int OUTER_WIN_SAMPLES   = 0;    // = SR * 0.4
static int OUTER_HOP_SAMPLES   = 0;    // = SR * 0.2

// FIFO til kontinuerlig 400 ms vindue-behandling
// (Vi bruger en fast buffer + memmove for enkelhed)
static const int MAX_FIFO_SAMPLES = 20000; // tilstrækkelig margin
static int16_t fifoBuf[MAX_FIFO_SAMPLES];
static int     fifoFill = 0;

// Feature extractor state
static FE_Config fe_cfg;
static FE_State  fe_state;

// Model output buffer
static float model_out[N_CLASSES];

// Seneste klasse til LED-stabilisering
static int lastClass = -1;

// --------------- LED's -----------------
static inline void leds_set_normal() {
    digitalWrite(PIN_LED_GREEN, HIGH);
    digitalWrite(PIN_LED_RED,   LOW);
}
static inline void leds_set_faulty() {
    digitalWrite(PIN_LED_GREEN, LOW);
    digitalWrite(PIN_LED_RED,   HIGH);
}
static inline void leds_off() {
    digitalWrite(PIN_LED_GREEN, LOW);
    digitalWrite(PIN_LED_RED,   LOW);
}



// Argmax
static int argmax(const float *x, int n) {
    int idx = 0; float best = x[0];
    for (int i=1; i<n; ++i) if (x[i] > best) { best = x[i]; idx = i; }
    return idx;
}

// Læs så mange samples som muligt fra PDM-driveren ind i fifoBuf
static void pullPdmIntoFifo() {
    // DMA-buffer
    const size_t dmaSamples = Microphone_PDM::instance().getNumberOfSamples();
    static int16_t dmaBuf[512];

    // Hent alle tilgængelige DMA-blokke
    while (Microphone_PDM::instance().samplesAvailable()) {
        // Sørg for plads i FIFO
        if (fifoFill + (int)dmaSamples > MAX_FIFO_SAMPLES) {
            int toDrop = (fifoFill + (int)dmaSamples) - MAX_FIFO_SAMPLES;
            if (toDrop > fifoFill) toDrop = fifoFill;
            if (toDrop > 0) {
                memmove(fifoBuf, fifoBuf + toDrop, (fifoFill - toDrop) * sizeof(int16_t));
                fifoFill -= toDrop;
            }
        }

        // Kopier en hel DMA-blok ud af driveren
        if (!Microphone_PDM::instance().copySamples(dmaBuf)) {
            break;
        }

        // Læg i FIFO
        memcpy(&fifoBuf[fifoFill], dmaBuf, dmaSamples * sizeof(int16_t));
        fifoFill += (int)dmaSamples;
    }
}

// Forsøg at behandle vinduer så snart >= OUTER_WIN_SAMPLES
static void processWindowsIfReady() {
    while (fifoFill >= OUTER_WIN_SAMPLES) {
        // 1) Tag de første OUTER_WIN_SAMPLES som ét 400 ms vindue
        const int16_t* pcm_400ms = fifoBuf;
        // Tidsmålinger
        system_tick_t t0, t1, t2, t3;
        t0 = millis();
        // 2) Kør MFCC-sekvens (T x F) + CMVN
        const float* seqTF = nullptr;
        int T = 0, F = 0;
        bool ok = fe_mfcc_sequence_from_window(&fe_cfg, &fe_state, pcm_400ms, &seqTF, &T, &F);
        if (!ok || !seqTF || T <= 0 || F != N_MFCC) {
            Log.warn("FE failed T=%d F=%d", T, F);
            break;
        }
        t1 = millis();
        // Kald modellen
        k2c_tensor in = { (float*)seqTF, 2, 247, {19,13,1,1,1} };     // 19*13 = 247
        float outbuf[2] = {0};
        k2c_tensor out = { outbuf, 1, 2, {2,1,1,1,1} };
        mymodel(&in, &out);
        model_out[0] = out.array[0];
        model_out[1] = out.array[1];
        t2 = millis();
        // 4) Vælg klasse og sæt LED
        int cls = argmax(model_out, N_CLASSES);
        if (cls != lastClass) {
            lastClass = cls;
            if (cls == 1) { // NORMAL
                leds_set_normal();
                Log.info("Class=NORMAL  p=[%.3f %.3f]", model_out[0], model_out[1]);
            } else {        // FAULTY
                leds_set_faulty();
                Log.info("Class=FAULTY  p=[%.3f %.3f]", model_out[0], model_out[1]);
            }
        }
        t3 = millis();
         Log.info("Timings: FE=%d ms, Model=%d ms, Total=%d ms",
                     (int)(t1 - t0), (int)(t2 - t1), (int)(t3 - t0));
        // 5) Skub vinduet frem med 200 ms (50% overlap)
        //    Vi smider OUTER_HOP_SAMPLES fra starten af FIFO og fortsætter
        int bytesToMove = (fifoFill - OUTER_HOP_SAMPLES) * sizeof(int16_t);
        if (bytesToMove > 0) {
            memmove(fifoBuf, fifoBuf + OUTER_HOP_SAMPLES, bytesToMove);
        }
        fifoFill -= OUTER_HOP_SAMPLES;
    }
}

// ------------------ Particle setup/loop -----------------
void setup() {
    Particle.connect();

    pinMode(PIN_LED_GREEN, OUTPUT);
    pinMode(PIN_LED_RED,   OUTPUT);
    leds_off();

    // Udregn samples for ydre vinduer
    OUTER_WIN_SAMPLES = (int)lroundf(SR * (OUTER_WIN_MS / 1000.0f));   // 6400 @16k
    OUTER_HOP_SAMPLES = (int)lroundf(SR * (OUTER_HOP_MS / 1000.0f));   // 3200 @16k

    // Init PDM @ 16 kHz 
    int err = Microphone_PDM::instance()
        .withOutputSize(Microphone_PDM::OutputSize::SIGNED_16)
        .withRange(Microphone_PDM::Range::RANGE_2048)
        .withSampleRate(SR)
        .init();
    if (err) {
        Log.error("PDM init err=%d", err);
    }
    err = Microphone_PDM::instance().start();
    if (err) {
        Log.error("PDM start err=%d", err);
    }

    // Init feature extractor 
    fe_cfg.sample_rate    = SR;
    fe_cfg.outer_win_ms   = OUTER_WIN_MS;
    fe_cfg.outer_hop_ms   = OUTER_HOP_MS;
    fe_cfg.n_fft          = NFFT;
    fe_cfg.win_length_ms  = WIN_MS;
    fe_cfg.hop_length_ms  = HOP_MS;
    fe_cfg.n_mels         = N_MELS;
    fe_cfg.n_mfcc         = N_MFCC;
    fe_cfg.pre_emph       = PRE_EMPH;
    fe_cfg.fmin           = FMIN_HZ;
    fe_cfg.fmax           = SR * 0.5f;
    fe_cfg.top_db         = TOP_DB;
    fe_cfg.hann_window    = nullptr; 

    if (!fe_init(&fe_cfg, &fe_state)) {
        Log.error("feature extractor init failed");
    }

    Log.info("Ready. SR=%d, outer=%dms/%dms => %d samples/%d hop, T=%d, F=%d",
             SR, OUTER_WIN_MS, OUTER_HOP_MS, OUTER_WIN_SAMPLES, OUTER_HOP_SAMPLES,
             fe_state.T, N_MFCC);
}

void loop() {
    // Mikrofon loop
    Microphone_PDM::instance().loop();

    // Hent samples ind i FIFO
    pullPdmIntoFifo();

    // Proces vindue
    processWindowsIfReady();

    delay(1);
}
