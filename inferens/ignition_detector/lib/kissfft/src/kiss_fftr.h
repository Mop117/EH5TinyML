/*
 *  Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#ifndef KISS_FTR_H
#define KISS_FTR_H

#include "kiss_fft.h"
#ifdef __cplusplus
extern "C" {
#endif
#ifndef KISS_FFT_MALLOC
#  define KISS_FFT_MALLOC  malloc
#endif
#ifndef KISS_FFT_FREE
#  define KISS_FFT_FREE    free
#endif

/*

 Real optimized version can save about 45% cpu time vs. complex fft of a real seq.



 */

typedef struct kiss_fftr_state *kiss_fftr_cfg;


#ifdef __cplusplus
extern "C" {
#endif

typedef struct kiss_fftr_state* kiss_fftr_cfg;

kiss_fftr_cfg kiss_fftr_alloc(int nfft, int inverse_fft,
                              void *mem, size_t *lenmem);

void kiss_fftr(kiss_fftr_cfg cfg, const kiss_fft_scalar *timedata,
               kiss_fft_cpx *freqdata);

void kiss_fftri(kiss_fftr_cfg cfg, const kiss_fft_cpx *freqdata,
                kiss_fft_scalar *timedata);

#ifdef __cplusplus
}
#endif
/*
 nfft must be even

 If you don't care to allocate space, use mem = lenmem = NULL
*/



/*
 input timedata has nfft scalar points
 output freqdata has nfft/2+1 complex points
*/


/*
 input freqdata has  nfft/2+1 complex points
 output timedata has nfft scalar points
*/

#define kiss_fftr_free KISS_FFT_FREE

#ifdef __cplusplus
}
#endif
#endif
