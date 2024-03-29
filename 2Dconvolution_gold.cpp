/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>
#include <helper_timer.h>
#include <helper_functions.h>
#include <helper_math.h>
////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold( float*, const float*, const float*, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A convolved with B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param kernel_size         height and width of matrix A
//! @param hB         height of matrices B and C
//! @param wB         width of matrices B and C
////////////////////////////////////////////////////////////////////////////////
void
computeGold(float* C, const float* A, const float* B, unsigned int hB, unsigned int wB)
{
	FILE *grafic;
	float timp;
	grafic = fopen("grafic.txt", "a");
    //pentru măsurarea timpului de execuție în kernel
    StopWatchInterface *kernelTime = NULL;
    sdkCreateTimer(&kernelTime);
    sdkResetTimer(&kernelTime);
    sdkStartTimer(&kernelTime);
	// For each element in the result matrix matrix
	for (unsigned int i = 0; i < hB; ++i){
        for (unsigned int j = 0; j < wB; ++j) {
			double sum = 0;
			// check the start and end values of m and n to prevent overrunning the 
			//  matrix edges
			unsigned int mbegin = (i < 2)? 2 - i : 0;
			unsigned int mend = (i > (hB - 3))?
									hB - i + 2 : 5;
			unsigned int nbegin = (j < 2)? 2 - j : 0;
			unsigned int nend = (j > (wB - 3))?
									(wB-j) + 2 : 5;
			// overlay A over B centered at element (i,j).  For each 
			//  overlapping element, multiply the two and accumulate
			for(unsigned int m = mbegin; m < mend; ++m) {
				for(unsigned int n = nbegin; n < nend; n++) {
					sum += A[m * 5 + n] * 
							B[wB*(i + m - 2) + (j+n - 2)];
				}
			}
			// store the result
			C[i*wB + j] = (float)sum;
        }
	}
	sdkStopTimer(&kernelTime);
    timp = sdkGetTimerValue(&kernelTime);
    fprintf(grafic,"%f ",timp);
    fclose(grafic);
}

