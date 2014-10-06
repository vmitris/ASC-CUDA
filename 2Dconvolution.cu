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

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>
#include <helper_timer.h>
#include <helper_functions.h>
#include <helper_math.h>

// includes, project
#include "2Dconvolution.h"

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(int width, int height);
Matrix AllocateMatrix(int width, int height);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P);
void ConvolutionOnDeviceShared(const Matrix M, const Matrix N, Matrix P);

////////////////////////////////////////////////////////////////////////////////
// Înmulțirea fără memorie partajată
////////////////////////////////////////////////////////////////////////////////
__global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P)
{

    float Cvalue = 0;
    //se calculeaza pozitia elementulu din matricea P
  	int row = blockIdx.y * blockDim.y + threadIdx.y;
  	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int i,j;
	//se retine in Cvalue suma dupa formula din enunt
	for(i=0;i<5;++i){
		for(j=0;j<5;j++){
			if((row+i-2>=0) && (col+j-2>=0) && (row+i-2<N.height) && (col+j-2<N.width) && (row<N.height) && (col < N.width))
				Cvalue += M.elements[i*5+j]*N.elements[(row+i-2)*N.width+col+j-2];
		}
	}
	//se adauga in P pe pozitia row*N.width+col valoarea sumei
  	P.elements[row * N.width + col] =(float)Cvalue;
}


////////////////////////////////////////////////////////////////////////////////
// Înmulțirea cu memorie partajată
////////////////////////////////////////////////////////////////////////////////
__global__ void ConvolutionKernelShared(Matrix M, Matrix N, Matrix P)
{

    //TODO: calculul rezultatului convoluției

}

////////////////////////////////////////////////////////////////////////////////
// Returnează 1 dacă matricele sunt ~ egale
////////////////////////////////////////////////////////////////////////////////
int CompareMatrices(Matrix A, Matrix B)
{
    int i;
    if(A.width != B.width || A.height != B.height || A.pitch != B.pitch)
        return 0;
    int size = A.width * A.height;
    for(i = 0; i < size; i++)
        if(fabs(A.elements[i] - B.elements[i]) > MAX_ERR)
            return 0;
    return 1;
}
void GenerateRandomMatrix(Matrix m)
{
    int i;
    int size = m.width * m.height;

    srand(time(NULL));

    for(i = 0; i < size; i++)
        m.elements[i] = rand() / (float)RAND_MAX;
}

////////////////////////////////////////////////////////////////////////////////
// main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
    int width = 0, height = 0;
    FILE *f, *out, *grafic;
    if(argc < 2)
    {
        printf("Argumente prea puține, trimiteți id-ul testului care trebuie rulat\n");
        return 0;
    }
    char name[100];
    sprintf(name, "./tests/test_%s.txt", argv[1]);
    f = fopen(name, "r");
    out = fopen("out.txt", "a");
    grafic = fopen("grafic.txt", "a");
    fscanf(f, "%d%d", &width, &height);
    fprintf(grafic,"%d ",width*height);
    Matrix M;//kernel de pe host
    Matrix N;//matrice inițială de pe host
    Matrix P;//rezultat fără memorie partajată calculat pe GPU
    Matrix PS;//rezultatul cu memorie partajată calculat pe GPU
    
    M = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE);
    N = AllocateMatrix(width, height);        
    P = AllocateMatrix(width, height);
    PS = AllocateMatrix(width, height);
	
    GenerateRandomMatrix(M);
    GenerateRandomMatrix(N);
    
    // M * N pe device
    ConvolutionOnDevice(M, N, P);

    // M * N pe device cu memorie partajată
    ConvolutionOnDeviceShared(M, N, PS);
    // calculează rezultatul pe CPU pentru comparație
    Matrix reference = AllocateMatrix(P.width, P.height);
    computeGold(reference.elements, M.elements, N.elements, N.height, N.width);
	fprintf(grafic,"%s","\n");
    // verifică dacă rezultatul obținut pe device este cel așteptat
    int res = CompareMatrices(reference, P);
    printf("Test global %s\n", (1 == res) ? "PASSED" : "FAILED");
    fprintf(out, "Test global %s %s\n", argv[1], (1 == res) ? "PASSED" : "FAILED");

    // verifică dacă rezultatul obținut pe device cu memorie partajată este cel așteptat
  //  int ress = CompareMatrices(reference, PS);
    int ress = CompareMatrices(reference, PS);
    printf("Test shared %s\n", (1 == ress) ? "PASSED" : "FAILED");
    fprintf(out, "Test shared %s %s\n", argv[1], (1 == ress) ? "PASSED" : "FAILED");
   
    // Free matrices
    FreeMatrix(&M);
    FreeMatrix(&N);
    FreeMatrix(&P);
    FreeMatrix(&PS);

    fclose(f);
    fclose(out);
    fclose(grafic);
    return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P)
{

	/*am adaugat variabila FILE * grafic
	ce reprezinta fisierul de output pentru timp*/


    Matrix Md, Nd, Pd; //matricele corespunzătoare de pe device
	FILE *grafic;
	float timp;
	grafic = fopen("grafic.txt", "a");
    //pentru măsurarea timpului de execuție în kernel
    StopWatchInterface *kernelTime = NULL;
    sdkCreateTimer(&kernelTime);
    sdkResetTimer(&kernelTime);
    //alocare matricele de pe device
	Md = AllocateDeviceMatrix(M.width,M.height);
	Nd = AllocateDeviceMatrix(N.width,N.height);
	Pd = AllocateDeviceMatrix(N.width,N.height);

    //copiere date de pe host (M, N) pe device (MD, Nd)
    int size = M.width * M.height * sizeof(float);
	cudaMemcpy( Md.elements, M.elements, size, cudaMemcpyHostToDevice);
	size = N.width * N.height * sizeof(float);
	cudaMemcpy( Nd.elements, N.elements, size, cudaMemcpyHostToDevice);
   
    //setare configurație de rulare a kernelului
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
 	dim3 dimGrid((N.width + BLOCK_SIZE - 1) /dimBlock.x,(N.height + BLOCK_SIZE - 1) /dimBlock.y);
    sdkStartTimer(&kernelTime);
    
    //lansare în execuție a kernelului
    ConvolutionKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
    cudaThreadSynchronize(); 
    sdkStopTimer(&kernelTime);
    timp = sdkGetTimerValue(&kernelTime);
    printf ("Timp execuție kernel: %f ms\n", timp);
    fprintf(grafic,"%f ",timp);
    
    //copiere rezultat pe host
    size = N.width * N.height * sizeof(float);
    cudaMemcpy( P.elements, Pd.elements, size, cudaMemcpyDeviceToHost);
    
    //eliberarea memoriei matricelor de pe device
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);
    fclose(grafic);
}

//nu am realizat varianta cu memory shared
void ConvolutionOnDeviceShared(const Matrix M, const Matrix N, Matrix P)
{
    //Matrix Md, Nd, Pd; //matricele corespunzătoare de pe device
	FILE *grafic;
	float timp;
	grafic = fopen("grafic.txt", "a");
    //pentru măsurarea timpului de execuție în kernel
    StopWatchInterface *kernelTime = NULL;
    sdkCreateTimer(&kernelTime);
    sdkResetTimer(&kernelTime);
    //TODO: alocați matricele de pe device

    //TODO: copiați datele de pe host (M, N) pe device (MD, Nd)

    //TODO: setați configurația de rulare a kernelului

    sdkStartTimer(&kernelTime);
    //TODO: lansați în execuție kernelul    
    cudaThreadSynchronize();
    sdkStopTimer(&kernelTime);
    timp = sdkGetTimerValue(&kernelTime);
    printf ("Timp execuție kernel cu memorie partajată: %f ms\n", timp);
    fprintf(grafic,"%f ",timp);
    //TODO: copiaţi rezultatul pe host
    //TODO: eliberați memoria matricelor de pe device
    fclose(grafic);
}


// Alocă o matrice de dimensiune height*width pe device
Matrix AllocateDeviceMatrix(int width, int height)
{
	//TODO: alocați matricea și setați width, pitch și height
    Matrix m;
   	m.width = width;
   	m.height = height;
   	m.pitch = width;
   	size_t size = width * height * sizeof(float);
   	cudaMalloc((void**)&m.elements,size);
    return m;
}

// Alocă matrice pe host de dimensiune height*width
Matrix AllocateMatrix(int width, int height)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;    
    M.elements = (float*) malloc(size*sizeof(float));
    return M;
}    

// Eliberează o matrice de pe device
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Eliberează o matrice de pe host
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}
