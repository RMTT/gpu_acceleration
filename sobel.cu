#include "sobel.cuh"
#include <device_launch_parameters.h>
#include <ctime>
using namespace std;
using namespace cv;


__global__ void sobelInCuda(unsigned char* dataIn, unsigned char* dataOut, int imgHeight, int imgWidth) {
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	int index = yIndex * imgWidth + xIndex;
	int Gx = 0;
	int Gy = 0;

	if (xIndex > 0 && xIndex < imgWidth - 1 && yIndex > 0 && yIndex < imgHeight - 1) {
		Gx = dataIn[(yIndex - 1) * imgWidth + xIndex + 1] + 2 * dataIn[yIndex * imgWidth + xIndex + 1] +
			dataIn[(yIndex + 1) * imgWidth + xIndex + 1]
			- (dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[yIndex * imgWidth + xIndex - 1] +
				dataIn[(yIndex + 1) * imgWidth + xIndex - 1]);
		Gy = dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex - 1) * imgWidth + xIndex] +
			dataIn[(yIndex - 1) * imgWidth + xIndex + 1]
			- (dataIn[(yIndex + 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex + 1) * imgWidth + xIndex] +
				dataIn[(yIndex + 1) * imgWidth + xIndex + 1]);
		dataOut[index] = (abs(Gx) + abs(Gy)) / 2;
	}
}

void sobel(Mat srcImg, Mat dstImg, int imgHeight, int imgWidth) {
	int Gx = 0;
	int Gy = 0;
	for (int i = 1; i < imgHeight - 1; i++) {
		uchar* dataUp = srcImg.ptr<uchar>(i - 1);
		uchar* data = srcImg.ptr<uchar>(i);
		uchar* dataDown = srcImg.ptr<uchar>(i + 1);
		uchar* out = dstImg.ptr<uchar>(i);
		for (int j = 1; j < imgWidth - 1; j++) {
			Gx = (dataUp[j + 1] + 2 * data[j + 1] + dataDown[j + 1]) -
				(dataUp[j - 1] + 2 * data[j - 1] + dataDown[j - 1]);
			Gy = (dataUp[j - 1] + 2 * dataUp[j] + dataUp[j + 1]) -
				(dataDown[j - 1] + 2 * dataDown[j] + dataDown[j + 1]);
			out[j] = (abs(Gx) + abs(Gy)) / 2;
		}
	}
}

void callCuda(const Mat& gaussImg, const Mat& dstImg, int imgHeight, int imgWidth) {
	unsigned char* d_in;
	unsigned char* d_out;

	cudaMalloc((void**)&d_in, imgHeight * imgWidth * sizeof(unsigned char));
	cudaMalloc((void**)&d_out, imgHeight * imgWidth * sizeof(unsigned char));


	cudaMemcpy(d_in, gaussImg.data, imgHeight * imgWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(32, 32);
	dim3 blocksPerGrid((imgWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(imgHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

	clock_t start = clock();
	sobelInCuda << <blocksPerGrid, threadsPerBlock >> > (d_in, d_out, imgHeight, imgWidth);
	cudaDeviceSynchronize();
	clock_t end = clock();
	
	printf("gpu: %ldms\n",end - start);

	cudaMemcpy(dstImg.data, d_out, imgHeight * imgWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);


	cudaFree(d_in);
	cudaFree(d_out);
}