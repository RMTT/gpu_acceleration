#ifndef GPU_ACCELERATION_SOBEL_CUH
#define GPU_ACCELERATION_SOBEL_CUH

#include <cuda_runtime.h>
#include <opencv2\opencv.hpp>

using namespace cv;

extern __global__ void sobelInCuda(unsigned char* dataIn, unsigned char* dataOut, int imgHeight, int imgWidth);

extern void sobel(Mat srcImg, Mat dstImg, int imgHeight, int imgWidth);

extern void callCuda(const Mat& gaussImg, const Mat& dstImg, int imgHeight, int imgWidth);

#endif //GPU_ACCELERATION_SOBEL_CUH
