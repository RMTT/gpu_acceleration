#include <opencv2/opencv.hpp>
#include <ctime>
#include "sobel.cuh"
#include "matrix_power.cuh"

using namespace cv;


int main() {
    Mat gray_img = imread("C:\\Users\\RMT\\Downloads\\6k.jpg", 0);
    namedWindow("origin", 0);
    imshow("origin", gray_img);
    int height = gray_img.rows;
    int width = gray_img.cols;

    Mat gauss_image;
    GaussianBlur(gray_img, gauss_image, Size(3, 3), 0, 0, BORDER_DEFAULT);

    Mat result_cpu(height, width, CV_8UC1, Scalar(0)), result_gpu(height, width, CV_8UC1, Scalar(0));

    clock_t start, end;
    start = clock();
    sobel(gauss_image, result_cpu, height, width);
    end = clock();

    namedWindow("result_cpu", 0);
    imshow("result_cpu", result_cpu);
    printf("cpu: %ldms\n", end - start);

    callCuda(gauss_image, result_gpu, height, width);

    waitKey(0);
    return 0;
}