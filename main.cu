////////////////////////////////////////////////////////////////////
/// At first you must set stack commit and stack resservd to 78125000
////////////////////////////////////////////////////////////////////
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<math.h>
#include<vector>
#include<memory>
#include <chrono> 
#include<string> 
#include<math.h>
#include<fstream>


#include<cuda.h>
#include<stdio.h>
#include<iostream> 
#include<algorithm>
using namespace std;
using namespace cv;
__global__ void cuda_hello() {
	printf("Hello World from GPU!\n");
	int xx = threadIdx.x;
	printf("this threadIdx.x is %d \n", xx);
}
__global__ void saxpy(int n, float a, float *x, float *y, int* q)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = a*x[i] + y[i];
}

__global__ void addIntensity(int n, uchar* pixel) {
	//std::cout << pixel[blockIdx.x] << std::endl;
	int temp =(int) pixel[blockIdx.x];
	temp = temp + n;

	pixel[blockIdx.x] = (uchar)temp;
	//std::cout << pixel[blockIdx.x] << std::endl;
	//printf("this threadIdx.x is %d and %d \n", blockIdx.x , (pixel[blockIdx.x]));
}


int numOfColumnsResized;
int numOfRowsResized=0;
int kernelSize = 9;
int maxDisparity = 30;


void ReadBothImages(shared_ptr<Mat> leftImage, shared_ptr<Mat> rightImage) {

	try {
		cout << "this is test" << endl;
		*rightImage = cv::imread("1.png", CV_LOAD_IMAGE_GRAYSCALE);   // Read the right image
																  //rightImage->convertTo(*rightImage, CV_64F);
		*leftImage = cv::imread("2.png", CV_LOAD_IMAGE_GRAYSCALE);   // Read the left image
																 //leftImage->convertTo(*leftImage, CV_64F);
	}
	catch (char* error) {
		cout << "can not load the " << error << " iamge" << endl;
	}

	
	//imshow("test", *rightImage);

	//waitKey(5000);
}
int CalcCost(shared_ptr<Mat> leftImage_, shared_ptr<Mat> rightImage_, int row, int column, int kernelSize, int disparity, int NCols) {
	int cost = 0;
	for (int u = -int(kernelSize / 2); u <= int(kernelSize / 2); u++) {
		for (int v = -int(kernelSize / 2); v <= int(kernelSize / 2); v++) {
			int temp1 = row + u;
			int temp2 = column + v;
			int temp3 = row + u + disparity;
			int temp4 = column + v;
			// for error handeling.
			if (column + u + disparity >= NCols) {
				cout << "*****************************************************" << endl;
			}
			cost = cost + int(pow((leftImage_->at<uchar>(row + v, column + u) - (rightImage_->at<uchar>(row + v, column + u + disparity))), 2));
		}
	}
	return cost;
}
void  SSDstereo(shared_ptr<Mat> leftImage_, shared_ptr<Mat> rightImage_, shared_ptr<Mat> result_temp_, int kernelSize, int maxDisparity, int NRow, int NCols) {
	int tempCost = 0;
	int tempDisparity = 0;

	for (int u = (kernelSize / 2) + 1; u <(NCols - maxDisparity - kernelSize / 2) - 1; u++) {
		for (int v = (kernelSize / 2) + 1; v <NRow - (kernelSize / 2); v++) {
			double cost = 10000000;
			tempCost = 0;
			tempDisparity = 0;
			for (int i = 0; i < maxDisparity; i++) {
				tempCost = CalcCost(leftImage_, rightImage_, v, u, kernelSize, i, NCols);
				if (tempCost < cost) {
					cost = tempCost;
					tempDisparity = i;
				}
			}
			tempDisparity = tempDisparity * 255 / maxDisparity;
			result_temp_->at<uchar>(v, u) = tempDisparity;
			//std::cout << " tempDisparity for ("<< u<<","<<v<<") is "  << tempDisparity << std::endl;
		}
	}
	//std::cout << "debug" << std::endl;
	//cv::imshow("stereoOutput", *result_temp);
	//cv::waitKey(100);
}

int main(void)
{


	shared_ptr<Mat> rightImage = make_shared<Mat>();
	shared_ptr<Mat> leftImage = make_shared<Mat>();

	shared_ptr<Mat> rightImageResized = make_shared<Mat>();
	shared_ptr<Mat> leftImageResized = make_shared<Mat>();

	shared_ptr<Mat>  stereoResut = make_shared<Mat>();
	shared_ptr<Mat>  stereoResutResized = make_shared<Mat>();

	auto start = chrono::high_resolution_clock::now();
	ReadBothImages(leftImage, rightImage);
	const int numOfColumns = leftImage->cols;;
	const int numOfRows = leftImage->rows;

	stereoResut = make_shared<Mat>(numOfRows, numOfColumns, CV_8UC1);
	//SSDstereo(leftImage, rightImage, stereoResut, kernelSize, maxDisparity, numOfRows, numOfColumns);
	//cv::imshow("stereoOutput", *stereoResut);
	//cv::waitKey(1000);
	chrono::high_resolution_clock::time_point stop = chrono::high_resolution_clock::now();
	auto duration =(stop - start);
	auto value = duration.count();
	string duration_s = to_string(value);
	//ofstream repotredResult;

	//shared_ptr<Mat> rightGrayImage = make_shared<Mat>(numOfRows, numOfColumns, CV_8UC1);
	//cv::cvtColor(*rightImage, *rightGrayImage, CV_RGB2BGR);
	//cv::imshow("gray image", *rightImage);
	//cv::waitKey(10000);

	dim3 gridSize(numOfColumns, numOfRows);
	dim3 blockSize(kernelSize, kernelSize, 3);

	uchar** imArray2D= new uchar* [numOfRows];
	for (int i = 0; i < numOfRows; i++) {
		imArray2D[i] = new uchar[numOfColumns];
	}
	for (int j = 0; j < numOfRows; j++) {
		for (int i = 0; i < numOfColumns; i++) {
			imArray2D[j][i] = rightImage->at<uchar>(j, i);
		}
	}
	cout << "copy to array is done!!!!!!" << endl;
	cout << int(imArray2D[200][359] )<< endl;
	cout << rightImage->at<uchar>(200,359) << endl;

	uchar* imArrary1D = new uchar[numOfColumns*numOfRows];
	int temp;
	for (int i = 0; i < numOfColumns*numOfRows; i++) {
		imArrary1D[i] = imArray2D[int(i/numOfColumns)][i%numOfColumns];
		//cout << (int)imArray2D[int(i / numOfColumns)][i%numOfColumns] << endl;
	}


	uchar* imArray1D_d;
	cudaMalloc((void**)&imArray1D_d, numOfColumns*numOfRows * sizeof(uchar));
	cudaMemcpy(imArray1D_d,imArrary1D,numOfColumns*numOfRows*sizeof(uchar), cudaMemcpyHostToDevice);
	addIntensity <<<(numOfRows*numOfColumns ), 1 >>>(190, imArray1D_d);
	cudaMemcpy(imArrary1D, imArray1D_d, numOfColumns*numOfRows * sizeof(uchar), cudaMemcpyDeviceToHost);
	cout << "adding to array is done!!!!!!" << endl;
	for (int i = 0; i < numOfColumns*numOfRows; i++) {
		imArray2D[int(i / numOfColumns)][i%numOfColumns] = imArrary1D[i];
	}

	for (int j = 0; j < numOfRows; j++) {
		for (int i = 0; i < numOfColumns; i++) {
			rightImage->at<uchar>(j, i)=(uchar)imArray2D[j][i] ;
		}
	}
	cudaFree(imArray1D_d);
	imshow(" Left  !!!   .....", *leftImage);
	imshow("After effect right image !!!   .....", *rightImage);
	waitKey(10000);
	cout << int(imArray2D[200][359]) << endl;

	int N = 1 << 2;
	float *x, *y, *d_x, *d_y;
	int *qq;
	int* qq_d;
	x = (float*)malloc(N * sizeof(float));
	y = (float*)malloc(N * sizeof(float));
	qq = (int*)malloc(N * sizeof(int));
	cudaMalloc((void **)&d_x, N * sizeof(float));
	cudaMalloc(&d_y, N * sizeof(float));
	cudaMalloc(&qq_d, N * sizeof(int));

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
		qq[i] = 4;
	}

	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(qq_d, qq, N * sizeof(int), cudaMemcpyHostToDevice);









	//cuda_hello << < 6, 5 >> > ();

	 //Perform SAXPY on 1M elements
	saxpy <<<(N + 255) / 256, 256 >>>(N, 2.0f, d_x, d_y, qq_d);

	cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = max(maxError, abs(y[i] - 4.0f));
	printf("Max error: %f\n", maxError);
	for (int i = 0; i < N; i++) {
		cout << y[i] << endl;

	}
	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
	printf("Enter your family name: ");
	char str[80];
	scanf("%79s", str);
}