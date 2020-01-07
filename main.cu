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
#include <device_functions.h>

#include<stdio.h>
#include<iostream> 
#include<algorithm>
using namespace std;
using namespace cv;
int numOfColumnsResized;
int numOfRowsResized = 0;
int kernelSize = 4;
int maxDisparity = 13;

__global__ void IDAS_Stereo(int kSize, int MxDisparity,int nR,int nC, uchar* leftIm, uchar* rightIm, uchar*resultIm)
{
	__shared__ int costs[1];
	__shared__ int minCost;
	__shared__ int minCostIndex;
	__shared__ bool minCostGard;

	//costs[blockIdx.z] =0 ;
	minCost = 1000000000;
	minCostIndex = 100;
	__syncthreads();
	int rightPixelIndexU = int(kSize / 2) + blockIdx.x + threadIdx.x- int(kSize / 2);
	int rightPixelIndexV = int(kSize / 2) + blockIdx.y + threadIdx.y - int(kSize / 2);
	int leftPixelIndexU = rightPixelIndexU +threadIdx.z;
	int leftPixelIndexV = rightPixelIndexV;
	int leftPixelIndex = leftPixelIndexV*nC + leftPixelIndexU;
	int rightPixelIndex = rightPixelIndexV*nC + rightPixelIndexU;
	/*int dif =abs( leftIm[leftPixelIndex] - rightIm[rightPixelIndex]);
	costs[blockIdx.z] = costs[blockIdx.z] + dif;

	__syncthreads();
	if (costs[blockIdx.z] < minCost) {
		minCost = costs[blockIdx.z];
		minCostIndex = blockIdx.z;
	}
	__syncthreads();*/

	resultIm[(blockIdx.y + int(kSize / 2))* nC + blockIdx.x + int(kSize / 2)] = leftIm[rightPixelIndex];// uchar(int(minCostIndex * 255 / 10));
	
	//__syncthreads();
}




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
	const int numOfColumns = ((int)leftImage->cols/32)*32;
	const int numOfRows = ((int)leftImage->rows/32)*32;
	cout << "numOfRows is " << numOfRows << " and numOfColumns is " << numOfColumns << endl;
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



	uchar** imArray2DL= new uchar* [numOfRows];
	uchar** imArray2DR = new uchar*[numOfRows];
	for (int i = 0; i < numOfRows; i++) {
		imArray2DL[i] = new uchar[numOfColumns];
		imArray2DR[i] = new uchar[numOfColumns];
	}
	for (int j = 0; j < numOfRows; j++) {
		for (int i = 0; i < numOfColumns; i++) {
			imArray2DL[j][i] = leftImage->at<uchar>(j, i);
			imArray2DR[j][i] = rightImage->at<uchar>(j, i);
		}
	}
	cout << "copy to array is done!!!!!!" << endl;
	uchar* imArrary1DL = new uchar[numOfColumns*numOfRows];
	uchar* imArrary1DR = new uchar[numOfColumns*numOfRows];
	int temp;
	for (int i = 0; i < numOfColumns*numOfRows; i++) {
		imArrary1DL[i] = imArray2DL[int(i / numOfColumns)][i%numOfColumns];
		imArrary1DR[i] = imArray2DL[int(i / numOfColumns)][i%numOfColumns];
	}


	uchar* imArray1DL_d;
	uchar* imArray1DR_d;
	uchar* imArray1DResult_d;
	cudaMalloc((void**)&imArray1DL_d, numOfColumns*numOfRows * sizeof(uchar));
	cudaMalloc((void**)&imArray1DR_d, numOfColumns*numOfRows * sizeof(uchar));
	cudaMalloc((void**)&imArray1DResult_d, numOfColumns*numOfRows * sizeof(uchar));
	cudaMemcpy(imArray1DL_d, imArrary1DL, numOfColumns*numOfRows * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(imArray1DR_d, imArrary1DR, numOfColumns*numOfRows * sizeof(uchar), cudaMemcpyHostToDevice);
	dim3 blocks3D(kernelSize, kernelSize, maxDisparity);
	dim3 grid2D(numOfColumns-2*int(kernelSize /2)- maxDisparity, numOfRows-2*int(kernelSize /2),1);
	//addIntensity <<<(numOfRows*numOfColumns ), 1 >>>(190, imArray1DL_d);
	IDAS_Stereo<<<grid2D, blocks3D >>>(kernelSize, maxDisparity, numOfRows, numOfColumns, imArray1DL_d, imArray1DR_d, imArray1DResult_d);
	cudaMemcpy(imArrary1DL, imArray1DResult_d, numOfColumns*numOfRows * sizeof(uchar), cudaMemcpyDeviceToHost);
	cout << "adding to array is done!!!!!!" << endl;
	for (int i = 0; i < numOfColumns*numOfRows; i++) {
		imArray2DL[int(i / numOfColumns)][i%numOfColumns] = imArrary1DL[i];
	}

	for (int j = 0; j < numOfRows; j++) {
		for (int i = 0; i < numOfColumns; i++) {
			leftImage->at<uchar>(j, i)=(uchar)imArray2DL[j][i] ;
		}
	}
	cudaFree(imArray1DL_d);
	cudaFree(imArray1DR_d);
	cudaFree(imArray1DResult_d);
	imshow(" Left  !!!   .....", *leftImage);
	imwrite("test.png", *leftImage);
	imshow("After effect right image !!!   .....", *rightImage);

	waitKey(1000);
	cout << int(imArray2DL[200][359]) << endl;








	printf("Enter your family name: ");
	char str[80];
	scanf("%79s", str);
}