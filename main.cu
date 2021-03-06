////////////////////////////////////////////////////////////////////
/// At first you must set stack commit and stack resservd to 78125000
////////////////////////////////////////////////////////////////////
#pragma once
#include <cooperative_groups.h>
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


#include <time.h>

#include<stdio.h>
#include<iostream> 
#include<algorithm>
using namespace std;
using namespace cv;
int numOfColumnsResized;
int numOfRowsResized = 0;
int kernelSize = 12;
int maxDisparity = 30;
int selectedDisparity =5;


__global__ void IDAS_Stereo_selective(int kSize, int MxDisparity, int nC , int nSelected, uchar* leftIm, uchar* rightIm, bool* resultIm)
{
	//thread_group my_block = this_thread_block();
	//thread_block block = this_thread_block();
	__shared__   int costs[6];
	__shared__   int dif[6 * 12 * 12];

	

	int rightPixelIndexU;
	int rightPixelIndexV;

	int leftPixelIndexU;
	int leftPixelIndexV;

	int leftPixelIndex;
	int rightPixelIndex;

	
	if (threadIdx.z<3) {
		//printf("the selected disparity i %d \n  ", nSelected);
		rightPixelIndexU = blockIdx.x + threadIdx.x + MxDisparity + 1;
		rightPixelIndexV = blockIdx.y + threadIdx.y;
		leftPixelIndexU = rightPixelIndexU + (threadIdx.z-1)+nSelected;
		leftPixelIndexV = rightPixelIndexV;
		leftPixelIndex = leftPixelIndexV*nC + leftPixelIndexU;
		rightPixelIndex = rightPixelIndexV*nC + rightPixelIndexU;
		dif[threadIdx.x+threadIdx.y*12+threadIdx.z*(144)] = abs(leftIm[leftPixelIndex] - rightIm[rightPixelIndex]);

		//costs[threadIdx.z] = costs[threadIdx.z] + dif[threadIdx.x + threadIdx.y * 12 + threadIdx.z * (144)];

	}
	else {
		rightPixelIndexU = blockIdx.x + threadIdx.x + MxDisparity + 1 + (threadIdx.z - 4);
		rightPixelIndexV = blockIdx.y + threadIdx.y;
		leftPixelIndexU = blockIdx.x + threadIdx.x + MxDisparity + 1+ nSelected;
		leftPixelIndexV = rightPixelIndexV;
		leftPixelIndex = leftPixelIndexV*nC + leftPixelIndexU;
		rightPixelIndex = rightPixelIndexV*nC + rightPixelIndexU;
		dif[threadIdx.x + threadIdx.y * 12 + threadIdx.z * (144)] = abs(leftIm[leftPixelIndex] - rightIm[rightPixelIndex]);
		//costs[threadIdx.z] = costs[threadIdx.z] + dif[threadIdx.x + threadIdx.y * 12 + threadIdx.z * (144)];
	}
	__syncthreads();


	if (threadIdx.x == 0 & threadIdx.y == 0) {
		costs[threadIdx.z] = 0;
		for (int i = 0; i < 12;i++) {
			for(int j=0;j<12;j++)
			costs[threadIdx.z] = costs[threadIdx.z] + dif[i + j* 12 + threadIdx.z * (144)];
		}
	}

	__syncthreads();

	if (threadIdx.x==0 & threadIdx.y==0 & threadIdx.z == 0) {
		bool isMinCenter_R_ref = false;
		bool isMinCenter_L_ref = false;
		if (costs[1] < costs[0] & costs[1] < costs[2]) { isMinCenter_R_ref = true; }
		if (costs[4] < costs[3] & costs[4] < costs[5]) { isMinCenter_L_ref = true; }
		if(isMinCenter_R_ref & isMinCenter_L_ref){
			
			resultIm[(blockIdx.y + int(kSize / 2))* nC + (blockIdx.x + MxDisparity+ int(kSize / 2)+2)] = true;
		}
	}
	__syncthreads();
	//cudaDeviceSynchronize();
	//printf("%i\n", int(minCostIndex * 255 / 20));
}




void ReadBothImages(shared_ptr<Mat> leftImage, shared_ptr<Mat> rightImage) {

	try {
		cout << "this is test" << endl;
		*rightImage = cv::imread("2.png", IMREAD_GRAYSCALE);   // Read the right image
																  //rightImage->convertTo(*rightImage, CV_64F);
		*leftImage = cv::imread("1.png", IMREAD_GRAYSCALE);   // Read the left image
																 //leftImage->convertTo(*leftImage, CV_64F);
	}
	catch (char* error) {
		cout << "can not load the " << error << " iamge" << endl;
	}

	//cv::resize(*rightImage, *rightImage, cv::Size(), 0.50, 0.50);
	//cv::resize(*leftImage, *leftImage, cv::Size(), 0.50, 0.50);

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

	//ofstream repotredResult;

	//shared_ptr<Mat> rightGrayImage = make_shared<Mat>(numOfRows, numOfColumns, CV_8UC1);
	//cv::cvtColor(*rightImage, *rightGrayImage, CV_RGB2BGR);
	//cv::imshow("gray image", *rightImage);
	//cv::waitKey(10000);



	uchar** imArray2DL= new uchar* [numOfRows];
	uchar** imArray2DR = new uchar*[numOfRows];
	bool** imArrary2DR_result= new bool*[numOfRows];
	for (int i = 0; i < numOfRows; i++) {
		imArray2DL[i] = new uchar[numOfColumns];
		imArray2DR[i] = new uchar[numOfColumns];
		imArrary2DR_result[i] = new bool[numOfColumns];
	}
	for (int j = 0; j < numOfRows; j++) {
		for (int i = 0; i < numOfColumns; i++) {
			imArray2DL[j][i] = leftImage->at<uchar>(j, i);
			imArray2DR[j][i] = rightImage->at<uchar>(j, i);
			imArrary2DR_result[j][i] = false;
		}
	}
	cout << "copy to array is done!!!!!!" << endl;
	uchar* imArrary1DL = new uchar[numOfColumns*numOfRows];
	uchar* imArrary1DR = new uchar[numOfColumns*numOfRows];
	bool* imArrary1DR_result = new bool[numOfColumns*numOfRows];
	int temp;
	for (int i = 0; i < numOfColumns*numOfRows; i++) {
		imArrary1DL[i] = imArray2DL[int(i / numOfColumns)][i%numOfColumns];
		imArrary1DR[i] = imArray2DR[int(i / numOfColumns)][i%numOfColumns];
		imArrary1DR_result[i]= imArrary2DR_result[int(i / numOfColumns)][i%numOfColumns];
	}


	uchar* imArray1DL_d;
	uchar* imArray1DR_d;
	bool* imArray1DResult_d;
	cudaMalloc((void**)&imArray1DL_d, numOfColumns*numOfRows * sizeof(uchar));
	cudaMalloc((void**)&imArray1DR_d, numOfColumns*numOfRows * sizeof(uchar));
	cudaMalloc((void**)&imArray1DResult_d, numOfColumns*numOfRows * sizeof(bool));
	cudaMemcpy(imArray1DL_d, imArrary1DL, numOfColumns*numOfRows * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(imArray1DR_d, imArrary1DR, numOfColumns*numOfRows * sizeof(uchar), cudaMemcpyHostToDevice);
	dim3 blocks3D(kernelSize, kernelSize,6);
	dim3 grid2D(numOfColumns-2*(maxDisparity+1)-(kernelSize-1), numOfRows-kernelSize-1,1);
	//addIntensity <<<(numOfRows*numOfColumns ), 1 >>>(190, imArray1DL_d);
	IDAS_Stereo_selective <<<grid2D, blocks3D >>>(kernelSize, maxDisparity, numOfColumns, selectedDisparity, imArray1DL_d, imArray1DR_d, imArray1DResult_d);
	cudaDeviceSynchronize();
	cudaMemcpy(imArrary1DR_result, imArray1DResult_d, numOfColumns*numOfRows * sizeof(bool), cudaMemcpyDeviceToHost);
	cout << "adding to array is done!!!!!!" << endl;
	for (int i = 0; i < numOfColumns*numOfRows; i++) {
		imArrary2DR_result[int(i / numOfColumns)][i%numOfColumns] = imArrary1DR_result[i];
	}

	for (int j = 0; j < numOfRows; j++) {
		for (int i = 0; i < numOfColumns; i++) {
				if(imArrary2DR_result[j][i]==true)
			leftImage->at<uchar>(j, i)=(uchar)255 ;
		}
	}
	
	cudaFree(imArray1DL_d);
	cudaFree(imArray1DR_d);
	cudaFree(imArray1DResult_d);
	chrono::high_resolution_clock::time_point stop = chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> duration = (stop - start);;
	auto value = duration.count();
	string duration_s = to_string(value);
	cout << "time of run is " << value << endl;
	imshow(" Left  !!!   .....", *leftImage);
	imwrite("test.png", *leftImage);
	imshow("After effect right image !!!   .....", *rightImage);

	waitKey(1000);
	//cout << int(imArray2DL[200][359]) << endl;








	printf("\n \n \n  \t \t \t :)  ");
	char str[80];
	scanf("%79s", str);
}