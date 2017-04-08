#include <iostream>
#include <cv.h>
#include <highgui.h> 
#include <math.h>
#include <stdio.h>

using namespace std;
using namespace cv;

int I2_SobelThreshold = 50;
int I3_SobelThreshold = 254;

int I1_HoughpolarThreshold = 300;
int I2_HoughpolarThreshold = 100;
int I3_HoughpolarThreshold = 250;

int HoughMax = 30;
int I1_HoughRectangularThreshold = 160;
int I2_HoughRectangularThreshold = 130;
int I3_HoughRectangularThreshold = 350;

float cos_tab[181];
float sin_tab[181];

Mat img1 = imread("T1.jpg");
Mat polarCoordinates1 = imread("T1.jpg");
Mat rectangularCoordinates1 = imread("T1.jpg");

Mat img2 = imread("T2.jpg");
Mat polarCoordinates2 = imread("T2.jpg");
Mat rectangularCoordinates2 = imread("T2.jpg");

Mat img3 = imread("T3.jpg");
Mat polarCoordinates3 = imread("T3.jpg");
Mat rectangularCoordinates3 = imread("T3.jpg");

Mat edge1(Size(img1.cols, img1.rows), CV_8UC3, CV_RGB(255, 255, 255));

Mat Sobel_edge2(Size(img2.cols, img2.rows), CV_8UC3, CV_RGB(0, 0, 0));

Mat gray3(Size(img3.cols, img3.rows), CV_8UC3, CV_RGB(0, 0, 0));
Mat Sobel_edge3(Size(img3.cols, img3.rows), CV_8UC3, CV_RGB(0, 0, 0));

void edge1_Detection(int y, int x){
	int sum = 0;
	if (img1.at<Vec3b>(y, x)[0] > 0){
		sum = 255;
	}
	edge1.at<Vec3b>(y, x)[0] = sum;
	edge1.at<Vec3b>(y, x)[1] = sum;
	edge1.at<Vec3b>(y, x)[2] = sum;
}
void Sobel_edge2_Detection(int y, int x){
	int sum = 0, Gx = 0, Gy = 0;
	Gx = -(img2.at<Vec3b>(y - 1, x - 1)[0] + 2 * img2.at<Vec3b>(y - 1, x)[0] + img2.at<Vec3b>(y - 1, x + 1)[0]);
	Gx = Gx + (img2.at<Vec3b>(y + 1, x - 1)[0] + 2 * img2.at<Vec3b>(y + 1, x)[0] + img2.at<Vec3b>(y + 1, x + 1)[0]);
	Gy = -(img2.at<Vec3b>(y - 1, x - 1)[0] + 2 * img2.at<Vec3b>(y, x - 1)[0] + img2.at<Vec3b>(y + 1, x - 1)[0]);
	Gy = Gy + (img2.at<Vec3b>(y - 1, x + 1)[0] + 2 * img2.at<Vec3b>(y, x + 1)[0] + img2.at<Vec3b>(y + 1, x + 1)[0]);
	sum = abs(Gx) + abs(Gy);
	if (I2_SobelThreshold <= sum){
		sum = 255;
	}else{
		sum = 0;
	}
	Sobel_edge2.at<Vec3b>(y, x)[0] = sum;
	Sobel_edge2.at<Vec3b>(y, x)[1] = sum;
	Sobel_edge2.at<Vec3b>(y, x)[2] = sum;
}
void gray_img3(int y, int x){
	int B = img3.at<Vec3b>(y, x)[0];
	int G = img3.at<Vec3b>(y, x)[1];
	int R = img3.at<Vec3b>(y, x)[2];

	int gray = 0.299 * R + 0.587 * G + 0.114 * B;
	gray3.at<Vec3b>(y, x)[0] = gray;
	gray3.at<Vec3b>(y, x)[1] = gray;
	gray3.at<Vec3b>(y, x)[2] = gray;
}
void Sobel_edge3_Detection(int y, int x){
	int sum = 0, Gx = 0, Gy = 0;
	Gx = -(img3.at<Vec3b>(y - 1, x - 1)[0] + 2 * img3.at<Vec3b>(y - 1, x)[0] + img3.at<Vec3b>(y - 1, x + 1)[0]);
	Gx = Gx + (img3.at<Vec3b>(y + 1, x - 1)[0] + 2 * img3.at<Vec3b>(y + 1, x)[0] + img3.at<Vec3b>(y + 1, x + 1)[0]);
	Gy = -(img3.at<Vec3b>(y - 1, x - 1)[0] + 2 * img3.at<Vec3b>(y, x - 1)[0] + img3.at<Vec3b>(y + 1, x - 1)[0]);
	Gy = Gy + (img3.at<Vec3b>(y - 1, x + 1)[0] + 2 * img3.at<Vec3b>(y, x + 1)[0] + img3.at<Vec3b>(y + 1, x + 1)[0]);
	sum = abs(Gx) + abs(Gy);
	if (I3_SobelThreshold <= sum)	{
		sum = 255;
	}else{
		sum = 0;
	}
	Sobel_edge3.at<Vec3b>(y, x)[0] = sum;
	Sobel_edge3.at<Vec3b>(y, x)[1] = sum;
	Sobel_edge3.at<Vec3b>(y, x)[2] = sum;
}
void cos_sin_map(){
	float pi = 3.14159265358979, theta = 0;
	for (int angle = 0; angle <= 180; angle++){
		theta = angle * pi / 180.0;
		cos_tab[angle] = cos(theta);
		sin_tab[angle] = sin(theta);
	}
}
void Hough_transform_polar(int cols, int rows, int which_picture, int Counter_ThresSholding){
	int rmax = 0;
	rmax = (cols * cols) + (rows * rows);
	rmax = sqrt((double)rmax);
	rmax = rmax + 1;
	int **hough_matrix;
	int hough_heigh = rmax * 2;
	int hough_width = 180;
	hough_matrix = new int*[hough_heigh];
	for (int i = 0; i <= hough_heigh; i++){
		hough_matrix[i] = new int[hough_width];
	}
	for (int x = 0; x <= hough_heigh; x++) {
		for (int y = 0; y <= hough_width; y++){
			hough_matrix[x][y] = 0;
		}
	}
	for (int y = 0; y < rows; y++){	
		for (int x = 0; x < cols; x++) {
			switch (which_picture)			{
			case 1:
				if (edge1.at<Vec3b>(y, x)[0] == 255){
					for (int angle = 0; angle <= hough_width; angle++){
						float p = 0.0;
						int np = 0;
						p = y * cos_tab[angle] + x * sin_tab[angle];
						np = p + rmax - 1;
						hough_matrix[np][angle] = hough_matrix[np][angle] + 1;
					}
				}
				break;
			case 2:
				if (Sobel_edge2.at<Vec3b>(y, x)[0] == 255){
					for (int angle = 0; angle <= hough_width; angle++){
						float p = 0.0;
						int np = 0;
						p = y * cos_tab[angle] + x * sin_tab[angle];
						np = p + rmax - 1;
						hough_matrix[np][angle] = hough_matrix[np][angle] + 1;
					}
				}
				break;
			case 3:
				if (Sobel_edge3.at<Vec3b>(y, x)[0] == 255){
					for (int angle = 0; angle <= hough_width; angle++){
						float p = 0.0;
						int np = 0;
						p = y * cos_tab[angle] + x * sin_tab[angle];
						np = p + rmax - 1;
						hough_matrix[np][angle] = hough_matrix[np][angle] + 1;
					}
				}
				break;
			}
		}
	}
	int max_matrix[100][4] = { 0 };
	int counter = 0;
	for (int x = 0; x <= hough_heigh; x++) {
		for (int y = 0; y <= hough_width; y++){
			if (hough_matrix[x][y] >= Counter_ThresSholding)			{
				max_matrix[counter][0] = counter + 1;
				max_matrix[counter][1] = x;
				max_matrix[counter][2] = y;
				max_matrix[counter][3] = hough_matrix[x][y];
				counter++;
			}
		}
	}
	counter = 0;
	do{
		int np = 0, p = 0, angle = 0;
		np = max_matrix[counter][1];
		p = np - rmax + 1;
		angle = max_matrix[counter][2];

		for (int y = 0; y < rows; y++){
			for (int x = 0; x < cols; x++){
				switch (which_picture){
				case 1:
					if (edge1.at<Vec3b>(y, x)[0] == 255){
						int p_ans = y * cos_tab[angle] + x * sin_tab[angle];
						if ((p >= p_ans) && (p <= p_ans + 1)){
							polarCoordinates1.at<Vec3b>(y, x)[0] = 255;
							polarCoordinates1.at<Vec3b>(y, x)[1] = 0;
							polarCoordinates1.at<Vec3b>(y, x)[2] = 0;
						}
					}
					break;
				case 2:
					if (Sobel_edge2.at<Vec3b>(y, x)[0] == 255){
						int p_ans = y * cos_tab[angle] + x * sin_tab[angle];
						if ((p >= p_ans) && (p <= p_ans + 1)){
							polarCoordinates2.at<Vec3b>(y, x)[0] = 255;
							polarCoordinates2.at<Vec3b>(y, x)[1] = 0;
							polarCoordinates2.at<Vec3b>(y, x)[2] = 0;
						}
					}
					break;
				case 3:
					if (angle < 25){
						if (Sobel_edge3.at<Vec3b>(y, x)[0] == 255){
							int p_ans = y * cos_tab[angle] + x * sin_tab[angle];
							if ((p >= p_ans) && (p <= p_ans + 1)){
								polarCoordinates3.at<Vec3b>(y, x)[0] = 255;
								polarCoordinates3.at<Vec3b>(y, x)[1] = 0;
								polarCoordinates3.at<Vec3b>(y, x)[2] = 0;
							}
						}
					}
					break;
				}
			}
		}
		counter++;
	} while (max_matrix[counter][0] != 0);
}
void Hough_transform_rectangular(int cols, int rows, int which_picture, int Counter_ThresSholding){	
	int **hough_matrix;
	int hough_heigh = HoughMax * HoughMax * HoughMax;
	int hough_width = HoughMax * 100;
	hough_matrix = new int*[hough_heigh];
	for (int i = 0; i <= hough_heigh; i++){
		hough_matrix[i] = new int[hough_width];
	}
	for (int x = 0; x <= hough_heigh; x++) {
		for (int y = 0; y <= hough_width; y++){
			hough_matrix[x][y] = 0;
		}
	}
	for (int y = 0; y < rows; y++){
		for (int x = 0; x < cols; x++) {
			switch (which_picture)
			{
			case 1:
				if (edge1.at<Vec3b>(y, x)[0] == 255)				{
					for (int integer_a = -(hough_width / (2 * 100)); integer_a < (hough_width / (2 * 100)); integer_a++){
						for (int decimal_a_1 = 0; decimal_a_1 < 10; decimal_a_1++){
							for (int decimal_a_2 = 0; decimal_a_2 < 10; decimal_a_2++){
								float a = integer_a + (float)decimal_a_1 / 10 + (float)decimal_a_2 / 100;
								float b = x*(-a) + y;
								int matrix_a = floor(a * 100) + (hough_width / 2);
								int matrix_b = b + (hough_heigh / 2);
								hough_matrix[matrix_b][matrix_a] = hough_matrix[matrix_b][matrix_a] + 1;
							}
						}
					}
				}
				break;
			case 2:
				if (Sobel_edge2.at<Vec3b>(y, x)[0] == 255){
					for (int integer_a = -(hough_width / (2 * 100)); integer_a < (hough_width / (2 * 100)); integer_a++){
						for (int decimal_a_1 = 0; decimal_a_1 < 10; decimal_a_1++){
							for (int decimal_a_2 = 0; decimal_a_2 < 10; decimal_a_2++){
								float a = integer_a + (float)decimal_a_1 / 10 + (float)decimal_a_2 / 100;
								float b = x*(-a) + y;
								int matrix_a = floor(a * 100) + (hough_width / 2);
								int matrix_b = b + (hough_heigh / 2);
								hough_matrix[matrix_b][matrix_a] = hough_matrix[matrix_b][matrix_a] + 1;
							}
						}
					}
				}
				break;
			case 3:
				if (Sobel_edge3.at<Vec3b>(y, x)[0] == 255){
					for (int integer_a = -(hough_width / (2 * 100)); integer_a < (hough_width / (2 * 100)); integer_a++){
						for (int decimal_a_1 = 0; decimal_a_1 < 10; decimal_a_1++){
							for (int decimal_a_2 = 0; decimal_a_2 < 10; decimal_a_2++){
								float a = integer_a + (float)decimal_a_1 / 10 + (float)decimal_a_2 / 100;
								float b = x*(-a) + y;
								int matrix_a = floor(a * 100) + (hough_width / 2);
								int matrix_b = b + (hough_heigh / 2);
								hough_matrix[matrix_b][matrix_a] = hough_matrix[matrix_b][matrix_a] + 1;
							}
						}
					}
				}
				break;
			}
		}
	}
	int max_matrix[300][4] = { 0 };
	int counter = 0;
	for (int b = 0; b <= hough_heigh; b++) {
		for (int a = 0; a <= hough_width; a++){
			if (hough_matrix[b][a] >= Counter_ThresSholding){
				max_matrix[counter][0] = counter + 1;
				max_matrix[counter][1] = b - (hough_heigh / 2);
				max_matrix[counter][2] = a - (hough_width / 2);
				max_matrix[counter][3] = hough_matrix[b][a];
				counter++;
			}
		}
	}
	counter = 0;
	do{
		float a = 0;
		int b = 0;
		int y = 0;
		b = max_matrix[counter][1];
		a = (float)max_matrix[counter][2] / 100;
		printf("%d	|	%d	|	%f	|	%d	\n", max_matrix[counter][0], max_matrix[counter][1], a, max_matrix[counter][3]);
		for (int x = 0; x < cols; x++) {
			switch (which_picture){
			case 1:
				y = a*x + b;
				if ((y < rows) && (y >= 0))	{
					rectangularCoordinates1.at<Vec3b>(y, x)[0] = 255;
					rectangularCoordinates1.at<Vec3b>(y, x)[1] = 0;
					rectangularCoordinates1.at<Vec3b>(y, x)[2] = 0;
				}
				break;
			case 2:
				y = a*x + b;
				if ((y < rows) && (y >= 0))	{
					rectangularCoordinates2.at<Vec3b>(y, x)[0] = 255;
					rectangularCoordinates2.at<Vec3b>(y, x)[1] = 0;
					rectangularCoordinates2.at<Vec3b>(y, x)[2] = 0;
				}
				break;
			case 3:
				if (a >(-0.25))	{
					y = a*x + b;
					if ((y < rows) && (y >= 0))	{
						rectangularCoordinates3.at<Vec3b>(y, x)[0] = 255;
						rectangularCoordinates3.at<Vec3b>(y, x)[1] = 0;
						rectangularCoordinates3.at<Vec3b>(y, x)[2] = 0;
					}
				}
				break;
			}
		}
		counter++;
	} while (max_matrix[counter][0] != 0);
}
int main() {
	
	for (int y = 0; y < img1.rows; y++){
		for (int x = 0; x < img1.cols; x++) {
			edge1_Detection(y, x);
		}
	}
	
	for (int y = 1; y < img2.rows - 1; y++){
		for (int x = 1; x < img2.cols - 1; x++) {
			Sobel_edge2_Detection(y, x);
		}
	}
	
	for (int y = 0; y < img3.rows; y++){//I3
		for (int x = 0; x < img3.cols; x++) {
			gray_img3(y, x);
		}
	}
	for (int y = 1; y < img3.rows - 1; y++){
		for (int x = 1; x < img3.cols - 1; x++) {
			Sobel_edge3_Detection(y, x);
		}
	}
	Hough_transform_rectangular(img1.cols, img1.rows, 1, I1_HoughRectangularThreshold);
	Hough_transform_rectangular(img2.cols, img2.rows, 2, I2_HoughRectangularThreshold);
	Hough_transform_rectangular(img3.cols, img3.rows, 3, I3_HoughRectangularThreshold);

	cos_sin_map();
	Hough_transform_polar(img1.cols, img1.rows, 1, I1_HoughpolarThreshold);
	Hough_transform_polar(img2.cols, img2.rows, 2, I2_HoughpolarThreshold);
	Hough_transform_polar(img3.cols, img3.rows, 3, I3_HoughpolarThreshold);

	imshow("T1_rectangularCoordinates", rectangularCoordinates1);
	imshow("T1_polarCoordinates", polarCoordinates1);
	imwrite("T1_rectangularCoordinates.jpg", rectangularCoordinates1);
	imwrite("T1_polarCoordinates.jpg", polarCoordinates1);

	imshow("T2_rectangularCoordinates", rectangularCoordinates2);
	imshow("T2_polarCoordinates", polarCoordinates2);
	imwrite("T2_rectangularCoordinates.jpg", rectangularCoordinates2);
	imwrite("T2_polarCoordinates.jpg", polarCoordinates2);

	imshow("T3_rectangularCoordinates", rectangularCoordinates3);
	imshow("T3_polarCoordinates", polarCoordinates3);
	imwrite("T3_rectangularCoordinates.jpg", rectangularCoordinates3);
	imwrite("T3_polarCoordinates.jpg", polarCoordinates3);

	waitKey(0);
}