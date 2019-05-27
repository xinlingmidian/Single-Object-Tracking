#include "lbp.h"
//提取HLBP纹理特征  Haar+MBP

//Sobel算子  
//int Soble[8][9] = { {1,2,1,0,0,0,-1,-2,-1},{2,1,0,1,0,-1,0,-1,-2},{1,0,-1,2,0,-2,1,0,-1},{0,-1,-2,1,0,-1,2,1,0},
//	   {-1,-2,-1,0,0,0,1,2,1},{-2,-1,0,-1,0,1,0,1,2},{-1,0,1,-2,0,2,-1,0,1},{0,1,2,-1,0,1,-2,-1,0} };

void correlation(uchar* center,uchar* result) ;
//LBP  
Mat LBP(Mat img)
{
	Mat result;
	result.create(img.rows, img.cols, img.type());
	result.setTo(0);
	

	/*for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			uchar center = img.at<uchar>(i, j);
			uchar code = 0;
			code |= (img.at<uchar>(i - 1, j - 1) >= center) << 7;
			code |= (img.at<uchar>(i - 1, j) >= center) << 6;
			code |= (img.at<uchar>(i - 1, j + 1) >= center) << 5;
			code |= (img.at<uchar>(i, j + 1) >= center) << 4;
			code |= (img.at<uchar>(i + 1, j + 1) >= center) << 3;
			code |= (img.at<uchar>(i + 1, j) >= center) << 2;
			code |= (img.at<uchar>(i + 1, j - 1) >= center) << 1;
			code |= (img.at<uchar>(i, j - 1) >= center) << 0;
			result.at<uchar>(i - 1, j - 1) = code;
		}
	}*/

	

	for (int i = 2; i < img.rows - 2; i++) {
		for (int j = 2; j < img.cols - 2; j++) {
			//uchar center = img.at<uchar>(i, j);
			int mid = 0;
			uchar sobel[24], temp[8];
			sobel[0] = img.at<uchar>(i - 1, j - 1);
			sobel[1] = img.at<uchar>(i - 1, j);
			sobel[2] = img.at<uchar>(i - 1, j + 1);
			sobel[3] = img.at<uchar>(i, j + 1);
			sobel[4] = img.at<uchar>(i + 1, j + 1);
			sobel[5] = img.at<uchar>(i + 1, j);
			sobel[6] = img.at<uchar>(i + 1, j - 1);
			sobel[7] = img.at<uchar>(i, j - 1);
			sobel[8] = img.at<uchar>(i - 2, j - 2);
			sobel[9] = img.at<uchar>(i - 2, j - 1);
			sobel[10] = img.at<uchar>(i - 2, j);
			sobel[11] = img.at<uchar>(i - 2, j + 1);
			sobel[12] = img.at<uchar>(i - 2, j + 2);
			sobel[13] = img.at<uchar>(i - 1, j + 2);
			sobel[14] = img.at<uchar>(i, j + 2);
			sobel[15] = img.at<uchar>(i + 1, j + 2);
			sobel[16] = img.at<uchar>(i + 2, j + 2);
			sobel[17] = img.at<uchar>(i + 2, j + 1);
			sobel[18] = img.at<uchar>(i + 2, j);
			sobel[19] = img.at<uchar>(i + 2, j - 1);
			sobel[20] = img.at<uchar>(i + 2, j - 2);
			sobel[21] = img.at<uchar>(i + 1, j - 2);
			sobel[22] = img.at<uchar>(i, j - 2);
			sobel[23] = img.at<uchar>(i - 1, j - 2);
			for (int k = 0; k < 24; k++)
			{
				mid = mid + sobel[k];
			}
			mid = mid / 24;
			//mid = 5;
			correlation(sobel, temp);
			uchar code = 0;
			code |= (temp[0] >= mid) << 7;
			code |= (temp[1] >= mid) << 6;
			code |= (temp[2] >= mid) << 5;
			code |= (temp[3] >= mid) << 4;
			code |= (temp[4] >= mid) << 3;
			code |= (temp[5] >= mid) << 2;
			code |= (temp[6] >= mid) << 1;
			code |= (temp[7] >= mid) << 0;
			result.at<uchar>(i - 1, j - 1) = code;
		}
	}

	return result;
}

void correlation(uchar* center, uchar* result)
{
	result[0] = center[22] + center[23] + center[8] + center[9] + center[10] - center[6] - center[7] - center[0] - center[1] - center[2];
	result[1] = center[8] + center[9] + center[10] + center[11] + center[12] - center[23] - center[0] - center[1] - center[2] - center[13];
	result[2] = center[10] + center[11] + center[12] + center[13] + center[14] - center[0] - center[1] - center[2] - center[3] - center[4];
	result[3] = center[12] + center[13] + center[14] + center[15] + center[16] - center[11] - center[2] - center[3] - center[4] - center[17];
	result[4] = center[14] + center[15] + center[16] + center[17] + center[18] - center[2] - center[3] - center[4] - center[5] - center[6];
	result[5] = center[16] + center[17] + center[18] + center[19] + center[20] - center[15] - center[4] - center[5] - center[6] - center[21];
	result[6] = center[18] + center[19] + center[20] + center[21] + center[22] - center[4] - center[5] - center[6] - center[7] - center[0];
	result[7] = center[20] + center[21] + center[22] + center[23] + center[8] - center[19] - center[6] - center[7] - center[0] - center[9];
} 