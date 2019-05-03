#include <iostream>  
#include "opencv2/opencv.hpp" 
#include "opencv2/video/background_segm.hpp"
using namespace cv;
using namespace std;

#define UNKNOWN_FLOW_THRESH 1e9  

void makecolorwheel(vector<Scalar> &colorwheel) //这里相当于做一个画板 
{
	int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;

	int i;

	/*把各种颜色丢进画板*/
	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
	for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
	for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
	for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
	for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
	for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

void motionToColor(Mat flow, Mat &color)
{
	if (color.empty()) //判断是否有数据
		color.create(flow.rows, flow.cols, CV_8UC3);

	static vector<Scalar> colorwheel; //这个容器里有r,g,b三种颜色 
	if (colorwheel.empty())         //做画板
		makecolorwheel(colorwheel);

	float maxrad = -1;  //确定运动范围，相当于运动半径

	/*查找最大光流值（根号（fx^2+fy^2）的最大值）用于将fx和fy归一化*/
	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);  //坐标为（i,j）的点的光流值
			float fx = flow_at_point[0];  //光流的两个分量
			float fy = flow_at_point[1];
			if ((fabs(fx) > UNKNOWN_FLOW_THRESH) || (fabs(fy) > UNKNOWN_FLOW_THRESH))  //光流分量分别和设定的阈值比较
				continue;
			float rad = sqrt(fx * fx + fy * fy);  //用于确定运动范围
			maxrad = maxrad > rad ? maxrad : rad;  //取最大值
		}
	}

	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;  //访问光流图的像素值
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);

			float fx = flow_at_point[0] / maxrad;  //归一化光流分量
			float fy = flow_at_point[1] / maxrad;
			if ((fabs(fx) > UNKNOWN_FLOW_THRESH) || (fabs(fy) > UNKNOWN_FLOW_THRESH))
			{
				data[0] = data[1] = data[2] = 0;  //
				continue;
			}
			float rad = sqrt(fx * fx + fy * fy);

			float angle = atan2(-fy, -fx) / CV_PI;
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);
			int k0 = (int)fk;
			int k1 = (k0 + 1) % colorwheel.size();
			float f = fk - k0;

			for (int b = 0; b < 3; b++)
			{
				float col0 = colorwheel[k0][b] / 255.0;
				float col1 = colorwheel[k1][b] / 255.0;
				float col = (1 - f) * col0 + f * col1;
				if (rad <= 1)
					col = 1 - rad * (1 - col); // 增大饱和半径  
				else
					col *= .75; // 超出范围  
				data[2 - b] = (int)(255.0 * col);
			}
		}
	}
}

bool cmp(vector<Point>&v1, vector<Point>&v2)//用于轮廓排序的函数
{
	return contourArea(v1) > contourArea(v2);  //轮廓按面积从大到小排列
}

const float scale = 0.25;//由于利用原始图像计算光流耗时长，所以将原图像缩小为原来的1/4再计算
const int invscale = (int)1 / scale;//这里是将在小图像中找到的目标的矩形
int main()
{
	VideoCapture cap;
	cap.open(0);
	//cap.open("C:/Users/lenovo/Desktop/【重要】视觉程序/追踪程序/cv_walker.mp4");  

	if (!cap.isOpened())
	{
		cout << "打开视频/相机失败！\n";
		return -1;
	}

	Mat prevgray, gray, flow, frame, halfframe;  //后面要用到的变量
	namedWindow("flow", 1);

	Mat motion2color;//光流图  
	Ptr<BackgroundSubtractorMOG2> mog = createBackgroundSubtractorMOG2();//背景模型
	Mat fgmask;//前景（二值图像）
	Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));//形态学滤波结构元

	Rect rect;//轮廓的外接矩形
	//计算直方图要用到的变量
	vector<Mat>hsvframe;
	Mat hsv;
	vector<int>hueNum;//直方图的Bin数
	hueNum.push_back(30);
	vector<float>hueRange;//计算的范围
	hueRange.push_back(0);
	hueRange.push_back(180);
	vector<int>channels;//计算直方图的通道
	channels.push_back(0);//H通道
	MatND hist;

	while (1)
	{
		double t = (double)getTickCount();
	Begin://当视频中没有运动的目标时，回到这重新采集图像
		cap >> frame;
		if (frame.empty())
		{
			cout << "未采集到图像！\n";
			return -1;
		}
		imshow("原始图像", frame);

		resize(frame, halfframe, Size(), scale, scale);//缩放图像，高斯和光流都缩小
		imshow("缩小原始图像", halfframe);

		mog->apply(halfframe, fgmask, 0.01);//这里是高斯背景模型提取的前景，二值图像

		Mat mask(halfframe.rows, halfframe.cols, CV_8UC1);//计算直方图的呃掩码

		/*下面用光流法提取前景*/
		cvtColor(halfframe, gray, CV_BGR2GRAY); //计算光流时要转换为灰度图像

		if (prevgray.data)
		{
			calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
			motionToColor(flow, motion2color);//计算光流  
		}

		std::swap(prevgray, gray);  //将gray和pregray的内容交换，相当于取两个相邻的帧，用于计算光流（计算光流时要两个相邻的帧）

		if (!motion2color.data)//没有光流图时重新采集图像
		{
			goto Begin;
		}
		imshow("flow", motion2color);

		t = (double)getTickCount() - t;  //计算光流的时间
		cout << "计算光流的时间: " << t / ((double)getTickFrequency()*1000.) << "ms" << endl;

		Mat ofgray;//光流图转换为灰度图
		cvtColor(motion2color, ofgray, COLOR_BGR2GRAY);
		imshow("ofgray", ofgray);

		threshold(ofgray, ofgray, 242, 255, THRESH_BINARY_INV);//阈值化处理提取前景
		imshow("threshold", ofgray);

		Mat fgimg;//将高斯背景模型提取的前景和光流法提取的前景融合，得到最终的前景用于后续的操作
		fgimg = fgmask & ofgray;

		//GaussianBlur(ofgray,ofgray,Size(3,3),0,0);//滤波去除噪声
		medianBlur(fgimg, fgimg, 5);//中值滤波
		imshow("最终前景", fgimg);
		/*下面三步是对前景进行预处理去除噪声和填补空洞*/

		morphologyEx(fgimg, fgimg, MORPH_DILATE, element);//膨胀处理
		imshow("前景膨胀", fgimg);//显示前景
		morphologyEx(fgimg, fgimg, MORPH_ERODE, element);//腐蚀处理
		imshow("前景腐蚀", fgimg);//显示前景

		/*开始查找前景的轮廓*/
		vector<vector<Point>>contours;
		vector<Vec4i>hierarchy;
		findContours(fgimg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		if (contours.size() < 1)
		{
			goto Begin;
		}

		std::sort(contours.begin(), contours.end(), cmp);//将轮廓的面积从小到大排序

		cvtColor(halfframe, hsv, COLOR_BGR2HSV);//将BGR颜色空间转换为HSV颜色空间，追踪是基于H通道的直方图
		hsvframe.push_back(hsv);

		for (size_t i = 0; i < contours.size(); ++i)
		{
			if (contourArea(contours[i]) < contourArea(contours[0]) / 5)
			{
				break;
			}
			rect = boundingRect(contours[i]);//获得轮廓的外接矩形
			mask = 0;
			mask(rect) = 255;//rect为ROI设为白色

			/*下面开始计算直方图*/
			calcHist(hsvframe, channels, mask, hist, hueNum, hueRange);
			double maxValue;
			minMaxLoc(hist, 0, &maxValue, 0, 0);
			hist = hist * 255 / maxValue;

			/*下面开始反向投影*/
			Mat backProject;
			calcBackProject(hsvframe, channels, hist, backProject, hueRange, 1);

			/*下面开始追踪*/
			Rect initRec = rect;//用于初始化的矩形
			RotatedRect rotRect = CamShift(backProject, initRec,  //开始跟踪
				TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1)); //终止条件
			Rect rectB = rotRect.boundingRect();
			rect &= rectB;//取两个矩形的公共部分

			Rect rectD(rect.x*invscale, rect.y*invscale, rect.width*invscale, rect.height*invscale);
			rectangle(frame, rectD, Scalar(0, 0, 255), 3, 8);
			//resize(frame,frame,Size(1280,720));//这里是把原始图像resize到想显示的图像分辨率（行*列）
			imshow("追踪结果", frame);
		}

		char c = (char)waitKey(10);
		if (c == (char)27 || c == 'q' || c == 'Q')
			break;
	}  //end while
	return 0;
}

