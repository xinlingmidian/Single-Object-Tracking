#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include<vector>
using namespace cv;
using namespace std;

bool cmp(vector<Point> &v1, vector<Point> &v2)//将检测到的轮廓按面积大小进行排序
{
	return contourArea(v1) > contourArea(v2);
}

const float scale = 0.5;//缩放系数
const int invscale = (int)1 / scale;//导数是在追踪到目标后将举行画到原始图像帧中

int main()
{
	//VideoCapture cap("C:/Users/lenovo/Desktop/【重要】视觉程序/追踪程序/highwayII_raw.AVI");//打开视频
	VideoCapture cap(0);//相机
	if (!cap.isOpened())//判断是否成功打开
	{
		cout << "打开视频/相机失败！\n";
		return -1;
	}

	Mat frame, fgmask;//两个变量分别为原图像、提取前景（二值形式：前景为1，背景为0）
	int frameNum = 0;//帧数
	Ptr<BackgroundSubtractorMOG2> mog = createBackgroundSubtractorMOG2();
	//BackgroundSubtractorMOG2 mog;//高斯背景模型对象
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));//形态学滤波结构元
	Rect rect;//轮廓外部矩形边界

	/*计算直方图用到的变量*/
	Mat hsvFrame;
	vector<Mat>vecHsvFrame;
	vector<int> channels;
	channels.push_back(0);//只绘制h通道直方图
	vector<int> hueNum;
	hueNum.push_back(30);//bin=30,即30个柱形图
	vector<float>hueRanges;//范围
	hueRanges.push_back(0);
	hueRanges.push_back(180);
	MatND hist;

	while (1)//开始处理图像
	{
	Begin://当没有检测到目标是重新采集图像
		cap >> frame;//读取帧
		if (frame.empty())//是否读取到帧
		{
			cout << "未采集到图像！\n";
			break;
		}
		imshow("原始视频", frame);//显示原视频

		Mat halfframe;
		resize(frame, halfframe, Size(), scale, scale);//将原始帧尺寸转换为原来的一半
		imshow("halfframe", halfframe);

		Mat mask(halfframe.rows, halfframe.cols, CV_8UC1);//求直方图的掩码

		mog->apply(halfframe, fgmask, 0.01);//提取前景，0.01为学习率
		imshow("前景", fgmask);//显示前景

		/*对前景进行处理*/
		medianBlur(fgmask, fgmask, 5);//中值滤波
		imshow("前景滤波", fgmask);//显示前景
		morphologyEx(fgmask, fgmask, MORPH_DILATE, element);//膨胀处理
		imshow("前景膨胀", fgmask);//显示前景
		morphologyEx(fgmask, fgmask, MORPH_ERODE, element);//腐蚀处理
		imshow("前景腐蚀", fgmask);//显示前景

		/*查找前景的轮廓*/
		vector<vector<Point>>contours;//定义函数参数
		vector<Vec4i>hierarchy;

		findContours(fgmask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);//查找轮廓
		if (contours.size() < 1)//没有找到轮廓时重新采集图像
		{
			frameNum++;
			goto Begin;
		}
		sort(contours.begin(), contours.end(), cmp);//轮廓按面积从大到小进行排序

		cvtColor(halfframe, hsvFrame, COLOR_BGR2HSV);//转换至hsv空间
		vecHsvFrame.push_back(hsvFrame);

		for (size_t i = 0; i < contours.size(); ++i)
		{
			if (contourArea(contours[i]) < contourArea(contours[0]) / 5)//删除小轮廓
				break;
			rect = boundingRect(contours[i]);//矩形外部边界
			mask = 0;
			mask(rect) = 255;//rect为ROI设为白色

			calcHist(vecHsvFrame, channels, mask, hist, hueNum, hueRanges, false);//计算直方图
			double maxValue;
			minMaxLoc(hist, 0, &maxValue, 0, 0);//获得直方图中的最大值
			hist = hist * 255 / maxValue;

			Mat backProject;//计算反向投影用于追踪
			calcBackProject(vecHsvFrame, channels, hist, backProject, hueRanges, 1);

			Rect search = rect;//初始化跟踪的搜索框
			RotatedRect trackBox = CamShift(backProject, search,          //进行跟踪
				TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
			Rect rect2 = trackBox.boundingRect();
			rect &= rect2;

			Rect fullrect(rect.x*invscale, rect.y*invscale, rect.width*invscale, rect.width*invscale);//将在缩小的帧中找到的目标恢复至原始大小
			rectangle(frame, fullrect, Scalar(0, 0, 255), 3);

			imshow("追踪结果", frame);
		}//end for

		char c = (char)waitKey(10);
		if (c == (char)27 || c == 'q' || c == 'Q')
			break;
	}//end while();
	return 1;
}