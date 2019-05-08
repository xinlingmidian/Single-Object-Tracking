#include "stdafx.h"
#include<iostream> 
#include <fstream>
#include <sstream>
#include <algorithm>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <math.h>
#include <time.h>
#include "kcftracker.hpp"
#include "lbp.h"

using namespace std;
using namespace cv;

# define R_BIN      8  /* 红色分量的直方图条数 */
# define G_BIN      8  /* 绿色分量的直方图条数 */
# define B_BIN      8  /* 蓝色分量的直方图条数 */ 

# define R_SHIFT    5  /* 与上述直方图条数对应 */
# define G_SHIFT    5  /* 的R、G、B分量左移位数 */
# define B_SHIFT    5  /* log2( 256/16 )为移动位数*/
# define LBP_SHIFT  5  /* LBP特征的直方图偏移量*/
/*为了产生一个服从正态分布的随机数，采用Park and Miller方法，有参考论文，看不懂*/
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876


typedef struct __SpaceState {  /* 状态空间变量 */
	int xt;               /* x坐标位置 */
	int yt;               /* y坐标位置 */
	float v_xt;           /* x方向运动速度 */
	float v_yt;           /* y方向运动速度 */
	int Hxt;              /* x方向半窗宽 */
	int Hyt;              /* y方向半窗宽 */
	float at_dot;         /* 尺度变换速度,粒子所代表的那一片区域的尺度变化速度 */
} SPACESTATE;


unsigned char *img;							 //把iplimg改到char*,便于计算...
unsigned char * himg;

int Wid, Hei;								 //图像的大小
//int WidIn,HeiIn;							 //输入的半宽与半高
int WidOut, HeiOut;							 //输出的半宽与半高
int xin, yin;								 //跟踪时输入的中心点
int xout, yout;								 //跟踪时得到的输出中心点
int nbin = 512;									 //直方图条数
int Lsize = 8;                               //LBP直方图条数 8

const float DELTA_T = (float)0.05;           //帧频，可以为30、25、15、10等0.05
//const int POSITION_DISTURB = 15;           //位置扰动幅度15
const float VELOCITY_DISTURB = 40.0;         //速度扰动幅值40.0
const float SCALE_DISTURB = 0.0;             //窗宽高扰动幅度0.0
const float SCALE_CHANGE_D = (float)0.001;   //尺度变换速度扰动幅度0.001
const float Pi_Thres = (float)0.9;           //权重阈值0.9
long ran_seed = 802163120;                   //随机数种子,为全局变量,设置缺省值802163120
int NParticle = 100;                         //粒子个数75
float* ModelHist = NULL;                     //模型直方图
float * LBPHist = NULL;                      //颜色直方图 
SPACESTATE* states = NULL;                   //状态数组
float* weights = NULL;                       //每个粒子的权重

bool pause = true;                           //是否暂停
bool drawing_box = false;                    //判断是否画矩形框
float xMin, yMin, width, height;             //跟踪框的坐标(xMin,yMin),宽,高


Mat frame,gray_image,texture;				 //帧
Rect initbox;                                //初始跟踪框


/*
计算一幅图像中某个区域的彩色直方图分布
输入参数：
int x0, y0：           指定图像区域的中心点
int Wx, Hy：           指定图像区域的半宽和半高
unsigned char * image：图像数据，按从左至右，从上至下的顺序扫描，
颜色排列次序：RGB, RGB, ...
(或者：YUV, YUV, ...)
int W, H：             图像的宽和高
输出参数：
float * ColorHist：    彩色直方图，颜色索引按：
i = r * G_BIN * B_BIN + g * B_BIN + b排列
int bins：             彩色直方图的条数R_BIN*G_BIN*B_BIN（这里取8x8x8=512）
*/
void CalcuColorHistogram(int x0, int y0, int Wx, int Hy,
	unsigned char * image, int W, int H,
	float * ColorHist, float * TextureHist, int bins)
{

	int x_begin, y_begin;  /* 指定图像区域的左上角坐标 */
	int y_end, x_end;
	int x, y, i, index, index2;
	int r, g, b;
	float k, r2, f;
	int a2, count = 0;

	for (i = 0; i < bins; i++)     /* 直方图各个值赋0 有问题*/
		ColorHist[i] = 0.0;
	for (i = 0; i < Lsize; i++)
		TextureHist[i] = 0.0;
	/* 考虑特殊情况：x0, y0在图像外面，或者，Wx<=0, Hy<=0 */
	/* 此时强制令彩色直方图为0 */
	if ((x0 < 0) || (x0 >= W) || (y0 < 0) || (y0 >= H)
		|| (Wx <= 0) || (Hy <= 0))
	{
		return;
	}

	x_begin = x0 - Wx;               /* 计算实际高宽和区域起始点 */
	y_begin = y0 - Hy;
	if (x_begin < 0) x_begin = 0;
	if (y_begin < 0) y_begin = 0;
	x_end = x0 + Wx;
	y_end = y0 + Hy;
	if (x_end >= W) x_end = W - 1;//超出范围的话就用画的框的边界来赋值粒子的区域
	if (y_end >= H) y_end = H - 1;
	a2 = Wx * Wx + Hy * Hy;                /* 计算半径平方a^2 */
	f = 0.0;                         /* 归一化系数 */
	for (y = y_begin; y <= y_end; y++)
		for (x = x_begin; x <= x_end; x++)
		{
			r = image[(y*W + x) * 3] >> R_SHIFT;   /* 计算直方图 */
			g = image[(y*W + x) * 3 + 1] >> G_SHIFT; /*移位位数根据R、G、B条数 */
			b = image[(y*W + x) * 3 + 2] >> B_SHIFT;
			index2 = himg[y*W + x] >> LBP_SHIFT;
			index = r * G_BIN * B_BIN + g * B_BIN + b;//把当前rgb换成一个索引
			r2 = (float)(((y - y0)*(y - y0) + (x - x0)*(x - x0))*1.0 / a2); /* 计算半径平方r^2 */
			k = 1 - r2;   /* k(r) = 1-r^2, |r| < 1; 其他值 k(r) = 0 ，影响力*/
			f = f + k;
			ColorHist[index] = ColorHist[index] + k;  /* 计算核密度加权彩色直方图 */
			TextureHist[index2] = TextureHist[index2] + 1;
			count++;
		}
	for (i = 0; i < bins; i++)     /* 归一化直方图 */
	{
		ColorHist[i] = ColorHist[i] / f;
	}
	for (i = 0; i < Lsize; i++)
	{
		TextureHist[i] = TextureHist[i] / count;
	}
	return;
}

/*
计算Bhattacharyya系数
输入参数：
float * p, * q：       两个彩色直方图密度估计
int bbins：            直方图条数
返回值：
Bhattacharyya系数
*/
float CalcuBhattacharyya(float * p, float * q, int bbins)
{
	int i;
	float rho;

	rho = 0.0;
	for (i = 0; i < bbins; i++)
		rho = (float)(rho + sqrt(p[i] * q[i]));

	return(rho);
}

//# define RECIP_SIGMA  3.98942280401    1/(sqrt(2*pi)*sigma), 这里sigma = 0.1 
# define SIGMA2       0.02           /* 2*sigma^2, 这里sigma = 0.1 */
# define ALPHA        0.5
# define BETA         0.5
# define sigmac       -0.02
# define sigmag       -0.02
/*根据巴氏系数计算各个权值*/
float CalcuWeightedPi(float rho1, float rho2)
{
	float pi_n, d_color2;
	float d_grad2 = 1 - rho2;
	d_color2 = 1 - rho1;
	//float D2 = ALPHA * d_color2 + BETA * d_grad2;
	//pi_n = (float)(exp(-D2 / SIGMA2));
	float a = d_color2 * ALPHA / sigmac;
	float b = d_grad2 * BETA / sigmag;
	pi_n = (float)(exp(a + b));
	return(pi_n);
}

/*
获得一个[0,1]之间的随机数
*/
float rand0_1()
{
	//return(ran0(&ran_seed));
	return(rand() / float(RAND_MAX));
}

/*
获得一个x～N(u,sigma)Gaussian分布的随机数
*/
float randGaussian(float u, float sigma)
{
	float x1, x2, v1, v2;
	float s = 100.0;
	float y;
	/*
	使用筛选法产生正态分布N(0,1)的随机数(Box-Mulles方法)
	1. 产生[0,1]上均匀随机变量X1,X2
	2. 计算V1=2*X1-1,V2=2*X2-1,s=V1^2+V2^2
	3. 若s<=1,转向步骤4，否则转1
	4. 计算A=(-2ln(s)/s)^(1/2),y1=V1*A, y2=V2*A
	y1,y2为N(0,1)随机变量
	*/
	while (s > 1.0)
	{
		x1 = rand0_1();
		x2 = rand0_1();
		v1 = 2 * x1 - 1;
		v2 = 2 * x2 - 1;
		s = v1 * v1 + v2 * v2;
	}
	y = (float)(sqrt(-2.0 * log(s) / s) * v1);
	/*
	根据公式:z = sigma * y + u
	将y变量转换成N(u,sigma)分布
	*/
	return(sigma * y + u);
}

/*
初始化系统
int x0, y0：         初始给定的图像目标区域坐标
int Wx, Hy：         目标的半宽高
unsigned char * img：图像数据,RGB形式
int W, H：           图像宽高
*/
int Initialize(int x0, int y0, int Wx, int Hy,
	unsigned char * img, int W, int H)
{
	int i, j;
	srand((unsigned int)(time(NULL)));
	states = new SPACESTATE[NParticle]; /* 申请状态数组的空间 */
	weights = new float[NParticle];     /* 申请粒子权重数组的空间 */
	nbin = R_BIN * G_BIN * B_BIN; /* 确定直方图条数 */
	ModelHist = new float[nbin]; /* 申请直方图内存 */
	if (ModelHist == NULL) return(-1);
	LBPHist = new float[Lsize];
	if (LBPHist == NULL)return (-1);

	/* 计算目标模板直方图 */
	CalcuColorHistogram(x0, y0, Wx, Hy, img, W, H, ModelHist, LBPHist, nbin);
	/* 初始化粒子状态(以(x0,y0,1,1,Wx,Hy,0.1)为中心呈N(0,0.4)正态分布) */
	states[0].xt = x0;
	states[0].yt = y0;
	states[0].v_xt = (float)0.0; // 1.0
	states[0].v_yt = (float)0.0; // 1.0
	states[0].Hxt = Wx;
	states[0].Hyt = Hy;
	states[0].at_dot = (float)0.0; // 0.1
	weights[0] = (float)(1.0 / NParticle); /* 0.9; */
	float rn[7];
	for (i = 1; i < NParticle; i++)
	{
		for (j = 0; j < 7; j++) rn[j] = randGaussian(0, (float)0.6); /* 产生7个随机高斯分布的数 */
		states[i].xt = (int)(states[0].xt + rn[0] * Wx);
		states[i].yt = (int)(states[0].yt + rn[1] * Hy);
		states[i].v_xt = (float)(states[0].v_xt + rn[2] * VELOCITY_DISTURB);
		states[i].v_yt = (float)(states[0].v_yt + rn[3] * VELOCITY_DISTURB);
		states[i].Hxt = (int)(states[0].Hxt + rn[4] * SCALE_DISTURB);
		states[i].Hyt = (int)(states[0].Hyt + rn[5] * SCALE_DISTURB);
		states[i].at_dot = (float)(states[0].at_dot + rn[6] * SCALE_CHANGE_D);
		/* 权重统一为1/N，让每个粒子有相等的机会 */
		weights[i] = (float)(1.0 / NParticle);
	}
	return(1);
}

/*
计算归一化累计概率c'_i
输入参数：
float * weight：    为一个有N个权重（概率）的数组
int N：             数组元素个数
输出参数：
float * cumulateWeight： 为一个有N+1个累计权重的数组，
cumulateWeight[0] = 0;
*/
void NormalizeCumulatedWeight(float * weight, float * cumulateWeight, int N)
{
	int i;
	for (i = 0; i < N + 1; i++)
		cumulateWeight[i] = 0;
	for (i = 0; i < N; i++)
		cumulateWeight[i + 1] = cumulateWeight[i] + weight[i];
	for (i = 0; i < N + 1; i++)
		cumulateWeight[i] = cumulateWeight[i] / cumulateWeight[N];

	return;
}

/*
折半查找，在数组NCumuWeight[N]中寻找一个最小的j，使得
NCumuWeight[j] <=v
float v：              一个给定的随机数
float * NCumuWeight：  权重数组
int N：                数组维数
返回值：
数组下标序号
*/
int BinearySearch(float v, float * NCumuWeight, int N)
{
	int l, r, m;

	l = 0; 	r = N - 1;   /* extreme left and extreme right components' indexes */
	while (r >= l)
	{
		m = (l + r) / 2;
		if (v >= NCumuWeight[m] && v < NCumuWeight[m + 1]) return(m);
		if (v < NCumuWeight[m]) r = m - 1;
		else l = m + 1;
	}
	return(0);
}

/*
重新进行重要性采样
输入参数：
float * c：          对应样本权重数组pi(n)
int N：              权重数组、重采样索引数组元素个数
输出参数：
int * ResampleIndex：重采样索引数组
*/
void ImportanceSampling(float * c, int * ResampleIndex, int N)
{
	float rnum, *cumulateWeight;
	int i, j;

	cumulateWeight = new float[N + 1]; /* 申请累计权重数组内存，大小为N+1 */
	NormalizeCumulatedWeight(c, cumulateWeight, N); /* 计算累计权重 */
	for (i = 0; i < N; i++)
	{
		rnum = rand0_1();       /* 随机产生一个[0,1]间均匀分布的数 */
		j = BinearySearch(rnum, cumulateWeight, N + 1); /* 搜索<=rnum的最小索引j */
		if (j == N) j--;
		ResampleIndex[i] = j;	/* 放入重采样索引数组 */
	}

	delete[] cumulateWeight;

	return;
}

/*
样本选择，从N个输入样本中根据权重重新挑选出N个
输入参数：
SPACESTATE * state：     原始样本集合（共N个）
float * weight：         N个原始样本对应的权重
int N：                  样本个数
输出参数：
SPACESTATE * state：     更新过的样本集
*/
void ReSelect(SPACESTATE * state, float * weight, int N)
{
	SPACESTATE * tmpState;
	int i, *rsIdx;//统计的随机数所掉区间的索引

	tmpState = new SPACESTATE[N];
	rsIdx = new int[N];

	ImportanceSampling(weight, rsIdx, N); /* 根据权重重新采样 */
	for (i = 0; i < N; i++)
		tmpState[i] = state[rsIdx[i]];//temState为临时变量,其中state[i]用state[rsIdx[i]]来代替
	for (i = 0; i < N; i++)
		state[i] = tmpState[i];

	delete[] tmpState;
	delete[] rsIdx;
	return;
}

/*
传播：根据系统状态方程求取状态预测量
状态方程为： S(t) = A S(t-1) + W(t-1)
W(t-1)为高斯噪声
输入参数：
SPACESTATE * state：      待求的状态量数组
int N：                   待求状态个数
输出参数：
SPACESTATE * state：      更新后的预测状态量数组
*/
void Propagate(SPACESTATE * state, int N)
{
	int i;
	int j;
	float rn[7];
	/* 对每一个状态向量state[i](共N个)进行更新 */
	for (i = 0; i < N; i++)  /* 加入均值为0的随机高斯噪声 */
	{
		for (j = 0; j < 7; j++) rn[j] = randGaussian(0, (float)0.6); /* 产生7个随机高斯分布的数 */
		state[i].xt = (int)(state[i].xt + state[i].v_xt * DELTA_T + rn[0] * state[i].Hxt + 0.5);
		state[i].yt = (int)(state[i].yt + state[i].v_yt * DELTA_T + rn[1] * state[i].Hyt + 0.5);
		state[i].v_xt = (float)(state[i].v_xt + rn[2] * VELOCITY_DISTURB);
		state[i].v_yt = (float)(state[i].v_yt + rn[3] * VELOCITY_DISTURB);
		state[i].Hxt = (int)(state[i].Hxt + state[i].Hxt*state[i].at_dot + rn[4] * SCALE_DISTURB + 0.5);
		state[i].Hyt = (int)(state[i].Hyt + state[i].Hyt*state[i].at_dot + rn[5] * SCALE_DISTURB + 0.5);
		state[i].at_dot = (float)(state[i].at_dot + rn[6] * SCALE_CHANGE_D);
		//circle(frame,Point(state[i].xt,state[i].yt),3, Scalar(0,255,0),-1);  //在每一帧上显示粒子位置
	}
	return;
}

/*
观测，根据状态集合St中的每一个采样，观测直方图，然后
更新估计量，获得新的权重概率
输入参数：
SPACESTATE * state：      状态量数组
int N：                   状态量数组维数
unsigned char * image：   图像数据，按从左至右，从上至下的顺序扫描，
颜色排列次序：RGB, RGB, ...
int W, H：                图像的宽和高
float * ObjectHist：      目标直方图
int hbins：               目标直方图条数
输出参数：
float * weight：          更新后的权重
*/
void Observe(SPACESTATE * state, float * weight, int N,
	unsigned char * image, int W, int H,
	float * ObjectHist, float * ObjectLBP)
{
	int i;
	float * ColorHist;
	float rho1, rho2;
	float * lbp;

	ColorHist = new float[nbin];
	lbp = new float[Lsize];

	for (i = 0; i < N; i++)
	{
		/* (1) 计算彩色直方图分布 */
		CalcuColorHistogram(state[i].xt, state[i].yt, state[i].Hxt, state[i].Hyt,
			image, W, H, ColorHist, lbp, nbin);
		/* (2) Bhattacharyya系数 */
		rho1 = CalcuBhattacharyya(ColorHist, ObjectHist, nbin);
		rho2 = CalcuBhattacharyya(lbp, ObjectLBP, Lsize);
		/* (3) 根据计算得的Bhattacharyya系数计算各个权重值 */
		weight[i] = CalcuWeightedPi(rho1, rho2);
	}

	delete[] ColorHist;
	delete[] lbp;

	return;
}

/*
估计，根据权重，估计一个状态量作为跟踪输出
输入参数：
SPACESTATE * state：      状态量数组
float * weight：          对应权重
int N：                   状态量数组维数
输出参数：
SPACESTATE * EstState：   估计出的状态量
*/
void Estimation(SPACESTATE * state, float * weight, int N,
	SPACESTATE & EstState)
{
	int i;
	float at_dot, Hxt, Hyt, v_xt, v_yt, xt, yt;
	float weight_sum;

	at_dot = 0;
	Hxt = 0; 	Hyt = 0;
	v_xt = 0;	v_yt = 0;
	xt = 0;  	yt = 0;
	weight_sum = 0;
	for (i = 0; i < N; i++) /* 求和 */
	{
		at_dot += state[i].at_dot * weight[i];
		Hxt += state[i].Hxt * weight[i];
		Hyt += state[i].Hyt * weight[i];
		v_xt += state[i].v_xt * weight[i];
		v_yt += state[i].v_yt * weight[i];
		xt += state[i].xt * weight[i];
		yt += state[i].yt * weight[i];
		weight_sum += weight[i];
	}
	//cout << weight_sum << endl;
	/* 求平均 */
	if (weight_sum <= 0) weight_sum = 1;   // 防止被0除，一般不会发生
	/*EstState.at_dot = at_dot / weight_sum;
	EstState.Hxt = (int)(Hxt + 0.5);
	EstState.Hyt = (int)(Hyt + 0.5);
	EstState.v_xt = v_xt;
	EstState.v_yt = v_yt;
	EstState.xt = (int)(xt + 0.5);
	EstState.yt = (int)(yt + 0.5);*/
	EstState.at_dot = at_dot / weight_sum;
	EstState.Hxt = (int)(Hxt / weight_sum);
	EstState.Hyt = (int)(Hyt / weight_sum);
	EstState.v_xt = v_xt / weight_sum;
	EstState.v_yt = v_yt / weight_sum;
	EstState.xt = (int)(xt / weight_sum);
	EstState.yt = (int)(yt / weight_sum);

	return;
}


/*
模型更新
输入参数：
SPACESTATE EstState：   状态量的估计值
float * TargetHist：    目标直方图
int bins：              直方图条数
float PiT：             阈值（权重阈值）
unsigned char * img：   图像数据，RGB形式
int W, H：              图像宽高
输出：
float * TargetHist：    更新的目标直方图
*/
# define ALPHA_COEFFICIENT      0.2     /* 目标模型更新权重取0.1-0.3 */

void ModelUpdate(SPACESTATE EstState, float * TargetHist, float * TargetLBP, float PiT,
	unsigned char * img, int W, int H)
{
	float * EstHist, Bha1, Bha2, Pi_E;
	int i;
	float * lbp_;

	EstHist = new float[nbin];
	lbp_ = new float[Lsize];
	/* (1)在估计值处计算目标直方图 */
	CalcuColorHistogram(EstState.xt, EstState.yt, EstState.Hxt,
		EstState.Hyt, img, W, H, EstHist, lbp_, nbin);
	/* (2)计算Bhattacharyya系数 */
	Bha1 = CalcuBhattacharyya(EstHist, TargetHist, nbin);
	Bha2 = CalcuBhattacharyya(lbp_, TargetLBP, Lsize);
	/* (3)计算概率权重 */
	Pi_E = CalcuWeightedPi(Bha1, Bha2);
	//float d1 = sqrt(1 - Bha1);
	//float d2 = sqrt(1 - Bha2);
	if (Pi_E > PiT)
	{
		for (i = 0; i < nbin; i++)
		{
			TargetHist[i] = (float)((1.0 - ALPHA_COEFFICIENT) * TargetHist[i]
				+ ALPHA_COEFFICIENT * EstHist[i]);

		}
		for (i = 0; i < Lsize; i++)
		{
			TargetLBP[i] = (float)((1.0 - ALPHA_COEFFICIENT) * TargetLBP[i]
				+ ALPHA_COEFFICIENT * lbp_[i]);
		}
	}
	/*if (d1 < 0.1)
	{
		for (i = 0; i < bins; i++)
		{
			TargetHist[i] = (float)((1.0 - ALPHA_COEFFICIENT) * TargetHist[i]
				+ ALPHA_COEFFICIENT * EstHist[i]);

		}
	}
	if (d2 < 0.2)
	{
		for (i = 0; i < Lsize; i++)
		{
			TargetLBP[i] = (float)((1.0 - ALPHA_COEFFICIENT) * TargetLBP[i]
				+ ALPHA_COEFFICIENT * lbp_[i]);
		}
	}*/
	delete[] EstHist;
	delete[] lbp_;
}

/*
系统清除
*/
void ClearAll()
{
	if (ModelHist != NULL) delete[] ModelHist;
	if (states != NULL) delete[] states;
	if (weights != NULL) delete[] weights;
	if (img != NULL) delete[] img;
	if (himg != NULL) delete[] himg;
	if (LBPHist != NULL) delete[] LBPHist;
	return;
}

/*粒子滤波跟踪*/
int ColorParticleTracking(unsigned char * image, int &W, int &H, 
	int & xc, int & yc,	int & Wx_h, int & Hy_h, float & max_weight)
{
	SPACESTATE EState;
	int i;
	/*选择：选择样本，并进行重采样*/
	ReSelect(states, weights, NParticle);
	/*传播：采样状态方程，对状态变量进行预测*/
	Propagate(states, NParticle);
	/*观测：对状态量进行更新，更新权值*/
	Observe(states, weights, NParticle, image, W, H, ModelHist, LBPHist );
	/*估计：对状态量进行估计，提取位置量*/
	Estimation(states, weights, NParticle, EState);
	xc = EState.xt;
	yc = EState.yt;
	Wx_h = EState.Hxt;
	Hy_h = EState.Hyt;
	/*模型更新*/
	ModelUpdate(EState, ModelHist, LBPHist, Pi_Thres, image, W, H);
	/*计算最大权重值*/
	max_weight = weights[0];
	for (i = 1; i < NParticle; i++)
		max_weight = max_weight < weights[i] ? weights[i] : max_weight;
	/*进行合法性检验，不合法返回-1*/
	if (xc < 0 || yc < 0 || xc >= W || yc >= H || Wx_h <= 0 || Hy_h <= 0) {
		return(-1);
	}
	else
		return(1);
}

//彩色图像转灰度图
void ToGray(Mat& src, Mat& dst)
{
	int i, j;
	for (i = 0; i < src.rows; i++)
	{
		for (j = 0; j < src.cols; j++)
		{
			dst.at<uchar>(i, j) = saturate_cast<uchar>((src.at<Vec3b>(i, j)[0] * 30 + src.at<Vec3b>(i, j)[1] * 150 +
				src.at<Vec3b>(i, j)[2] * 76) >> 8);
		}
	}
}

void rgb2gray(unsigned char *src, unsigned char *dest, int width, int height)
{
	int r, g, b;
	for (int i = 0; i < width*height; ++i)
	{
		b = *src++; // load 
		g = *src++; // load 
		r = *src++; // load 
		// build weighted average:
		*dest++ = (r * 76 + g * 150 + b * 30) >> 8;
	}
}

//把iplimage转到img数组中
void IplToImge(Mat src, int w, int h)
{
	int i, j;	          //转成正向图像
	for (i = 0; i < h; i++)   //行
		for (j = 0; j < w; j++)  //列
		{
			img[(i*w + j) * 3] = src.at<Vec3b>(i, j)[2];//R
			img[(i*w + j) * 3 + 1] = src.at<Vec3b>(i, j)[1];//G
			img[(i*w + j) * 3 + 2] = src.at<Vec3b>(i, j)[0];//B
			himg[i*w + j] = texture.at<uchar>(i, j);
		}
}



//通过鼠标划取一个初始矩形框
void on_MouseHandler(int event, int x, int y, int flags, void* param)
{
	Mat pFront1;
	int centerX, centerY;
	pFront1 = frame.clone();

	switch (event) {
	case EVENT_MOUSEMOVE:
	{
		if (drawing_box)
		{
			initbox.width = x - initbox.x;
			initbox.height = y - initbox.y;
		}
	}
	break;
	case EVENT_LBUTTONDOWN:
	{
		drawing_box = true;
		initbox = Rect(x, y, 0, 0);
	}
	break;
	case EVENT_LBUTTONUP:
	{
		drawing_box = false;
		if (initbox.width < 0)
		{
			initbox.x += initbox.width;
			initbox.width *= -1;
		}
		if (initbox.height < 0)
		{
			initbox.y += initbox.height;
			initbox.height *= -1;
		}
		xMin = initbox.x;
		yMin = initbox.y;
		width = initbox.width;
		height = initbox.height;
		rectangle(frame, initbox.tl(), initbox.br(), Scalar(0, 0, 250), 2, 8, 0);
		imshow("tracking", frame);
		return;
	}
	//gotbb=true;
	break;
	}

}

int bSelectObject = 0;
Point origin;
Rect selection;//一个矩形对象
int WidIn, HeiIn;//输入的半宽与半高
bool track = false;//是否跟踪

void mouseHandler(int event, int x, int y, int flags, void* param)
{
	int centerx, centery;

	if (bSelectObject)//如果画了方框
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = abs(x - origin.x);
		selection.height = abs(y - origin.y);
	}

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN://按下左键时
		origin = Point(x, y);//声明一个点的位置 origin即按下鼠标的起始点的位置
		selection = Rect(x, y, 0, 0);//框的构造函数 矩形类
		bSelectObject = 1;
		pause = true;
		break;
	case CV_EVENT_LBUTTONUP://释放左键时
		bSelectObject = 0;
		centerx = selection.x + selection.width / 2;
		centery = selection.y + selection.height / 2;
		WidIn = selection.width / 2;
		HeiIn = selection.height / 2;
		Initialize(centerx, centery, WidIn, HeiIn, img, Wid, Hei);
		track = true;
		pause = false;
		break;
	}
}



int main() {

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;
	uchar key = false;//用来设置暂停、停止键

	int rho_v;//表示合法值
	float max_weight;//最大权值
	int correct;
	int star = 0;

	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	Rect result;

	/*开始获取目标框位置*/
		// Read groundtruth for the 1st frame
	/*  	ifstream groundtruthFile;
		string groundtruth = "D:/图像学习/database/David/groundtruth_rect.txt";
		groundtruthFile.open(groundtruth);
		string firstLine;
		getline(groundtruthFile, firstLine);
		groundtruthFile.close();
		//cout<<firstLine<<endl;
		istringstream ss(firstLine);

		// Read groundtruth like a dumb
		float xMin, yMin, width, height;//, x3, y3, x4, y4;
		char ch;
		ss >> xMin;
		ss >> ch;
		ss >> yMin;
		ss >> ch;
		ss >> width;
		ss >> ch;
		ss >> height;
	*/
	/*结束获取目标框位置*/

	string first_file = "C:/Users/18016/Desktop/ObjectTracking/learnopencv-master/tracking/videos/01.png";
	VideoCapture sequence(first_file);
	VideoCapture seq;
	//seq.open("C:/Users/18016/Desktop/ObjectTracking/learnopencv-master/tracking/videos/chaplin.mp4");
	//seq.open(0);
	//seq.open("C:/Users/18016/Desktop/Benchmark_OTB/videos/crew_cif.y4m");
	seq.open("C:/Users/18016/Desktop/Benchmark_OTB/videos/3.mpeg");
	if (!sequence.isOpened())
	{
		cout << "Failed to open the image sequence!\n" << endl;
		return 1;
	}
	namedWindow("tracking", 1);
	setMouseCallback("tracking", on_MouseHandler, 0);
	//setMouseCallback("tracking", mouseHandler, 0);
	// Write Results
	ofstream resultsFile;
	string resultsPath = "D:\\output.txt";
	resultsFile.open(resultsPath);

	// Frame counter
	int nFrames = 0,lostcount = 0;

	for (;;) {
		// Read each frame from the list
		//sequence >> frame;
		seq >> frame;          //Mat frame; 帧
		if (frame.empty())
		{
			cout << "End of Sequence" << endl;
			break;
		}
		//if (frame.rows >= 480 || frame.cols >= 640)
		//	resize(frame, frame, Size(640, 480));
		imshow("tracking", frame);
		if (!star)//初始化
		{
			Wid = frame.cols;
			Hei = frame.rows;
			img = new unsigned char[Wid * Hei * 3];
			himg = new unsigned char[Wid*Hei];
			gray_image.create(Hei, Wid, CV_8UC1);
			star = 1;
		}
		//Hei = frame.rows;  //行
		//Wid = frame.cols;  //列
		//img = new unsigned char[Wid*Hei * 3];   //unsigned char *img;  便于计算...
		//himg = new unsigned char[Wid*Hei];
		//gray_image.create(Hei, Wid, CV_8UC1);
		//rgb2gray(frame.data, gray_image.data, Wid, Hei);
		ToGray(frame, gray_image);
		texture = LBP(gray_image);

		IplToImge(frame, Wid, Hei);
		//imshow("gray", texture);
		double time0 = static_cast<double>(getTickCount());
		// First frame, give the groundtruth to the tracker
		if (nFrames == 0) {
			while (pause)
				if (waitKey(0) == 'p')
					pause = false;
			tracker.init(Rect(xMin, yMin, width, height), frame);
			rectangle(frame, Point(xMin, yMin), Point(xMin + width, yMin + height), Scalar(0, 255, 255), 1, 8);
			Initialize(xMin + width / 2, yMin + height / 2, width / 2, height / 2, img, Wid, Hei);
			//resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
		}
		// Update 更新
		else {
			result = tracker.update(frame);
		
			rho_v = ColorParticleTracking(img, Wid, Hei, xout, yout, WidOut, HeiOut, max_weight);
			if (rho_v == 1 && max_weight > 0.0001) {
				//rectangle(frame, Point(xout - WidOut, yout - HeiOut),
				//	Point(xout + WidOut, yout + HeiOut), cvScalar(255, 0, 0), 1, 8);//蓝色
				//xin = xout;
				//yin = yout;         //上一帧的输出作为这一帧的输入
			//tracker.init(Rect(xout - WidOut, yout - HeiOut, WidOut * 2, HeiOut * 2), frame);
			//result.x = xout - result.width / 2;
			//result.y = yout - result.height / 2;
			//tracker.init(result, frame);
			//tracker.updateTrackerPosition(result);
				if (tracker.peak_value < 0.4) {
					if (lostcount > 10) {
						lostcount = 0;
						cout << "target lost" << endl;
						result.x = xout - result.width / 2;
						result.y = yout - result.height / 2;
						tracker.updateTrackerPosition(result);
					}
					else
						lostcount++;
				}
				else if (tracker.peak_value >= 0.55) {
					lostcount = 0;

					Initialize(result.x + result.width / 2, result.y + result.height / 2, result.width / 2, result.height / 2, img, Wid, Hei);
				}
			}
			else {
				cout << "pf lost." << endl;
			}
			
			/*else {
				Rect rect = result & Rect(Point(xout - WidOut, yout - HeiOut),
					Point(xout + WidOut, yout + HeiOut));
				lostcount = 0;
				//float x1 = result.x + result.width / 2;
				//float y1 = result.y + result.height / 2;
				//float r2 = sqrt((x1 - xout)*(x1 - xout) + (y1 - yout)*(y1 - yout));
				if(max_weight<0.1)
					cout << "true lost" << endl;
				else if (max_weight>=0.1&&(rect.area()) <= (result.area() / 2))
				{
					result.x = xout - result.width / 2;
					result.y = yout - result.height / 2;
					tracker.updateTrackerPosition(result);
				}
					
			}*/
			rectangle(frame, Point(result.x, result.y),
				Point(result.x + result.width, result.y + result.height),
				Scalar(0, 0, 255), 1, 8);//红色
		/*for ( int i = 0; i < NParticle; i++ )
		{
			states[i].xt = result.x+result.width/2;
			states[i].yt = result.y+result.height/2;
			states[i].Hxt = result.width/2;
			states[i].Hyt = result.height/2;
		}*/
		//rho_v = ColorParticleTracking( img, Wid, Hei, WidOut, HeiOut, max_weight);

		//tracker._roi.width=2*WidOut;
		//tracker._roi.height=2*HeiOut;
		//resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
		}
		cout << nFrames << endl;
		cout << tracker.peak_value << endl;
		cout << max_weight << endl;
	//resultsFile <<nFrames<<endl<< tracker.peak_value << endl<<max_weight<<endl;
		
		nFrames++;
		time0 = getTickFrequency() / ((double)getTickCount() - time0);
		string fps = to_string(time0);
		//putText(frame,"FPS: " + fps, Point(0, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 50, 170), 2);
		//cout<< time0 << endl;
		//if (!SILENT){
		imshow("tracking", frame);
		key = waitKey(1);
		if (key == 'p') pause = true;
		if (key == 'q') return 0;
		while (pause)
			if (waitKey(0) == 'p')
				pause = false;
		//waitKey(1);
		//}
		if (pause) {
			cvWaitKey(0);
		}
		else
			cvWaitKey(100);
		//waitKey(0);
	}
	ClearAll();
	resultsFile.close();
	//listFile.close();
}

