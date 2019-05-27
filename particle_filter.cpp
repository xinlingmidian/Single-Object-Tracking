#include <opencv2/opencv.hpp>
#include <time.h>
#include <iostream>
#include <fstream>
#include"lbp.h"
#include <numeric>

using namespace std;
using namespace cv;


//#define B(image,x,y) ((uchar*)(image->imageData + image->widthStep*(y)))[(x)*3]		//B
//#define G(image,x,y) ((uchar*)(image->imageData + image->widthStep*(y)))[(x)*3+1]	//G
//#define R(image,x,y) ((uchar*)(image->imageData + image->widthStep*(y)))[(x)*3+2]	//R

# define R_BIN      8  /* 红色分量的直方图条数 */
# define G_BIN      8  /* 绿色分量的直方图条数 */
# define B_BIN      8  /* 蓝色分量的直方图条数 */
# define H_BIN      6  /* 色度分量的直方图条数 */
# define S_BIN      8  /* 饱和度分量的直方图条数 */

# define R_SHIFT    5  /* 与上述直方图条数对应 */
# define G_SHIFT    5  /* 的R、G、B分量左移位数 */
# define B_SHIFT    5  /* log2( 256/8 )为移动位数 */
# define H_SHIFT    30
# define S_SHIFT    5 
# define LBP_SHIFT  5
typedef struct __SpaceState {  /* 状态空间变量 */
	int xt;               /* x坐标位置 */
	int yt;               /* y坐标位置 */
	float v_xt;           /* x方向运动速度 */
	float v_yt;           /* y方向运动速度 */
	int Hxt;              /* x方向半窗宽 */
	int Hyt;              /* y方向半窗宽 */
	float at_dot;         /* 尺度变换速度，粒子所代表的那一片区域的尺度变化速度 */
} SPACESTATE;

bool pause = false;//是否暂停
bool track = false;//是否跟踪
//IplImage *curframe=NULL;
//IplImage *pTrackImg =NULL;
unsigned char * img;//把iplimg改到char*  便于计算
unsigned char * himg;
int xin, yin;//跟踪时输入的中心点
int xout, yout;//跟踪时得到的输出中心点
int Wid, Hei;//图像的大小
int WidIn, HeiIn;//输入的半宽与半高
int WidOut, HeiOut;//输出的半宽与半高

float DELTA_T = (float)0.05;    /* 帧频，可以为30，25，15，10等0.05*/
float VELOCITY_DISTURB = 40.0;  /* 速度扰动幅值40.0*/
float SCALE_DISTURB = 0.0;      /* 窗宽高扰动幅度0.0*/
float SCALE_CHANGE_D = (float)0.001;   /* 尺度变换速度扰动幅度0.001*/

int NParticle = 100;       /* 粒子个数100 */
float * ModelHist = NULL; /* 模型直方图 */
float * LBPHist = NULL;   /* 颜色直方图 */
SPACESTATE * states = NULL;  /* 状态数组 */
float * weights = NULL;   /* 每个粒子的权重 */
int nbin;                 /* 直方图条数 */
float Pi_Thres = (float)0.90; /* 权重阈值0.90*/

int bSelectObject = 0;
Point origin;
Rect selection;//一个矩形对象
Mat frame, image, hsv, gray_image, hue, saturation, texture;
Mat channels[3], hist_hue, hist_st, hist_lbp;
int Lsize = 8; //8
float h_ranges[] = { 0,180 };
float s_ranges[] = { 0,256 };
float l_ranges[] = { 0,256 };
const float* H_ranges = h_ranges;
const float* S_ranges = s_ranges;
const float* L_ranges = l_ranges;

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
	int x, y, i, index,index2;
	int r, g, b;
	float k, r2, f;
	int a2,count = 0;

	for (i = 0; i < bins; i++)     /* 直方图各个值赋0 有问题*/
		ColorHist[i] = 0.0;
	for (i = 0; i < Lsize; i++)
		TextureHist[i] = 0.0;
	/* 考虑特殊情况：x0, y0在图像外面，或者，Wx<=0, Hy<=0 */
	/* 此时强制令彩色直方图为0 */
	if ((x0 < 0) || (x0 >= W) || (y0 < 0) || (y0 >= H)
		|| (Wx <= 0) || (Hy <= 0)) return;

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
			TextureHist[index2] = TextureHist[index2] + k;
			//count++;
		}
	for (i = 0; i < bins; i++)     /* 归一化直方图 */
	{
		ColorHist[i] = ColorHist[i] / f;
	}
	for (i = 0; i < Lsize; i++)
	{
		TextureHist[i] = TextureHist[i] / f;
	}
	//Mat roi_lbp(texture, Rect(Point(x_begin, y_begin), Point(x_end, y_end)));
	//Mat roi_lbp = Mat::zeros(hist_lbp.rows, hist_lbp.cols, hist_lbp.type());
	//calcHist(&roi_lbp, 1, 0, noArray(), lbp, 1, &Lsize, &L_ranges, true);
	//vector<float>ll(8);
	//calcHist(&roi_lbp, 1, 0, noArray(), ll, 1, &Lsize, &L_ranges, true);
	//Mat array_lbp = lbp.getMat();
	//float sumll = accumulate(ll.begin(),ll.end(),0);
	//for (i = 0; i < Lsize; i++)
	//{
	//	ll[i] = ll[i] / sumll;
	//}
	//normalize(lbp, lbp, 0, 255, NORM_MINMAX);
	return;
}

/*void CalcuHistogram(int x0, int y0, int Wx, int Hy,
	unsigned char * image, int W, int H,
	float * HS_Hist, int bins)
{
	int x_begin, y_begin;  //指定图像区域的左上角坐标
	int y_end, x_end;
	int x, y, i, index;
	int h, s;
	float k, r2, f;
	int a2;

	for (i = 0; i < bins; i++)     //直方图各个值赋0
		HS_Hist[i] = 0.0;
	if ((x0 < 0) || (x0 >= W) || (y0 < 0) || (y0 >= H)
		|| (Wx <= 0) || (Hy <= 0)) return;

	x_begin = x0 - Wx;               //计算实际高宽和区域起始点
	y_begin = y0 - Hy;
	if (x_begin < 0) x_begin = 0;
	if (y_begin < 0) y_begin = 0;
	x_end = x0 + Wx;
	y_end = y0 + Hy;
	if (x_end >= W) x_end = W - 1;//超出范围的话就用画的框的边界来赋值粒子的区域
	if (y_end >= H) y_end = H - 1;
	a2 = Wx * Wx + Hy * Hy;                //计算半径平方a^2
	f = 0.0;                         //归一化系数
	for (y = y_begin; y <= y_end; y++)
		for (x = x_begin; x <= x_end; x++)
		{
			h = image[(y*W + x) * 2] / H_SHIFT;   //计算直方图
			s = image[(y*W + x) * 2 + 1] >> S_SHIFT; //移位位数根据R、G、B条数
			index = h * H_BIN + s;//把当前hs换成一个索引  5*6+7=37
			r2 = (float)(((y - y0)*(y - y0) + (x - x0)*(x - x0))*1.0 / a2); // 计算半径平方r^2
			k = 1 - r2;   // k(r) = 1-r^2, |r| < 1; 其他值 k(r) = 0 ，影响力
			f = f + k;
			HS_Hist[index] = HS_Hist[index] + k;  // 计算核密度加权彩色直方图
		}
	for (i = 0; i < bins; i++)     // 归一化直方图
		HS_Hist[i] = HS_Hist[i] / f;

	return;
}
*/

/*
计算Bhattacharyya系数
输入参数：
float * p, * q：      两个彩色直方图密度估计
int bins：            直方图条数
返回值：
Bhattacharyya系数
*/
float CalcuBhattacharyya(float * p, float * q, int bins)
{
	int i;
	float rho;

	rho = 0.0;
	for (i = 0; i < bins; i++)
		rho = (float)(rho + sqrt(p[i] * q[i]));
	return(rho);
}


# define SIGMA2       0.02
# define ALPHA        0.7
# define BETA         0.3
# define sigmac       -0.02
# define sigmag       -0.01

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
	return(rand() / float(RAND_MAX));
}


/*
获得一个x - N(u,sigma)Gaussian分布的随机数
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
	根据公式
	z = sigma * y + u
	将y变量转换成N(u,sigma)分布
	*/
	return(sigma * y + u);
}



/*
初始化系统
int x0, y0：        初始给定的图像目标区域坐标
int Wx, Hy：        目标的半宽高
unsigned char * img：图像数据，RGB形式
int W, H：          图像宽高
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
	SPACESTATE * tmpState;//新的放狗的地方
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
		//circle(frame, Point(state[i].xt, state[i].yt), 3, CV_RGB(0, 255, 0), -1);
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
	float * ObjectHist, float * ObjectLBP, int hbins)
{
	int i;
	float * ColorHist;
	float rho1,rho2;
	float * lbp;

	ColorHist = new float[hbins];
	lbp = new float[Lsize];

	for (i = 0; i < N; i++)
	{
		/* (1) 计算彩色直方图分布 */
		CalcuColorHistogram(state[i].xt, state[i].yt, state[i].Hxt, state[i].Hyt,
			image, W, H, ColorHist, lbp, hbins);
		/* (2) Bhattacharyya系数 */
		rho1 = CalcuBhattacharyya(ColorHist, ObjectHist, hbins);
		rho2 = CalcuBhattacharyya(lbp, ObjectLBP, Lsize);
		/* (3) 根据计算得的Bhattacharyya系数计算各个权重值 */
		weight[i] = CalcuWeightedPi(rho1,rho2);
		//cout << "weight " << i <<" "<< weight[i] << endl;
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
	cout << "weight_sum " <<" "<< weight_sum<< endl;
	/* 求平均 */
	if (weight_sum <= 0) weight_sum = 1; /* 防止被0除，一般不会发生 */
	EstState.at_dot = at_dot / weight_sum;
	EstState.Hxt = (int)(Hxt / weight_sum);
	EstState.Hyt = (int)(Hyt / weight_sum);
	EstState.v_xt = v_xt / weight_sum;
	EstState.v_yt = v_yt / weight_sum;
	EstState.xt = (int)(xt / weight_sum);
	EstState.yt = (int)(yt / weight_sum);

	return;
}


/************************************************************
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
************************************************************/
# define ALPHA_COEFFICIENT      0.2     /* 目标模型更新权重*/

void ModelUpdate(SPACESTATE EstState, float * TargetHist, float * TargetLBP, int bins, 
	float PiT, unsigned char * img, int W, int H)
{
	float * EstHist, Bha1, Bha2, Pi_E;
	int i;
	float * lbp_;

	EstHist = new float[bins];
	lbp_ = new float[Lsize];
	/* (1)在估计值处计算目标直方图 */
	CalcuColorHistogram(EstState.xt, EstState.yt, EstState.Hxt,
		EstState.Hyt, img, W, H, EstHist, lbp_, bins);
	/* (2)计算Bhattacharyya系数 */
	Bha1 = CalcuBhattacharyya(EstHist, TargetHist, bins);
	Bha2 = CalcuBhattacharyya(lbp_, TargetLBP, Lsize);
	/* (3)计算概率权重 */
	Pi_E = CalcuWeightedPi(Bha1,Bha2);
	//float d1 = sqrt(1 - Bha1);
	//float d2 = sqrt(1 - Bha2);
	if (Pi_E > PiT)
	{
		for (i = 0; i < bins; i++)
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
	/*if (d1 < 0.3)
	{
		for (i = 0; i < bins; i++)
		{
			TargetHist[i] = (float)((1.0 - ALPHA_COEFFICIENT) * TargetHist[i]
				+ ALPHA_COEFFICIENT * EstHist[i]);

		}
	}
	if (d2 < 0.3)
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


int ColorParticleTracking(unsigned char * image, int W, int H,
	int & xc, int & yc, int & Wx_h, int & Hy_h,
	float & max_weight)
{
	SPACESTATE EState;
	int i;
	/* 选择：选择样本，并进行重采样*/
	ReSelect(states, weights, NParticle);
	/* 传播：采样状态方程，对状态变量进行预测 */
	Propagate(states, NParticle);
	/* 观测：对状态量进行更新 */
	Observe(states, weights, NParticle, image, W, H,
		ModelHist, LBPHist, nbin);
	/* 估计：对状态量进行估计，提取位置量 */
	Estimation(states, weights, NParticle, EState);
	xc = EState.xt;
	yc = EState.yt;
	Wx_h = EState.Hxt;
	Hy_h = EState.Hyt;
	/* 模型更新 */
	ModelUpdate(EState, ModelHist, LBPHist, nbin, Pi_Thres, image, W, H);
	/* 计算最大权重值 */
	max_weight = weights[0];
	for (i = 1; i < NParticle; i++)
		max_weight = max_weight < weights[i] ? weights[i] : max_weight;
	/* 进行合法性检验，不合法返回-1 */
	if (xc < 0 || yc < 0 || xc >= W || yc >= H || Wx_h <= 0 || Hy_h <= 0)
		return(-1);
	else
		return(1);
}

void ToGray(Mat& src,Mat& dst)
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


//把iplimage 转到img 数组中
void IplToImge(Mat src, int w, int h)
{
	int i, j;
	for (i = 0; i < h; i++)
		for (j = 0; j < w; j++)  //列
		{
			img[(i*w + j) * 3] = src.at<Vec3b>(i, j)[2];//R
			img[(i*w + j) * 3 + 1] = src.at<Vec3b>(i, j)[1];//G
			img[(i*w + j) * 3 + 2] = src.at<Vec3b>(i, j)[0];//B
			himg[i*w + j] = texture.at<uchar>(i, j);
		}
}



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
		//roi_lbp = Mat(texture, selection);
		track = true;
		pause = false;
		break;
	}
}



void main(int argc, char *argv[])//参数的个数  参数
{
	//CvCapture * capture = 0;//读取视频的类
	VideoCapture capture;
	//argc=1,表示只有一程序名称。argc=2，表示除了程序名外还有一个参数。
	//如果参数argv[1]是单个数字，则打开摄像头进行捕捉;/否则便是从视频文件中捕获，argv[1]用于指定.avi文件
	/*if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
	capture = cvCaptureFromCAM( argc == 2 ? argv[1][0] - '0' : 0 );//从摄像头读取视频
	else if( argc == 2 )*/
	//seq.open("C:/Users/18016/Desktop/ObjectTracking/learnopencv-master/tracking/videos/chaplin.mp4");
	//capture.open("C:/Users/18016/Desktop/Benchmark_OTB/videos/crew_cif.y4m");
	//capture.open("C:/Users/18016/Desktop/ObjectTracking/learnopencv-master/tracking/videos/chaplin.mp4");
	//capture.open(0);
	capture.open("C:/Users/18016/Desktop/Benchmark_OTB/videos/Meet_WalkTogether1.mpg");
	int rho_v;//表示合法性
	float max_weight;
	cvNamedWindow("video", 1);
	int star = 0;
	int nFrames = 0;
	//const float* ranges[] = { h_ranges,s_ranges };
	//if (capture.isOpened())
	//{
		//ofstream fout;
		//fout.open("E://test.txt");
	while (1)
	{
		//curframe=cvQueryFrame(capture); //抓取一帧
		capture >> frame;
		if (frame.empty())
		{
			cout << "End of Sequence" << endl;
			break;
		}
		imshow("video", frame);
		if (!star)//初始化
		{
			Wid = frame.cols;
			Hei = frame.rows;
			img = new unsigned char[Wid * Hei * 3];
			himg = new unsigned char[Wid*Hei];
			gray_image.create(Hei, Wid, CV_8UC1);
			star = 1;
		}
		//frame.copyTo(image);
		//cvtColor(frame, gray_image, COLOR_BGR2GRAY);
		ToGray(frame, gray_image);
		//medianBlur(gray_image, gray_image, 3);
		//rgb2gray(frame.data, gray_image.data, Wid, Hei);
		texture = LBP(gray_image);
		//pTrackImg = cvCloneImage(frame);//复制图片
		IplToImge(frame, Wid, Hei);//把iplimage转换到img数组中
		imshow("gray", gray_image);
		imshow("lbp", texture);
		//cout << "hahdha" << endl;
		if (track)
		{
			/*while (!pause)
				if (waitKey(0) == 'p')
					pause = true;*/
					//cvtColor(image, hsv, COLOR_BGR2HSV);

					//split(hsv, channels);
					//Mat roi_hue(channels[0], selection);
					//Mat roi_saturation(channels[1], selection);	
					//calcHist(&roi_hue,1,0,noArray(),hist_hue,1,&size,&H_ranges,true);
					//calcHist(&roi_saturation, 1, 0, noArray(), hist_st, 1, &size, &S_ranges, true);
					//calcHist(&roi_lbp, 1, 0, noArray(), hist_lbp, 1, &Lsize, &L_ranges, true);
					//normalize(hist_lbp,hist_lbp,0,255, NORM_MINMAX);
					/* 跟踪一帧 */
			rho_v = ColorParticleTracking(img, Wid, Hei, xout, yout, WidOut, HeiOut, max_weight);//合法性
			/* 画框: 新位置为蓝框 */
			if (rho_v == 1 && max_weight > 0.0001)  /* 判断是否目标丢失 */
			{
				rectangle(frame, cvPoint(xout - WidOut, yout - HeiOut), 
					cvPoint(xout + WidOut, yout + HeiOut), cvScalar(255, 0, 0), 2, 8, 0);
				rectangle(texture, cvPoint(xout - WidOut, yout - HeiOut),
					cvPoint(xout + WidOut, yout + HeiOut), cvScalar(255, 0, 0), 2, 8, 0);
				xin = xout; yin = yout;//上一帧的输出作为这一帧的输入
				WidIn = WidOut; HeiIn = HeiOut;

				//fout<<xout<<"  "<<yout<<endl;

			}
			//else
			//{
			//	cout << "target lost" << endl;
			//}
		}
		cout << nFrames << endl;
		nFrames++;
		imshow("video", frame);
		cvSetMouseCallback("video", mouseHandler, 0);
		if (pause) {
			cvWaitKey(0);
		}
		else
			cvWaitKey(50);
		//cvReleaseImage(frame);
	}
	//cvReleaseCapture(&capture);
	//fout.close();	
//}
//释放图像
//cvDestroyAllWindows();
	ClearAll();
}
