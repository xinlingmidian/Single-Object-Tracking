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

using namespace std;
using namespace cv;

# define R_BIN      16  /* 红色分量的直方图条数 */
# define G_BIN      16  /* 绿色分量的直方图条数 */
# define B_BIN      16  /* 蓝色分量的直方图条数 */ 

# define R_SHIFT    4  /* 与上述直方图条数对应 */
# define G_SHIFT    4  /* 的R、G、B分量左移位数 */
# define B_SHIFT    4  /* log2( 256/16 )为移动位数*/
/*为了产生一个服从正态分布的随机数，采用Park and Miller方法，有参考论文，看不懂*/
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876
const int Num=10;  //帧差的间隔
const int T=40;  //Tf
const int Re=30;  //
const float ai=0.08; //学习率
const int CONTOUR_MIN_AREA=50;
const int CONTOUR_MAX_AREA=10000;

typedef struct __SpaceState{  /* 状态空间变量 */
	int xt;               /* x坐标位置 */
	int yt;               /* y坐标位置 */
	float v_xt;           /* x方向运动速度 */
	float v_yt;           /* y方向运动速度 */
	int Hxt;              /* x方向半窗宽 */
	int Hyt;              /* y方向半窗宽 */
	float at_dot;         /* 尺度变换速度 */
} SPACESTATE;
unsigned char *img;//便于计算...
int Wid,Hei; //图像的大小
//int WidIn,HeiIn; //输入的半宽与半高
int WidOut,HeiOut;//输出的半宽与半高
int xin,yin;   //跟踪时输入的中心点
int xout,yout;//跟踪时得到的输出中心点
int nbin;                 /* 直方图条数 */

const int neighbors = 8;
const int numpat = static_cast<int>(std::pow(2.0, static_cast<double>(neighbors)));
const int m_grid_x = 1,m_grid_y = 1;
int bins = m_grid_x*m_grid_y*numpat;            //纹理直方图

const float DELTA_T = (float)0.05;  /*帧频，可以为30、25、15、10等0.05*/
const int POSITION_DISTURB = 15;      /* 位置扰动幅度   */
const float VELOCITY_DISTURB = 40.0;  /* 速度扰动幅值  40.0 */
const float SCALE_DISTURB = 0.0;      /* 窗宽高扰动幅度 */
const float SCALE_CHANGE_D = (float)0.001;   /* 尺度变换速度扰动幅度 0.001*/
const float Pi_Thres = (float)0.8;  //权重阈值
long ran_seed = 802163120;  // 随机数种子，为全局变量，设置缺省值
int NParticle = 75;   //粒子个数75
float* ModelHist0 = NULL;//模型直方图
SPACESTATE* states = NULL;
float* weights = NULL;   //每个粒子的权重
Mat curframe;
Mat lbpdst;
Mat curgray;
Mat pBackImg;
Mat pFront;
Mat pTrackImg;

bool pause=true;
//bool gotbb=false;
bool drawing_box = false;  //判断是否画矩形框
float xMin, yMin, width, height;/*跟踪框的坐标,宽,高,*/
//bool mark = false;

Mat frame;
Rect initbox;

//设置种子数，一般利用系统时间来进行设置，也可以直接传入一个long型整数
long set_seed(long setvalue)
{
	if(setvalue != 0)
		ran_seed = setvalue;  //如果传入的参数setvalue!=0，设置该数为种子
	else
		ran_seed = time(NULL);   // 否则利用系统时间为种子数
	return(ran_seed);
}


void CalcuColorHistogram( int x0, int y0, int Wx, int Hy, 
						 unsigned char * image, int W, int H,
						 float * ColorHist, int bins )
{
	int x_begin, y_begin;  /* 指定图像区域的左上角坐标 */
	int y_end, x_end;
	int x, y, i, index;
	int r, g, b;
	float k, r2, f;
	int a2;

	for ( i = 0; i < bins; i++ )     /* 直方图各个值赋0 */
		ColorHist[i] = 0.0;
	/* 考虑特殊情况：x0, y0在图像外面，或者，Wx<=0, Hy<=0 */
	/* 此时强制令彩色直方图为0 */
	if ( ( x0 < 0 ) || (x0 >= W) || ( y0 < 0 ) || ( y0 >= H ) 
		|| ( Wx <= 0 ) || ( Hy <= 0 ) ) return;

	x_begin = x0 - Wx;               /* 计算实际高宽和区域起始点 */
	y_begin = y0 - Hy;
	if ( x_begin < 0 ) x_begin = 0;
	if ( y_begin < 0 ) y_begin = 0;
	x_end = x0 + Wx;
	y_end = y0 + Hy;
	if ( x_end >= W ) x_end = W-1;
	if ( y_end >= H ) y_end = H-1;
	a2 = Wx*Wx+Hy*Hy;                /* 计算核函数半径平方a^2 */
	f = 0.0;                         /* 归一化系数 */
	for ( y = y_begin; y <= y_end; y++ )
		for ( x = x_begin; x <= x_end; x++ )
		{
			r = image[(y*W+x)*3] >> R_SHIFT;   /* 计算直方图 */
			g = image[(y*W+x)*3+1] >> G_SHIFT; /*移位位数根据R、G、B条数 */
			b = image[(y*W+x)*3+2] >> B_SHIFT;//相当于将R\G\B三个分量做加和，像素值范围0~255，分八份，则对应二值化向右移5位
			index = r * G_BIN * B_BIN + g * B_BIN + b;//把当前rgb换成一个索引
			//index = r * G_BIN + g ;
			r2 = (float)(((y-y0)*(y-y0)+(x-x0)*(x-x0))*1.0/a2); /* 计算半径平方r^2 */
			//区域内的点到中心点距离的平方最大值是a2，r2反应了区域内各点距离中心的远近
			k = 1 - r2;   /* 核函数k(r) = 1-r^2, |r| < 1; 其他值 k(r) = 0 ；用核函数分配权重*/
			f = f + k;
			ColorHist[index] = ColorHist[index] + k;  /* 计算核密度加权彩色直方图 */
		}
		for ( i = 0; i < bins; i++ )     /* 归一化直方图 */
			ColorHist[i] = ColorHist[i]/f;

		return;
}

float CalcuBhattacharyya( float * p, float * q, int bbins )
{
	int i;
	float rho;

	rho = 0.0;
	for ( i = 0; i < bbins; i++ )
		rho = (float)(rho + sqrt( p[i]*q[i] ));

	return( rho );
}
/*# define RECIP_SIGMA  3.98942280401  / * 1/(sqrt(2*pi)*sigma), 这里sigma = 0.1 * /*/
# define SIGMA2       0.02           /* 2*sigma^2, 这里sigma = 0.1 */
/*根据巴氏系数计算各个权值*/
float CalcuWeightedPi( float rho )
{
	float pi_n, d2;

	d2 = 1 - rho;
	//pi_n = (float)(RECIP_SIGMA * exp( - d2/SIGMA2 ));
	pi_n = (float)(exp( - d2/SIGMA2 ));

	return( pi_n );
}

float ran0(long *idum)
{
	long k;
	float ans;

	/* *idum ^= MASK;*/      /* XORing with MASK allows use of zero and other */
	k=(*idum)/IQ;            /* simple bit patterns for idum.                 */
	*idum=IA*(*idum-k*IQ)-IR*k;  /* Compute idum=(IA*idum) % IM without over- */
	if (*idum < 0) *idum += IM;  /* flows by Schrage’s method.               */
	ans=AM*(*idum);          /* Convert idum to a floating result.            */
	/* *idum ^= MASK;*/      /* Unmask before return.                         */
	return ans;
}

float rand0_1()
{
	return( ran0( &ran_seed ) );
}

/*
获得一个x～N(u,sigma)Gaussian分布的随机数
*/
float randGaussian( float u, float sigma )
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
	while ( s > 1.0 )
	{
		x1 = rand0_1();
		x2 = rand0_1();
		v1 = 2 * x1 - 1;
		v2 = 2 * x2 - 1;
		s = v1*v1 + v2*v2;
	}
	y = (float)(sqrt( -2.0 * log(s)/s ) * v1);
	/*
	根据公式:z = sigma * y + u
	将y变量转换成N(u,sigma)分布
	*/
	return( sigma * y + u );	
}

/*
初始化系统
int x0, y0：         初始给定的图像目标区域坐标
int Wx, Hy：         目标的半宽高
unsigned char * img：图像数据,RGB形式
int W, H：           图像宽高
*/
int Initialize(int x0,int y0,int Wx,int Hy,unsigned char* img,int W,int H)
{
	int i,j;
	float rn[7];
	set_seed(0);  //使用系统时钟作为种子
	states = new SPACESTATE[NParticle];//申请状态数组的空间
	if(states == NULL) 
		return (-2);
	weights = new float [NParticle];
	if(weights == NULL)
		return (-3);
	nbin = R_BIN * G_BIN * B_BIN;  //确定直方图条数
	ModelHist0 = new float [nbin];  //申请直方图内存
	if(ModelHist0 == NULL)
		return(-1);
	/*计算目标模板直方图*/
	CalcuColorHistogram(x0 ,y0 ,Wx ,Hy ,img ,W ,H ,ModelHist0 ,nbin );
	//cout << "aaaaaa" << endl;
	//cout << ModelHist0 << endl;
	//cout << "aaaaaa" << endl;
	/*  初始化粒子状态（以x0，y0，0，,0，Wx，Hy，0）为中心呈N（0,0.6）正态分布? */
	states[0].xt = x0;
	states[0].yt = y0;
	states[0].v_xt = (float)0.0; 
	states[0].v_yt = (float)0.0; 
	states[0].Hxt = Wx;
	states[0].Hyt = Hy;
	states[0].at_dot = (float)0.0; 
	weights[0] = (float)(1.0/NParticle); 
	for ( i = 1; i < NParticle; i++ )
	{
		for ( j = 0; j < 7; j++ ) rn[j] = randGaussian( 0, (float)0.6 ); /* 产生7个随机高斯分布的数 */
		//states[i].xt = x0;
		//states[i].yt = y0;
		states[i].xt = (int)(states[0].xt + rn[0] * Wx);
		states[i].yt = (int)(states[0].yt + rn[1] * Hy);
		states[i].v_xt = (float)( states[0].v_xt + rn[2] * VELOCITY_DISTURB );
		states[i].v_yt = (float)( states[0].v_yt + rn[3] * VELOCITY_DISTURB );
		states[i].Hxt = (int)( states[0].Hxt + rn[4] * SCALE_DISTURB);
		states[i].Hyt = (int)( states[0].Hyt + rn[5] * SCALE_DISTURB );
		states[i].at_dot = (float)( states[0].at_dot + rn[6] * SCALE_CHANGE_D );
		/* 权重统一为1/N，让每个粒子有相等的机会 */
		weights[i] = (float)(1.0/NParticle);
	}
	return( 1 );
}

void NormalizeCumulatedWeight( float * weight, float * cumulateWeight, int N )
{
	int i;

	for ( i = 0; i < N+1; i++ ) 
		cumulateWeight[i] = 0;
	for ( i = 0; i < N; i++ )
		cumulateWeight[i+1] = cumulateWeight[i] + weight[i];
	for ( i = 0; i < N+1; i++ )
		cumulateWeight[i] = cumulateWeight[i]/ cumulateWeight[N];

	return;
}

int BinearySearch( float v, float * NCumuWeight, int N )
{
	int l, r, m;

	l = 0; 	r = N-1;   /* extreme left and extreme right components' indexes */
	while ( r >= l)
	{
		m = (l+r)/2;
		if ( v >= NCumuWeight[m] && v < NCumuWeight[m+1] ) return( m );
		if ( v < NCumuWeight[m] ) r = m - 1;
		else l = m + 1;
	}
	return( 0 );
}

void ImportanceSampling( float * c, int * ResampleIndex, int N )
{
	float rnum, * cumulateWeight;
	int i, j;

	cumulateWeight = new float [N+1]; /* 申请累计权重数组内存，大小为N+1 */
	NormalizeCumulatedWeight( c, cumulateWeight, N ); /* 计算累计权重 */
	for ( i = 0; i < N; i++ )
	{
		rnum = rand0_1();       /* 随机产生一个[0,1]间均匀分布的数 */ 
		j = BinearySearch( rnum, cumulateWeight, N+1 ); /* 搜索<=rnum的最小索引j */
		if ( j == N ) j--;
		ResampleIndex[i] = j;	/* 放入重采样索引数组 */		
	}

	delete cumulateWeight;

	return;	
}

void ReSelect( SPACESTATE * state, float * weight, int N )
{
	SPACESTATE * tmpState;
	int i, * rsIdx;

	tmpState = new SPACESTATE[N];
	rsIdx = new int[N];

	ImportanceSampling( weight, rsIdx, N ); /* 根据权重重新采样 */
	for ( i = 0; i < N; i++ )
		tmpState[i] = state[rsIdx[i]];//temState为临时变量,其中state[i]用state[rsIdx[i]]来代替
	for ( i = 0; i < N; i++ )
		state[i] = tmpState[i];

	delete[] tmpState;
	delete[] rsIdx;
	return;
}

void Propagate( SPACESTATE * state, int N)
{
	int i;
	int j;
	float rn[7];

	/* 对每一个状态向量state[i](共N个)进行更新 */
	for ( i = 0; i < N; i++ )  /* 加入均值为0的随机高斯噪声 */
	{
		for ( j = 0; j < 7; j++ ) rn[j] = randGaussian( 0, (float)0.6 ); /* 产生7个随机高斯分布的数 */
		/*为什么这么更新我也没看懂*/
		state[i].xt = (int)(state[i].xt);
		state[i].yt = (int)(state[i].yt);
		state[i].v_xt = (float)(state[i].v_xt + rn[2] * VELOCITY_DISTURB);
		state[i].v_yt = (float)(state[i].v_yt + rn[3] * VELOCITY_DISTURB);
		state[i].Hxt = (int)(state[i].Hxt +state[i].Hxt*state[i].at_dot + rn[4] * SCALE_DISTURB + 0.5);
		state[i].Hyt = (int)(state[i].Hyt +state[i].Hyt*state[i].at_dot + rn[5] * SCALE_DISTURB + 0.5);
		state[i].at_dot = (float)(state[i].at_dot + rn[6] * SCALE_CHANGE_D);
		//circle(pTrackImg,Point(state[i].xt,state[i].yt),3, Scalar(0,255,0),-1);  //在每一帧上显示粒子位置
	}
	return;
}

void Observe( SPACESTATE * state, float * weight, int N,
			 unsigned char * image, int W, int H,
			 float * ObjectHist0 )
{
	int i;
	float * ColorHist0;
	float * weights0 = NULL;
	float rho0;
	float wei_sum = 0;

	ColorHist0 = new float[nbin];
	weights0 = new float[NParticle];

	for ( i = 0; i < N; i++ )
	{
		CalcuColorHistogram( state[i].xt, state[i].yt,state[i].Hxt, state[i].Hyt,
			image, W, H, ColorHist0, nbin );

		rho0 = CalcuBhattacharyya( ColorHist0, ObjectHist0, nbin );
		weights0[i] = CalcuWeightedPi( rho0 );	
		wei_sum += weights0[i];
	}
	for( i = 0; i < N; i++)
		weight[i] = weights0[i] / wei_sum;

	delete ColorHist0;
	delete weights0;

	return;	
}

void Estimation( SPACESTATE * state, float * weight, int N, 
				SPACESTATE & EstState )
{
	int i;
	float at_dot, Hxt, Hyt, v_xt, v_yt, xt, yt;
	float weight_sum;

	at_dot = 0;
	Hxt = 0; 	Hyt = 0;
	v_xt = 0;	v_yt = 0;
	xt = 0;  	yt = 0;
	weight_sum = 0;
	for ( i = 0; i < N; i++ ) /* 求和 */
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
	/* 求平均 */
	if ( weight_sum <= 0 ) weight_sum = 1; /* 防止被0除，一般不会发生 */
	EstState.at_dot = at_dot/weight_sum;
	EstState.Hxt = (int)(Hxt/weight_sum + 0.5 );
	EstState.Hyt = (int)(Hyt/weight_sum + 0.5 );
	EstState.v_xt = v_xt/weight_sum;
	EstState.v_yt = v_yt/weight_sum;
	EstState.xt = (int)(xt/weight_sum + 0.5 );
	EstState.yt = (int)(yt/weight_sum + 0.5 );
	return;
}

# define ALPHA_COEFFICIENT      0.2     /* 目标模型更新权重取0.1-0.3 */

int ModelUpdate( SPACESTATE EstState, float * TargetHist0,  float PiT,
				unsigned char * img, int W, int H )
{
	float * EstHist0, Bha0, Pi0;
	int i, rvalue = -1;

	EstHist0 = new float [nbin];
	/* (1)在估计值处计算目标颜色直方图 */
	CalcuColorHistogram( EstState.xt, EstState.yt, EstState.Hxt, 
		EstState.Hyt, img, W, H, EstHist0, nbin );
	/* (2)计算Bhattacharyya系数 */
	Bha0  = CalcuBhattacharyya( EstHist0, TargetHist0, nbin );
	/* (3)计算概率权重 */
	Pi0 = CalcuWeightedPi( Bha0 );

	if ( Pi0 > PiT ) 
	{
		for ( i = 0; i < nbin; i++ )
		{
			TargetHist0[i] = (float)((1.0 - ALPHA_COEFFICIENT) * TargetHist0[i]
			+ ALPHA_COEFFICIENT * EstHist0[i]);
		}
		rvalue = 1;
	}

	delete EstHist0;
	return( rvalue );
}

int ColorParticleTracking( unsigned char * image, int &W, int &H, int & xc, int & yc,
						   int & Wx_h, int & Hy_h,float & max_weight)
{
	SPACESTATE EState;
	int i;
	/*选择：选择样本，并进行重采样*/
	ReSelect(states,weights,NParticle);
	/*传播：采样状态方程，对状态变量进行预测*/
	Propagate(states,NParticle);
	/*观测：对状态量进行更新，更新权值*/
	//Observe(states, weights, NParticle , image, W, H, ModelHist ,nbin);
	Observe(states, weights, NParticle , image, W, H, ModelHist0);
	/*估计：对状态量进行估计，提取位置量*/
	Estimation(states, weights, NParticle, EState);
	xc = EState.xt;
	yc = EState.yt;
	Wx_h = EState.Hxt;
	Hy_h = EState.Hyt;
	/*模型更新*/
	//ModelUpdate(EState ,ModelHist ,nbin ,Pi_Thres ,image ,W ,H);
	ModelUpdate( EState ,ModelHist0 ,Pi_Thres ,image ,W ,H);
	/*计算最大权重值*/
	max_weight = weights[0];
	for(i = 1;i < NParticle ;i++)
		max_weight = max_weight < weights[i] ? weights[i] : max_weight;
	/*进行合法性检验，不合法返回-1*/
	if(xc < 0 || yc < 0 || xc >= W || yc >= H || Wx_h <= 0 || Hy_h <= 0)
		return(-1);
	else
		return(1);
}

void IplToImge(Mat src,int w,int h)
{
	int i,j;	
	for (i = 0;i<h;i++)  //转成正向图像  行
		for(j=0;j<w;j++)  //列
		{
			img[(i*w+j)*3] = src.at<Vec3b>(i,j)[2];
			img[(i*w+j)*3+1] = src.at<Vec3b>(i,j)[1];
			img[(i*w+j)*3+2] = src.at<Vec3b>(i,j)[0];
		}
}

//通过鼠标划取一个初始矩形框
void on_MouseHandler(int event,int x,int y,int flags,void* param)
{
	Mat pFront1;
	int centerX,centerY;
	pFront1 = frame.clone();

	switch( event ){
	case EVENT_MOUSEMOVE:
	{
		if (drawing_box)
		{
			initbox.width = x-initbox.x;
			initbox.height = y-initbox.y;
		}
	}
    break;
	case EVENT_LBUTTONDOWN:
	{
		drawing_box = true;
		initbox = Rect( x, y, 0, 0 );
	}
    break;
	case EVENT_LBUTTONUP:
	{		
		drawing_box = false;
		if( initbox.width < 0 )
		{
        initbox.x += initbox.width;
        initbox.width *= -1;
		}
		if( initbox.height < 0 )
		{
        initbox.y += initbox.height;
        initbox.height *= -1;
		}
		xMin = initbox.x;  
		yMin = initbox.y;
		width = initbox.width;  
		height = initbox.height;
		rectangle(pFront1 ,initbox.tl(),initbox.br(),Scalar(0,0,250),2,8,0); 
		imshow ("tracking",pFront1 );
		return;
	}
	//gotbb=true;
    break;
  }

}



int main(){

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;
	uchar key = false;//用来设置暂停、停止键

	int rho_v;//表示合法值
	float max_weight;//最大权值
	int correct;
	
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

	string first_file = "C://"; 
	VideoCapture sequence(first_file);
	VideoCapture seq;
	//seq.open("");
	seq.open(0);
	if (!sequence.isOpened())
	{
		cout << "Failed to open the image sequence!\n" << endl;
		return 1;
	}
	namedWindow("tracking", 1);
	setMouseCallback ("tracking",on_MouseHandler,0);
	// Write Results
	ofstream resultsFile;	
	string resultsPath = "D:\\output.txt";	
	resultsFile.open(resultsPath);	

	// Frame counter
	int nFrames = 0;

	for(;;){
		// Read each frame from the list
		//sequence >> frame;
		seq >> frame;          //Mat frame; 帧
		if(frame.empty())
		{
			cout << "End of Sequence" << endl;
			break;
		}
		if(frame.rows >= 480 || frame.cols >= 640)
			resize(frame,frame,Size(640,480));
		imshow("tracking", frame);
		Hei=frame.rows;  //行
		Wid=frame.cols;  //列
		img = new unsigned char [Wid*Hei*3];   //unsigned char *img;  便于计算...
		IplToImge(frame,Wid,Hei);
		double time0 = static_cast<double>(getTickCount());
		// First frame, give the groundtruth to the tracker
		if (nFrames == 0) {
			while(pause) 
				if(waitKey(0)=='p')
					pause = false;
			tracker.init( Rect(xMin, yMin, width, height), frame );
			rectangle( frame, Point( xMin, yMin ), Point( xMin+width, yMin+height), Scalar( 0, 255, 255 ), 1, 8 );
			Initialize(xMin+width/2 ,yMin+height/2 ,width/2 ,height/2 ,img ,Wid ,Hei);
			//resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
		}
		// Update 更新
		else{
			result = tracker.update(frame);
			if (tracker.peak_value < 0.6) {
				cout << "target lost"<<endl;
			}
			else {
				rectangle(frame, Point(result.x + result.width / 2 - WidOut, result.y + result.height / 2 - HeiOut),
					Point(result.x + result.width / 2 + WidOut, result.y + result.height / 2 + HeiOut),
					Scalar(0, 0, 255), 1, 8);//红色
			}
			/*for ( int i = 0; i < NParticle; i++ )
			{
				states[i].xt = result.x+result.width/2;
				states[i].yt = result.y+result.height/2;
				states[i].Hxt = result.width/2;
				states[i].Hyt = result.height/2;
			}*/
			//rho_v = ColorParticleTracking( img, Wid, Hei, WidOut, HeiOut, max_weight);
			rho_v = ColorParticleTracking(img, Wid, Hei, xout, yout, WidOut, HeiOut, max_weight);
			if (rho_v == 1 && max_weight > 0.0001) {
				rectangle(frame, cvPoint(xout - WidOut, yout - HeiOut),
					cvPoint(xout + WidOut, yout + HeiOut), cvScalar(255, 0, 0), 2, 8, 0);//蓝色
				xin = xout; 
				yin = yout;         //上一帧的输出作为这一帧的输入
			}
			else {
				cout << "pf lost." << endl;
			}	
			//tracker._roi.width=2*WidOut;
			//tracker._roi.height=2*HeiOut;
			//resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
		}
		cout << tracker.peak_value <<endl;
		resultsFile << tracker.peak_value << endl;
		cout<<nFrames<<endl;
		nFrames++;		
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		//cout<< time0 << endl;
		//if (!SILENT){
		imshow("tracking", frame);
		key = waitKey(1);
		if(key == 'p') pause = true;
		if(key == 'q') return 0;
		while(pause) 
			if(waitKey(0)=='p')
				pause = false;
		//waitKey(1);
		//}
	}
	resultsFile.close();
	//listFile.close();

}

