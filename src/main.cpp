#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

void bilateralFilter(const Mat & input, Mat & output, int r,double sI, double sS);
void bilateralFilterOpenCV(const Mat & input, Mat & output, int r,double sI, double sS);
void bilateralFilter8(const Mat & input, Mat & output, int r,double sI, double sS);
void bilateralFilter16(const Mat & input, Mat & output, int r,double sI, double sS);
void bilateralFilter32(const Mat & input, Mat & output, int r,double sI, double sS);
void bilateralFilterTexture(const Mat & input, Mat & dstMat, int r, double sI, double sS);
void bilateralFilterTextureOpm(const Mat & input, Mat & dstMat, int r, double sI, double sS);
void bilateralFilterTextureOpmShared(const Mat & input, Mat & dstMat, int r, double sI, double sS);

int main() {
	ifstream input("original_1408x1408_16bit.raw", ios::binary);
	ofstream output;
	int width = 768, height = 768;
	unsigned short* pData = (unsigned short*)malloc(width * height * sizeof(unsigned short));
	//Mat src(cvSize(width, height), CV_16UC1, Scalar::all(0));
	Mat src(cvSize(width, height), CV_16UC1, pData);
	if (input.good())
	{
		input.read((char*)src.data, 65536);
		input.read((char*)src.data, width * height * sizeof(unsigned short));
		input.close();

		printf("src[100][100] = %d\n", *((unsigned short*)(src.data)+100*src.cols+100));
		Mat dst(cvSize(width, height), CV_16UC1, (unsigned short)0);
		// Own bilateral filter (input,output,filter_half_size,sigmaI,sigmaS)
		//bilateralFilter16(src, dst, 4, 75.0, 75.0);
		Mat fsrc(cvSize(width, height), CV_32FC1);
		Mat fdst(cvSize(width, height), CV_32FC1);
		Mat csrc(cvSize(width, height), CV_8UC1);
		Mat cdst(cvSize(width, height), CV_8UC1);
		src.convertTo(dst, CV_16UC1);
		src.convertTo(fsrc, CV_32FC1);
		src.convertTo(fdst, CV_32FC1);
		src.convertTo(csrc, CV_8UC1);
		dst.convertTo(cdst, CV_8UC1);

		float sI = 66.0, sS = 1;
		char szName[256];
		printf("================================\n");
		bilateralFilterOpenCV(fsrc, fdst, 4, sI, sS);
		sprintf(szName, "sI_%.1f_sS_%.1f_filterCV.raw", sI, sS);
		fdst.convertTo(dst, CV_16UC1);
		output.open(szName, ios::binary);
		output.write((char*)dst.data, width*height*sizeof(unsigned short));
		output.close();
		printf("================================\n");
		bilateralFilter8(src, dst, 4, sI, sS);
		sprintf(szName, "sI_%.1f_sS_%.1f_filter8.raw", sI, sS);
		output.open(szName, ios::binary);
		output.write((char*)cdst.data, width*height*sizeof(unsigned char));
		output.close();
		printf("================================\n");
		bilateralFilter16(src, dst, 4, sI, sS);
		sprintf(szName, "sI_%.1f_sS_%.1f_filter16.raw", sI, sS);
		output.open(szName, ios::binary);
		output.write((char*)dst.data, width*height*sizeof(unsigned short));
		output.close();
		printf("================================\n");
		bilateralFilter32(fsrc, fdst, 4, sI, sS);
		fdst.convertTo(dst, CV_16UC1);
		sprintf(szName, "sI_%.1f_sS_%.1f_filter32.raw", sI, sS);
		output.open(szName, ios::binary);
		output.write((char*)dst.data, width*height*sizeof(unsigned short));
		output.close();

		printf("================================\n");
		bilateralFilterTexture(fsrc, fdst, 4, sI, sS);
		fdst.convertTo(dst, CV_16UC1);
		sprintf(szName, "sI_%.1f_sS_%.1f_filterTex.raw", sI, sS);
		output.open(szName, ios::binary);
		output.write((char*)dst.data, width*height*sizeof(unsigned short));
		output.close();

		printf("================================\n");
		bilateralFilterTextureOpm(fsrc, fdst, 4, sI, sS);
		fdst.convertTo(dst, CV_16UC1);
		sprintf(szName, "sI_%.1f_sS_%.1f_filterOpm.raw", sI, sS);
		output.open(szName, ios::binary);
		output.write((char*)dst.data, width*height*sizeof(unsigned short));
		output.close();

		printf("================================\n");
		bilateralFilterTextureOpmShared(fsrc, fdst, 4, sI, sS);
		fdst.convertTo(dst, CV_16UC1);
		sprintf(szName, "sI_%.1f_sS_%.1f_filterOpmSrd.raw", sI, sS);
		output.open(szName, ios::binary);
		output.write((char*)dst.data, width*height*sizeof(unsigned short));
		output.close();



	}else
	{
		printf("failed to open file.\n");
	}
	free(pData);
	pData = NULL;
}
