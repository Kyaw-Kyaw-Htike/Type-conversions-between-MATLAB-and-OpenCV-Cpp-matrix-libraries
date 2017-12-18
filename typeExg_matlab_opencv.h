#ifndef TYPEEXG_MATLAB_OPENCV_H
#define TYPEEXG_MATLAB_OPENCV_H

// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

#include "mex.h"
#include <string.h>  //for memcpy
#include "opencv2/opencv.hpp"
#include <cstdlib>     // exit, EXIT_FAILURE
#include <cstdio>


// this namespace contains helper functions to be used only in this file (not be called from outside)
// they are all put in a namespace to avoid clashing (resulting in linker errors) with other same
// function names in other header files
namespace hpers_TEMatOpen
{

	// Template for mapping C primitive types to MATLAB types (commented out ones are not supported)
	template<class T> inline mxClassID getMatlabType()				{ return mxUNKNOWN_CLASS; }
	template<> inline mxClassID getMatlabType<char>()				{ return mxINT8_CLASS; }
	template<> inline mxClassID getMatlabType<unsigned char>()		{ return mxUINT8_CLASS; }
	template<> inline mxClassID getMatlabType<short>()				{ return mxINT16_CLASS; }
	template<> inline mxClassID getMatlabType<unsigned short>()		{ return mxUINT16_CLASS; }
	template<> inline mxClassID getMatlabType<int>()				{ return mxINT32_CLASS; }
	//template<> inline mxClassID getMatlabType<unsigned int>()		{ return mxUINT32_CLASS; }
	//template<> inline mxClassID getMatlabType<long long>()			{ return mxINT64_CLASS; }
	//template<> inline mxClassID getMatlabType<unsigned long long>() { return mxUINT64_CLASS; }
	template<> inline mxClassID getMatlabType<float>()				{ return mxSINGLE_CLASS; }
	template<> inline mxClassID getMatlabType<double>()				{ return mxDOUBLE_CLASS; }

	template<class T>
	inline void checkTypeOrErr(const mxArray *m) {
		if (mxGetClassID(m) != getMatlabType<T>()) {
			printf("Type of input matrix does not match with template type");
			exit(EXIT_FAILURE);
		}
	}

	// convert typename and nchannels to opencv mat type such as CV_32FC1
	template <typename T>
	int getOpencvType(int nchannels)
	{
		int depth = cv::DataType<T>::depth;
		return (CV_MAT_DEPTH(depth) + (((nchannels)-1) << CV_CN_SHIFT));
	}
	
}


template <typename T, int nchannels>
void mxArray2matOpencv(const mxArray* mm, cv::Mat &mcv)
{
	// NB1: there is no choice but to deep copy as mxArray is column major and opencv matrix is row major.
	// NB2: assume that mxArray is 2D with any number of channels.
	// (which can be in a way thought of as a 3D matrix, and when nchannels=1, then it's a 2D matrix ).	
	int ndims = (int) mxGetNumberOfDimensions(mm);
	const size_t *dims = mxGetDimensions(mm);
	unsigned int nrows = (unsigned int) dims[0];
	unsigned int ncols = (unsigned int)dims[1];
	unsigned int nchannels_mm = ndims == 2 ? 1 : (unsigned int) dims[2]; // this should be the same as template input

	mcv.create(nrows, ncols, hpers_TEMatOpen::getOpencvType<T>(nchannels));
	T *ptr_mm = (T*)mxGetData(mm);
	unsigned long counter = 0;

	for (int k = 0; k < nchannels; k++)
		for (int j = 0; j < ncols; j++)
			for (int i = 0; i < nrows; i++)
				mcv.at<cv::Vec<T, nchannels>>(i, j)[k] = ptr_mm[counter++];

}


template <typename T, int nchannels>
void matOpencv2mxArray(const cv::Mat mcv, mxArray* &mm)
{
	// NB1: there is no choice but to deep copy as mxArray is column major and opencv matrix is row major.
	// NB2: assume that opencv mat is 2D with any number of channels 
	// (which can be in a way thought of as a 3D matrix, and when nchannels=1, then it's a 2D matrix ).
	
	int nrows = mcv.rows; 
	int ncols = mcv.cols;

	size_t dims[3] = {(size_t)nrows, (size_t)ncols, (size_t)nchannels};
	size_t ndims = nchannels > 1 ? 3 : 2;

	mm = mxCreateNumericArray(ndims, dims, hpers_TEMatOpen::getMatlabType<T>(), mxREAL);
	T *ptr_mm = (T*)mxGetData(mm);
	unsigned long counter = 0;

	for (int k = 0; k < nchannels; k++)
		for (int j = 0; j < ncols; j++)
			for (int i = 0; i < nrows; i++)
				 ptr_mm[counter++] = mcv.at<cv::Vec<T, nchannels>>(i, j)[k];

}


/*

Demo of the functions.

========== Code =============

MatlabEngWrapper mew;
mew.init();
mew.exec("img = imread('D:/Research/Datasets/INRIAPerson_Piotr/Test/images/set01/V000/I00000.png');");
mxArray* img = mew.receive("img");
cv::Mat m; mxArray2matOpencv<unsigned char, 3>(img, m); mxDestroyArray(img);
mxArray* img2; matOpencv2mxArray<unsigned char, 3>(m, img2);
mew.send("img2", img2);
mew.exec("img3 = imgaussfilt(img2,8);");
mxArray* img3 = mew.receive("img3");
cv::Mat m3; mxArray2matOpencv<unsigned char, 3>(img3, m3); mxDestroyArray(img3);
cv::cvtColor(m, m, CV_RGB2BGR);
cv::cvtColor(m3, m3, CV_RGB2BGR);
imshow_qt(m, ui.label_1);
imshow_qt(m3, ui.label_2);

========== Output in C++ QT =============

// Note, this will correctly output in C++ QT two images where the first image
// corresponding to ui.label_1 will have the input image and the second
// image displayed on ui.label_2 will have the smoothed image.


========== Code =============

MatlabEngWrapper mew;
mew.init();
mew.exec("clear all; X = [1,2;3,4]; X = uint16(X);");
mxArray* X = mew.receive("X");
cv::Mat X_cv; mxArray2matOpencv<unsigned short, 1>(X, X_cv); mxDestroyArray(X);
cv::Mat Y_cv = X_cv + 2;
mxArray* Y; matOpencv2mxArray<unsigned short, 1>(Y_cv, Y);
mew.send("Y", Y);

========== Output in Matlab =============

» whos, X, Y
  Name      Size            Bytes  Class     Attributes

  X         2x2                 8  uint16              
  Y         2x2                 8  uint16              


X =

      1      2
      3      4


Y =

      3      4
      5      6
	  
	  
========== Code =============

MatlabEngWrapper mew;
mew.init();	  
mew.exec("clear all; X(:,:,1) = [1,2;3,4]; X(:,:,2) = [1,2;3,4]; X = single(X);");
mxArray* X = mew.receive("X");
cv::Mat X_cv; mxArray2matOpencv<float, 2>(X, X_cv); mxDestroyArray(X);
cv::Mat Y_cv = X_cv + 2; // actually this will just add "2" to the first channel of the matrix
mxArray* Y; matOpencv2mxArray<float, 2>(Y_cv, Y);
mew.send("Y", Y);

========== Output in Matlab =============

» whos, X, Y
  Name      Size             Bytes  Class     Attributes

  X         2x2x2               32  single              
  Y         2x2x2               32  single              


X(:,:,1) =

     1     2
     3     4


X(:,:,2) =

     1     2
     3     4


Y(:,:,1) =

     3     4
     5     6


Y(:,:,2) =

     1     2
     3     4

*/


#endif