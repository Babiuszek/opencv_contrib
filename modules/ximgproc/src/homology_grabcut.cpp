/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "precomp.hpp"
#include "gcgraph.hpp"
#include <time.h>
#include <limits>

#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;
using namespace ml;

namespace cv {

#define FILTERS 13
#define CHANNELS 17
#define HOM_CHANNELS 10

#define Vecf_f	Vec<float, FILTERS>
#define Vecd_c	Vec<double, CHANNELS>
#define Vecd_dc Vec<double, 2*CHANNELS>

/*
This is implementation of image segmentation algorithm GrabCut described in
"GrabCut ? Interactive Foreground Extraction using Iterated Graph Cuts".
Carsten Rother, Vladimir Kolmogorov, Andrew Blake.
 */

/*
 GMM - Gaussian Mixture Model template
*/
template <typename DataType, int DataLength> class GMM
{
public:
	// Types for ease of reference
	typedef Vec< DataType, DataLength > DataVec;
	// Static component amount for division
    static const int componentsCount = 5;

	// The class itself
    GMM( Mat& _model );
    double operator()( const DataVec color ) const;
    double operator()( int ci, const DataVec color ) const;
    int whichComponent( const DataVec color ) const;

    void initLearning();
    void addSample( int ci, const DataVec color );
    void endLearning();

private:
    void calcInverseCovAndDeterm( int ci );
    Mat model;
    double* coefs;
    double* mean;
    double* cov;

    double inverseCovs[componentsCount][DataLength][DataLength];
    double covDeterms[componentsCount];

    double sums[componentsCount][DataLength];
    double prods[componentsCount][DataLength][DataLength];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};
template <typename DataType, int DataLength> GMM< DataType, DataLength >::GMM( Mat& _model )
{
    const int modelSize = DataLength/*mean*/ + DataLength*DataLength/*covariance*/ + 1/*component weight*/;
    if( _model.empty() )
    {
        _model.create( 1, modelSize*componentsCount, CV_64FC1 );
        _model.setTo(Scalar(0));
    }
    else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
        CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 43*componentsCount" );

    model = _model;

	// coefs has size of ComponentCount (each component has its own coefficient)
    coefs = model.ptr<double>(0);
	// Mean has the size of DataLength*ComponentCount (each DataLength has its own mean for each component)
    mean = coefs + componentsCount;
	// The rest is simply taken by Covariance matrices
    cov = mean + DataLength*componentsCount;

	// Initialize our GMMs
    for( int ci = 0; ci < componentsCount; ci++ )
        if( coefs[ci] > 0 )
             calcInverseCovAndDeterm( ci );
}
template <typename DataType, int DataLength> double GMM< DataType, DataLength >::operator()( const DataVec color ) const
{
    double res = 0;
	// Sum of all components coefs * (
    for( int ci = 0; ci < componentsCount; ci++ )
        res += coefs[ci] * (*this)(ci, color );
    return res;
}
template <typename DataType, int DataLength> double GMM< DataType, DataLength >::operator()( int ci, const DataVec color ) const
{
    double res = 0;
	// If possibility exists
    if( coefs[ci] > 0 )
    {
		// Make sure Determ is not 0, the matrix is not singular
        CV_Assert( covDeterms[ci] != 0 );
		// Find the difference of our color to the mean of that particular component
        Vec< DataType, DataLength > diff = color;
        double* m = mean + DataLength*ci;
		for (int i = 0; i < DataLength; i++)
			diff[i] -= m[i];
		// Calculate multiplier (component sum(inverseCov*diff)
		/*
        double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
                   + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
                   + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
				   */
		// The below loop simulates the formula above expanded into n DataLengths
		double mult = 0.0;
		for (int i = 0; i < DataLength; i++)
			for (int j = 0; j < DataLength; j++)
				mult += diff[i]*diff[j]*inverseCovs[ci][j][i];
        // Calculate return value?
		res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
    }
    return res;
}
template <typename DataType, int DataLength> int GMM< DataType, DataLength >::whichComponent( const DataVec color ) const
{
    int k = 0;
    double max = 0;

    for( int ci = 0; ci < componentsCount; ci++ )
    {
        double p = (*this)( ci, color );
        if( p > max )
        {
            k = ci;
            max = p;
        }
    }
    return k;
}
template <typename DataType, int DataLength> void GMM< DataType, DataLength >::initLearning()
{
	// Initialize every value we care about to 0
    for( int ci = 0; ci < componentsCount; ci++)
    {
		// All sums and prods of that component are set to 0
		for (int i = 0; i < DataLength; i++)
		{
			sums[ci][i] = 0;
			for (int j = 0; j < DataLength; j++)
				prods[ci][i][j] = 0;
		}
		// Same for it's sample counts
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}
template <typename DataType, int DataLength> void GMM< DataType, DataLength >::addSample( int ci, const DataVec color )
{
	// Add to sums
	for (int i = 0; i < DataLength; i++)
		sums[ci][i] += color[i];
	// Change prods
	for (int i = 0; i < DataLength; i++)
		for (int j = 0; j < DataLength; j++)
			prods[ci][i][j] += color[i]*color[j];
	// Increase our sample counters
    sampleCounts[ci]++;
    totalSampleCount++;
}
template <typename DataType, int DataLength> void GMM< DataType, DataLength >::endLearning()
{
	// Learning is done
    const double variance = 0.01;
    for( int ci = 0; ci < componentsCount; ci++ )
    {
		// Set the sample count. If a cluster is empty, just set it's coef
        int n = sampleCounts[ci];
        if( n == 0 )
            coefs[ci] = 0;
        else
        {
			// If it's not empty, we need to calculate it's means and covariance matrix
            coefs[ci] = (double)n/totalSampleCount;

			// Means, initialize the pointer
            double* m = mean + DataLength*ci;
			for (int i = 0; i < DataLength; i++)
				m[i] = sums[ci][i]/n; // Set all the means

			// Covariance matrix
            double* c = cov + DataLength*DataLength*ci;
			// The loops below simulate the formulas above for more DataLengths
			// cov(X, Y) = E(X * Y) - EX * EY; where E is expected value
			for (int i = 0; i < DataLength; i++)
				for (int j = 0; j < DataLength; j++)
					c[DataLength*i+j] = prods[ci][i][j]/n - m[i]*m[j];

			// Calculate determinant
			// We first write our data into a temporary Mat
			Mat c_mat;
			c_mat.create(DataLength, DataLength, CV_64FC1);
			for (int i = 0; i < DataLength; i++)
				for (int j = 0; j < DataLength; j++)
					c_mat.at<double>(i, j) = prods[ci][i][j]/n - m[i]*m[j];
			
			//double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
			double dtrm = determinant(c_mat);

			// And check for singular
			if ( dtrm <= std::numeric_limits<double>::epsilon() )
            {
                // Adds the white noise to avoid singular covariance matrix
				// We change the diagonal. This is more so to avoid marginally negative determinant
				for (int i = 0; i < DataLength*DataLength; i = i + DataLength + 1)
					c[i] += variance;
            }

            calcInverseCovAndDeterm(ci);
        }
    }
}
template <typename DataType, int DataLength> void GMM< DataType, DataLength >::calcInverseCovAndDeterm( int ci )
{
    if( coefs[ci] > 0 )
    {
		// Initialize our covariance matrix
        double *c = cov + DataLength*DataLength*ci;
		
		// Calculate determinant
		// We first write our data into a temporary Mat
		Mat c_mat;
		c_mat.create(DataLength, DataLength, CV_64FC1);
		for (int i = 0; i < DataLength; i++)
			for (int j = 0; j < DataLength; j++)
				c_mat.at<double>(i, j) = c[DataLength*i+j];
        //double dtrm =
        //      covDeterms[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
		covDeterms[ci] = determinant(c_mat);
		CV_Assert(covDeterms[ci] > 0.0);

		// Calculate inverse covariance matrix
		Mat c_inv;
		c_inv.create(DataLength, DataLength, CV_64FC1);
		double ans = invert(c_mat, c_inv);
		// Make sure not singular, using invert function output;
        CV_Assert( ans != 0 );

		// Copy ou answer to GMM data
		for (int i = 0; i < DataLength; i++)
			for (int j = 0; j < DataLength; j++)
				inverseCovs[ci][i][j] = c_inv.at<double>(i, j);
    }
}

/*
  Check size, type and element values of mask matrix.
 */
static void checkMask( const Mat& img, const Mat& mask )
{
    if( mask.empty() )
        CV_Error( CV_StsBadArg, "mask is empty" );
    if( mask.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "mask must have CV_8UC1 type" );
    if( mask.cols != img.cols || mask.rows != img.rows )
        CV_Error( CV_StsBadArg, "mask must have as many rows and cols as img" );
    for( int y = 0; y < mask.rows; y++ )
    {
        for( int x = 0; x < mask.cols; x++ )
        {
            uchar val = mask.at<uchar>(y,x);
            if( val!=GC_PR_BGD && val!=GC_PR_FGD && val!=GC_BGD && val!=GC_FGD)
                CV_Error( CV_StsBadArg, "mask element value must be equal"
                    "GC_PR_BGD or GC_PR_FGD or GC_BGD or GC_FGD" );
        }
    }
}

/*
  Initialize mask using rectangular.
*/
static void initMaskWithRect( Mat& mask, Size imgSize, Rect rect )
{
    mask.create( imgSize, CV_8UC1 );
    mask.setTo( GC_BGD );

    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, imgSize.width-rect.x);
    rect.height = std::min(rect.height, imgSize.height-rect.y);

    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
}

//====================[ Multi-dimensional versions of original functions ]=====================
template <typename ImgType, int DataLength> static
double calcBeta( const Mat& img )
{
    double beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec< ImgType, DataLength > color = img.at< Vec< ImgType, DataLength > >(y,x);
            if( x>0 ) // left
            {
                Vec< ImgType, DataLength > diff = color - img.at< Vec< ImgType, DataLength > >(y,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 && x>0 ) // upleft
            {
                Vec< ImgType, DataLength > diff = color - img.at< Vec< ImgType, DataLength > >(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // up
            {
                Vec< ImgType, DataLength > diff = color - img.at< Vec< ImgType, DataLength > >(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // upright
            {
                Vec< ImgType, DataLength > diff = color - img.at< Vec< ImgType, DataLength > >(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    if( beta <= std::numeric_limits<double>::epsilon() )
        beta = 0;
    else
        beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );

    return beta;
}

template <typename ImgType, typename DataType, int DataLength>
static void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma )
{
	// gamma is 50 and used for straight edges, gammaDivSqrt2 is used for diagonal ones
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);

	// Create materials, each having amount of vertices equal to our img
	// CV_64FC1 <- this defines a vector of 1 value of 64-bit float (double), used for defining Materials
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );

	// Main loop
	// As these things are, they are equal to roughly:
	// W = ( gamma / dist(i,j) ) * exp ( - diff.dot(diff) / 2*avg(diff.dot(diff)) )
	// diff.dot(diff) / 2*avg(diff.dot(diff)) changes from 0 to infinity (theoretically)
	// Where diff is the difference between color vectors
	// Note: We apply a gauss with variance equal to 1/2*avg difference in color
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec< DataType, DataLength > color = (Vec<DataType, DataLength>)img.at< Vec< ImgType, DataLength > >(y,x);
            if( x-1>=0 ) // left
            {
                Vec< DataType, DataLength > diff = color - 
					(Vec<DataType, DataLength>)img.at< Vec< ImgType, DataLength > >(y,x-1);
                leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                leftW.at<double>(y,x) = 0;
            if( x-1>=0 && y-1>=0 ) // upleft
            {
                Vec< DataType, DataLength > diff = color - 
					(Vec<DataType, DataLength>)img.at< Vec< ImgType, DataLength > >(y-1,x-1);
                upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                upleftW.at<double>(y,x) = 0;
            if( y-1>=0 ) // up
            {
                Vec< DataType, DataLength > diff = color - 
					(Vec<DataType, DataLength>)img.at< Vec< ImgType, DataLength > >(y-1,x);
                upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                upW.at<double>(y,x) = 0;
            if( x+1<img.cols && y-1>=0 ) // upright
            {
                Vec< DataType, DataLength > diff = color - 
					(Vec<DataType, DataLength>)img.at< Vec< ImgType, DataLength > >(y-1,x+1);
                uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                uprightW.at<double>(y,x) = 0;
        }
    }
}

template <typename ImgType, typename DataType, int DataLength>
static void initGMMs( const Mat& img, const Mat& mask,
	GMM< DataType, DataLength >& bgdGMM, GMM< DataType, DataLength >& fgdGMM )
{
	// Always 10 iterations, always clustering into 2 centers (logical)
	const int kMeansItCount = 10;
	const int kMeansType = KMEANS_PP_CENTERS;

	// Create vectors representing sumples and push our points into proper containers
	Mat bgdLabels, fgdLabels;
	std::vector< Vec< float, DataLength > > bgdSamples, fgdSamples;
	Point p;
	for( p.y = 0; p.y < img.rows; p.y++ )
	{
		for( p.x = 0; p.x < img.cols; p.x++ )
		{
		if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
			bgdSamples.push_back( (Vec< float, DataLength >)img.at< Vec< ImgType, DataLength > >(p) );
		else // GC_FGD | GC_PR_FGD
			fgdSamples.push_back( (Vec< float, DataLength >)img.at< Vec< ImgType, DataLength > >(p) );
		}
	}
	// Standard debug, none should be empty

	// Anti-crash quick-fix, random pixels are moved if too little are picked for either FGD or BGD
	if (bgdSamples.size() <= 5)
	{
		RNG rng(rand());
		for (int i = 0; i < 6; ++i)
		{
			int it = rng.next()%fgdSamples.size();
			bgdSamples.push_back( fgdSamples[it] );
			fgdSamples.erase( fgdSamples.begin() + it );
		}
	}
	else if (fgdSamples.size() <= 5)
	{
		RNG rng(rand());
		for (int i = 0; i < 6; ++i)
		{
			int it = rng.next()%bgdSamples.size();
			fgdSamples.push_back( bgdSamples[it] );
			bgdSamples.erase( bgdSamples.begin() + it );
		}
	}

	// Debuf, has meaning if upper rewrites are removed
	if (bgdSamples.empty() || fgdSamples.empty())
	{
		std::cout << "bgdSamples=" << bgdSamples.size() << ", fgdSamples=" << fgdSamples.size() << std::endl;
		imwrite("bin/error/er_image.png", img);
		imwrite("bin/error/er_mask.png", mask);
	}
	CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );

	// Transform vector of Vec3f into an actual 2D material
	// Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP) <- probably this one
	Mat _bgdSamples( (int)bgdSamples.size(), DataLength, CV_32FC1, &bgdSamples[0][0] );
	// Run the K-means algorythm
	// (_data = bgdSamples, K=componentsCount(5), _bestLabels=bgdLabels(output),
	//	TermCriteria, attempts=0, flags=kMeansType(2), _centers=noArray(default))
	kmeans( _bgdSamples, GMM< DataType, DataLength >::componentsCount, bgdLabels,
		TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

	// Do the same for FGD...
	Mat _fgdSamples( (int)fgdSamples.size(), DataLength, CV_32FC1, &fgdSamples[0][0] );
	kmeans( _fgdSamples, GMM< DataType, DataLength >::componentsCount, fgdLabels,
		TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

	// Learn GMMs
	bgdGMM.initLearning();
	for( int i = 0; i < (int)bgdSamples.size(); i++ )
		bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
	bgdGMM.endLearning();

	fgdGMM.initLearning();
	for( int i = 0; i < (int)fgdSamples.size(); i++ )
		fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
	fgdGMM.endLearning();
}

template <typename ImgType, typename DataType, int DataLength>
static void assignGMMsComponents( const Mat& img, const Mat& mask,
	const GMM< DataType, DataLength >& bgdGMM, const GMM< DataType, DataLength >& fgdGMM, Mat& compIdxs )
{
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            Vec< DataType, DataLength > color = (Vec< DataType, DataLength >)img.at< Vec< ImgType, DataLength > >(p);
            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
                bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
        }
    }
}

template <typename ImgType, typename DataType, int DataLength>
static void learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs,
	GMM< DataType, DataLength >& bgdGMM, GMM< DataType, DataLength >& fgdGMM )
{
	bgdGMM.initLearning();
	fgdGMM.initLearning();
	Point p;
	for( int ci = 0; ci < GMM< DataType, DataLength >::componentsCount; ci++ )
	{
		for( p.y = 0; p.y < img.rows; p.y++ )
		{
			for( p.x = 0; p.x < img.cols; p.x++ )
			{
				if( compIdxs.at<int>(p) == ci )
				{
					if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
						bgdGMM.addSample( ci, img.at< Vec< ImgType, DataLength > >(p) );
					else
						fgdGMM.addSample( ci, img.at< Vec< ImgType, DataLength > >(p) );
				}
			}
		}
	}
	bgdGMM.endLearning();
	fgdGMM.endLearning();
}

template <typename ImgType, typename DataType, int DataLength>
static void constructGCGraph( const Mat& img, const Mat& mask,
	const GMM< DataType, DataLength >& bgdGMM, const GMM< DataType, DataLength >& fgdGMM,
	double lambda, const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
	GCGraph<double>& graph )
{
	// Initialize our graph values
    int vtxCount = img.cols*img.rows,
		// Edges go left, upleft, up, right. That is why we substract cols+rows 3 times
		// Left		-> Extra rows
		// Upleft	-> Extra rows + cols
		// Up		-> Extra cols
		// Upright	-> Extra rows + cols
		// We then substract corners twice, need to add them back hence the + 2
		// Each edge is double way (to and from) for Ford Fulkerson, so we multiply the total by 2
        edgeCount = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);
	// Create our graph
    graph.create(vtxCount, edgeCount);

	// Main loop, iterates for each point
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++)
        {
            // add node
            int vtxIdx = graph.addVtx();
            Vec< DataType, DataLength > color = (Vec< DataType, DataLength >)img.at< Vec< ImgType, DataLength > >(p);

            // set t-weights
			// Note - which is sink and which is source has no meaning, just the segmentation
            double fromSource, toSink;
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
				// It is inconclusive, set it's edges as derived from GMMs (using log likelyhood for that color)
                fromSource = -log( bgdGMM(color) );
                toSink = -log( fgdGMM(color) );
            }
            else if( mask.at<uchar>(p) == GC_BGD )
            {
                fromSource = 0;
                toSink = lambda;
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
            graph.addTermWeights( vtxIdx, fromSource, toSink );

            // set n-weights
			// Derive the edge values from previously calculated Material matrices
            if( p.x>0 )
            {
                double w = leftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-1, w, w );
            }
            if( p.x>0 && p.y>0 )
            {
                double w = upleftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols-1, w, w );
            }
            if( p.y>0 )
            {
                double w = upW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols, w, w );
            }
            if( p.x<img.cols-1 && p.y>0 )
            {
                double w = uprightW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols+1, w, w );
            }
        }
    }
}
//==========================[ End of multi-dimensional functions ]=============================

// Functions to update toLog strings
std::string toString(float value)
{
	// Init
	std::string Ans = "";
	bool negative = value < 0.f ? true : false;
	char c;

	if (negative)
		value = -value;

	// Integer
	int t = (int)floor(value);
	if (t == 0)
		Ans = Ans + "0";
	value -= t;
	while (t > 0)
	{
		c = 48 + t % 10;
		Ans = c + Ans;
		t /= 10;
	}

	if (value > 0)
		Ans = Ans + ".";
	
	if (negative)
		Ans = "-" + Ans;

	// Fraction
	std::string fraction = "";
	t = (int)floor(value*1000000);
	while (t > 0)
	{
		c = 48 + t % 10;
		fraction = c + fraction;
		t /= 10;
	}
	// Done
	return Ans + fraction;
}

/*
  Estimate segmentation using MaxFlow algorithm
*/
static void estimateSegmentation( GCGraph<double>& graph, Mat& mask )
{
    graph.maxFlow();
    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                if( graph.inSourceSegment( p.y*mask.cols+p.x /*vertex index*/ ) )
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}

// Shrinking function, creates a material by times smaller
Mat* shrink( const Mat& input, Mat& mask, const int by )
{
	// Create our shrunk Material
	Mat* output = new Mat( input.rows/by, input.cols/by, CV_64FC(2*CHANNELS) );

	// For each point in ouput image...
	Point p_i; // Input iterator
	Point p_o; // Output iterator
	for( p_o.y = 0; p_o.y < output->rows; p_o.y++ )
	{
		for( p_o.x = 0; p_o.x < output->cols; p_o.x++ )
		{
			// ...we take it's values (vector of 3 colors, grayscale and 13 Schmidt filters,
			// each having mean and standard deviation)
			Vecd_dc& values = output->at<Vecd_dc>(p_o);
			// ...initialize base values as 0
			for (int i = 0; i < 2*CHANNELS; i++)
				values[i] = 0.0;

			// And calculate these values using the area from input image. We first calculate the means
			int wwidth = 0;
			int wheight = 0;
			for ( p_i.y = by*p_o.y; (p_i.y < by*(p_o.y+1)) && (p_i.y < input.rows); p_i.y++)
			{
				++wheight;
				for ( p_i.x = by*p_o.x; (p_i.x < by*(p_o.x+1)) && (p_i.x < input.cols); p_i.x++)
				{
					++wwidth;
					Vecd_c color = input.at<Vecd_c>(p_i);
					for (int i = 0; i < CHANNELS; i++)
						values[i] += color.val[i];
				}
			}
			for (int i = 0; i < CHANNELS; i++)
				values[i] /= wwidth*wheight;
			
			// Then calculate the standard deviations
			for ( p_i.y = by*p_o.y; (p_i.y < by*(p_o.y+1)) && (p_i.y < input.rows); p_i.y++)
			{
				for ( p_i.x = by*p_o.x; (p_i.x < by*(p_o.x+1)) && (p_i.x < input.cols); p_i.x++)
				{
					Vecd_c color = input.at<Vecd_c>(p_i);
					for (int i = 0; i < CHANNELS; i++)
						values[CHANNELS+i] += pow(values[i] - color.val[i], 2);
				}
			}
			for (int i = CHANNELS; i < 2*CHANNELS; i++)
				values[i] = sqrt(values[i] / (wwidth*wheight));
		}
	}
	
	// Now time to shrink the mask
	Mat out_mask;
	out_mask.create( output->rows, output->cols, CV_8UC1 );
	out_mask.setTo( Scalar(2) );
	for( p_o.y = 0; p_o.y < out_mask.rows; p_o.y++ )
	{
		for( p_o.x = 0; p_o.x < out_mask.cols; p_o.x++ )
		{
			// Take a single point on output mask
			uchar& value = out_mask.at<uchar>(p_o);

			// And compare it to its by*by factors. GC_FGD > GC_BGD > GC_PR_FGD > GC_PR_BGD
			for ( p_i.y = by*p_o.y; (p_i.y < by*(p_o.y+1)) && (p_i.y < mask.rows); p_i.y++)
				for ( p_i.x = by*p_o.x; (p_i.x < by*(p_o.x+1)) && (p_i.x < mask.cols); p_i.x++)
					if (mask.at<uchar>(p_i) == GC_FGD)
						value = GC_FGD;
					else if (mask.at<uchar>(p_i) == GC_BGD && value != GC_FGD)
						value = GC_BGD;
					else if (value != GC_FGD && value != GC_BGD)
						value = max( mask.at<uchar>(p_i), value );
		}
	}
	out_mask.copyTo(mask);
	
	// And done!
	return output;
}

// Function that copies the answer from shrunk image onto the bigger mask
void expandShrunkMat(const Mat& input, Mat& output, const int by)
{
	Point p_i; // Input iterator
	Point p_o; // Output iterator
	
	// Go through each point of shrunk, input mask...
	for( p_i.y = 0; p_i.y < input.rows; ++p_i.y )
	{
		for( p_i.x = 0; p_i.x < input.cols; ++p_i.x )
		{
			uchar flag = input.at<uchar>(p_i);
			// And set its value onto every single pixel from a corresponding area on output
			for ( p_o.y = by*p_i.y; (p_o.y < by*(p_i.y+1)) && (p_o.y < output.rows); ++p_o.y)
				for ( p_o.x = by*p_i.x; (p_o.x < by*(p_i.x+1)) && (p_o.x < output.cols); ++p_o.x)
					if (output.at<uchar>(p_o) != GC_FGD && output.at<uchar>(p_o) != GC_BGD)
						output.at<uchar>(p_o) = flag;
		}
	}
	// End of input for
}

// Change our base image into grayscale and expand it into n dimensions for further filtering
Mat* grey_and_expand( const Mat& input )
{
	// Create output material of input size and n dimensions
	Mat* output = new Mat(input.rows, input.cols, CV_64FC(CHANNELS));
	
	// Calculate grayscale values
	Point p;
	for (p.y = 0; p.y < input.rows; p.y++)
	{
		for (p.x = 0; p.x < input.cols; p.x++)
		{
			const Vec3b vi = input.at<Vec3b>(p);
			Vecd_c& vo = output->at<Vecd_c>(p);
			
			// Writing first three channels into new matrix
			for (int i = 0; i < 3; ++i)
				vo[i] = vi[i];

			// 1) (0.2126*R + 0.7152*G + 0.0722*B) <- Relative luminance according to wiki
			// 2) (0.299*R + 0.587*G + 0.114*B) <- Suggested by W3C Working Draft
			// 3) sqrt( 0.299*R^2 + 0.587*G^2 + 0.114*B^2 ) <- Photoshop does something close to this
			// Calculate grayscale value, here we are using 3rd formula
			vo[3] = sqrt( 0.299*vi[0]*vi[0] + 0.587*vi[1]*vi[1] + 0.114*vi[2]*vi[2] );
			// Set all other values to the calculated value
			for (int i = 4; i < CHANNELS; i++)
				vo[i] = vo[3];
		}
	}

	// All done, return the answer
	return output;
}

template <typename ImgType, typename DataType, int DataLength>
int perform_grabcut_on( const Mat& img, Mat& mask, int iterCount, double epsilon)
{
	// Shrink our image and mask
	Mat bgdModel = Mat(); // Our local model
	Mat fgdModel = Mat(); // Same as above

	// Building GMMs for local models
	GMM<DataType, DataLength> bgdGMM( bgdModel ), fgdGMM( fgdModel );
	Mat compIdxs( img.size(), CV_32SC1 );
	
	// BREAK: Program breaks on initGMMs if the area is extremely small - K means algorythm breaks
	initGMMs< ImgType, DataType, DataLength >( img, mask, bgdGMM, fgdGMM );

	// Simple parameters of our algorythm, used for setting up edge flows
	const double gamma = 50; // Gamma seems to be just a parameter for lambda, here 50
	const double lambda = 9*gamma; // Lambda is simply a max value for flow, be it from source or to target, here 450
	const double beta = calcBeta< ImgType, DataLength >( img ); // Beta is a parameter, here 1/(2*avg(sqr(||color[i] - color[j]||)))
												// 1 / 2*average distance in colors between neighbours

	// NWeights, the flow capacity of our edges
	Mat leftW, upleftW, upW, uprightW;
	calcNWeights< ImgType, DataType, DataLength >( img, leftW, upleftW, upW, uprightW, beta, gamma );

	Mat prev(mask.rows, mask.cols, CV_8UC1);
	int total_iters = 0;
	unsigned int A, B, C;
	// The main loop
	do {
		// Save the previously calculated mask
		mask.copyTo(prev);
		
		// Simply initialize the graph we will be using throughout the algorythm. It is created empty
		GCGraph<double> graph;
		
		// Check the image at mask, and depending if it's FGD or BGD, return the number of component that suits it most.
		// Answer (component numbers) is stored in compIdxs, it does not store anything else
		assignGMMsComponents< ImgType, DataType, DataLength >( img, mask, bgdGMM, fgdGMM, compIdxs );

		// This one adds samples to proper GMMs based on best component
		// Strengthens our predictions?
		learnGMMs< ImgType, DataType, DataLength >( img, mask, compIdxs, bgdGMM, fgdGMM );

		// NOTE: As far as I can tell these two will be primarily worked upon
		// Construct GraphCut graph, including initializing s and t values for source and sink flows
		constructGCGraph< ImgType, DataType, DataLength >( img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph );

		// Using max flow algorythm calculate segmentation
		estimateSegmentation( graph, mask );

		// Calculate stopping check for pixels which:
		// A - were previously FGD (left)
		// B - are currently FGD (right)
		// C - were or are FGD (center)
		A = B = C = 0;
		Point p;
		for (p.y = 0; p.y < mask.rows; ++p.y)
			for (p.x = 0; p.x < mask.cols; ++p.x)
				if (prev.at<uchar>(p) == GC_PR_FGD)
				{
					++A;
					if (mask.at<uchar>(p) == GC_PR_FGD)
						{++C; ++B;}
				}
				else if (mask.at<uchar>(p) == GC_PR_FGD)
					++B;
		// Update iterCount and total_iters
		iterCount = max(iterCount-1, -1);
		total_iters++;
	}
	while (iterCount != 0 && ((1.0 - (double)C/(A+B-C)) > epsilon));
	return total_iters;
}

int one_step_grabcut(InputArray _img, InputArray _mask, OutputArray _output_mask,
		InputArray _key, std::string& toLog, int iterCount, double epsilon)
{
	// Standard null checking procedure
	if( _img.empty() )
		CV_Error( CV_StsBadArg, "image is empty" );
	if( _img.type() != CV_8UC3 )
		CV_Error( CV_StsBadArg, "image mush have CV_8UC3 type" );

	// Load image
	Mat img = _img.getMat();
	Mat mask = _mask.getMat();
	Mat key = _key.getMat();
	
	// Load ground truth and output array
	Mat& output_mask = _output_mask.getMatRef();
	mask.copyTo( output_mask );

	// Load and check the mask
	checkMask( img, output_mask );

	// Fill the log file with proper messages, since one-step does not perform them
	toLog = toLog + "NA" + ";"; // Expansion
	toLog = toLog + "NA" + ";"; // Convolution
	toLog = toLog + "NA" + ";"; // Homology vector length
	toLog = toLog + "NA" + ";"; // Shrinkage
	toLog = toLog + "NA" + ";"; // Iteration Time 1
	toLog = toLog + "NA" + ";"; // Enlargement

	// Perform grabcut
	int total_iters = 0;
	for (int i = 0; i < iterCount; ++i)
	{
		clock_t start = clock();
		total_iters += perform_grabcut_on<uchar, double, 3>(img, output_mask, 1, epsilon);
		clock_t finish = clock();
		double it_time2 = (((double)(finish - start)) / CLOCKS_PER_SEC);
	}

	// Save and return output
	return total_iters;
}

int two_step_grabcut( InputArray _img, InputArray _mask, InputArray _filters, 
		OutputArray _output_mask, InputArray _key, std::string& toLog, std::string& path,
		int pcaCount, int iterCount, double epsilon, int by )
{	
	// Standard null checking procedure
	if( _img.empty() )
		CV_Error( CV_StsBadArg, "image is empty" );
	if( _img.type() != CV_8UC3 )
		CV_Error( CV_StsBadArg, "image mush have CV_8UC3 type" );
	
	// Load the input mats
	Mat img = _img.getMat();
	Mat mask = _mask.getMat();
	Mat filters = _filters.getMat();
	Mat key = _key.getMat();
	Mat& output_mask = _output_mask.getMatRef();
	checkMask( img, mask );
	mask.copyTo(output_mask); // Required for expandedShrunkMask function between grabcuts

	// Initialization
	clock_t start = clock();
	Mat* img_cg = grey_and_expand( img ); //17 CHANNELS Dimensional Grey
	clock_t finish = clock();
	double expansion_time = (((double)(finish - start)) / CLOCKS_PER_SEC);
	toLog = toLog + toString( expansion_time ) + ";";

	// Applying filters
	start = clock();
	Mat img_cg_v[CHANNELS]; // Vector of values for filter2D usage
	Mat filters_v[FILTERS]; // Vector of filters for filter2D usage
	split( *img_cg, img_cg_v );
	split( filters, filters_v );
	for (int i = 0; i < FILTERS; i++)
	{
		// Apply the filter. Default values are:
		// Point(-1,-1) (center of filter), delta=0.0, border handling is REFLECT_101
		filter2D( img_cg_v[i+4], img_cg_v[i+4], CV_64F, filters_v[i] );
	}
	// Build back our final solution
	merge( img_cg_v, CHANNELS, *img_cg );
	finish = clock();
	double convolution_time = (((double)(finish - start)) / CLOCKS_PER_SEC);
	toLog = toLog + toString( convolution_time ) + ";";

	// Homology vector length
	toLog = toLog + "NA;";
	
	// Shrink the image and mask to get 34 channels
	start = clock();
	Mat* img_dc = shrink( *img_cg, mask, by ); // Image double channels (shrunk)
	finish = clock();
	double shrink_time = (((double)(finish - start)) / CLOCKS_PER_SEC);
	toLog = toLog + toString( shrink_time ) + ";";
	
	// Perform grabcut and save its time for future references
	int total_iters = 0;
	start = clock();
	for (int i = 0; i < iterCount; ++i)
		total_iters += perform_grabcut_on< double, double, 2*CHANNELS >( *img_dc, mask, 1, epsilon );
	finish = clock();
	double it_time1 = (((double)(finish - start)) / CLOCKS_PER_SEC);
	toLog = toLog + toString( it_time1 ) + ";";

	// Resize mask back to original size
	start = clock();
	expandShrunkMat( mask, output_mask, by );
	finish = clock();
	double enlarge_time = (((double)(finish - start)) / CLOCKS_PER_SEC);
	toLog = toLog + toString( enlarge_time ) + ";";

	delete img_cg;
	delete img_dc;
	return total_iters;
}

// Create a single filter in accordance to
// http://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html
void make_filter( Mat& f, const int sup, const int sigma, const int tau, const int which )
{	
	// All calculations done using parallel execution with C++11 lambda.
	// Calculate cos(...)*exp(...) part
	Point p;
	for (p.y = 0; p.y < f.rows; p.y++)
		for (p.x = 0; p.x < f.cols; p.x++)
		{
			Vecf_f& value = f.at<Vecf_f>(p);
			value[which] = cos((float)(value[which]*(M_PI*tau/sigma))) * exp(-(value[which]*value[which])/(2*sigma*sigma));
		}
	//f.forEach<Vecf_f>([&](Vecf_f& value, const int position[]) -> void{
	//	value[which] = cos((float)(value[which]*(M_PI*tau/sigma))) * exp(-(value[which]*value[which])/(2*sigma*sigma));
	//});

	// Calculate mean
	float mean = 0.0;
	for (p.y = 0; p.y < f.rows; p.y++)
		for (p.x = 0; p.x < f.cols; p.x++)
		{
			Vecf_f& value = f.at<Vecf_f>(p);
			mean += value[which];
		}
	//f.forEach<Vecf_f>([&](Vecf_f& value, const int position[]) -> void{
	//	mean += value[which];
	//});
	mean /= sup*sup;

	// f=f-mean(f(:));
	for (p.y = 0; p.y < f.rows; p.y++)
		for (p.x = 0; p.x < f.cols; p.x++)
		{
			Vecf_f& value = f.at<Vecf_f>(p);
			value[which] -= mean;
		}
	//f.forEach<Vecf_f>([&](Vecf_f& value, const int position[]) -> void{
	//	value[which] -= mean;
	//});
	
	// Calculate sum
	float sum = 0.0;
	for (p.y = 0; p.y < f.rows; p.y++)
		for (p.x = 0; p.x < f.cols; p.x++)
		{
			Vecf_f& value = f.at<Vecf_f>(p);
			sum += value[which] > 0 ? value[which] : -value[which];
		}
	//f.forEach<Vecf_f>([&](Vecf_f& value, const int position[]) -> void{
	//	sum += value[which] > 0 ? value[which] : -value[which];
	//});

	// f=f/sum(abs(f(:)))
	for (p.y = 0; p.y < f.rows; p.y++)
		for (p.x = 0; p.x < f.cols; p.x++)
		{
			Vecf_f& value = f.at<Vecf_f>(p);
			value[which] /= sum;
		}
	//f.forEach<Vecf_f>([&](Vecf_f& value, const int position[]) -> void{
	//	value[which] /= sum;
	//});
}

// Create the Schmid filter bank in accordace to
// http://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html
void create_filters(OutputArray _filters, int size)
{
	// Initalize our Material
	Mat& filters = _filters.getMatRef();
	filters.create( Size( size, size ), CV_32FC(13) );

	// Initialize the radius value
	// Parallel execution using C++11 lambda.
	int hsup = (size-1)/2;
	Point p;
	for (p.x = 0; p.x < filters.rows; p.x++)
		for (p.y = 0; p.y < filters.cols; p.y++)
		{
			Vecf_f& value = filters.at<Vecf_f>(p);
			value[0] = sqrt( (float)( (-hsup+p.x)*(-hsup+p.x) + (-hsup+p.y)*(-hsup+p.y) ) );
			for (int i = 1; i < 13; i++)
				value[i] = value[0];
		}
	//filters.forEach<Vecf_f>([&](Vecf_f& value, const int position[]) -> void{
	//	value[0] = sqrt( (float)( (-hsup+position[0])*(-hsup+position[0]) + (-hsup+position[1])*(-hsup+position[1]) ) );
	//	for (int i = 1; i < 13; i++)
	//		value[i] = value[0];
	//});

	// Create all 13 filters
	make_filter( filters, size, 2, 1, 0 );
	make_filter( filters, size, 4, 1, 1 );
	make_filter( filters, size, 4, 2, 2 );
	make_filter( filters, size, 6, 1, 3 );
	make_filter( filters, size, 6, 2, 4 );
	make_filter( filters, size, 6, 3, 5 );
	make_filter( filters, size, 8, 1, 6 );
	make_filter( filters, size, 8, 2, 7 );
	make_filter( filters, size, 8, 3, 8 );
	make_filter( filters, size, 10, 1, 9 );
	make_filter( filters, size, 10, 2, 10 );
	make_filter( filters, size, 10, 3, 11 );
	make_filter( filters, size, 10, 4, 12 );
}

int grabCut(InputArray _img, InputArray _mask, OutputArray _output_mask, double epsilon)
{
	// Standard null checking procedure
	if( _img.empty() )
		CV_Error( CV_StsBadArg, "image is empty" );
	if( _img.type() != CV_8UC3 )
		CV_Error( CV_StsBadArg, "image mush have CV_8UC3 type" );

	// Load image
	Mat img = _img.getMat();
	Mat mask = _mask.getMat();
	
	// Load ground truth and output array
	Mat& output_mask = _output_mask.getMatRef();
	mask.copyTo( output_mask );

	// Load and check the mask
	checkMask( img, output_mask );

	// Perform grabcut
	int total_iters = perform_grabcut_on<uchar, double, 3>(img, output_mask, 1, epsilon);

	// Save and return output
	return total_iters;
}

int roughCut( InputArray _img, InputArray _mask, OutputArray _output_mask,
		double epsilon, int windowSize, InputArray _filters )
{	
	// Standard null checking procedure
	if( _img.empty() )
		CV_Error( CV_StsBadArg, "image is empty" );
	if( _img.type() != CV_8UC3 )
		CV_Error( CV_StsBadArg, "image must have CV_8UC3 type" );
	
	// Load the input mats
	Mat img = _img.getMat();
	Mat mask = _mask.getMat();
	Mat filters;
	if ( _filters.empty() ) {
		static Mat defaultShmidFilters;
		if (defaultShmidFilters.empty()) {
			create_filters( defaultShmidFilters, 49 );
		}
		filters = defaultShmidFilters;
	} else filters = _filters.getMat();
	Mat& output_mask = _output_mask.getMatRef();
	checkMask( img, mask );
	mask.copyTo(output_mask); // Required for expandedShrunkMask function between grabcuts

	// Initialization
	Mat* img_cg = grey_and_expand( img ); //17 CHANNELS Dimensional Grey

	// Applying filters
	Mat img_cg_v[CHANNELS]; // Vector of values for filter2D usage
	Mat filters_v[FILTERS]; // Vector of filters for filter2D usage
	split( *img_cg, img_cg_v );
	split( filters, filters_v );
	for (int i = 0; i < FILTERS; i++)
	{
		// Apply the filter. Default values are:
		// Point(-1,-1) (center of filter), delta=0.0, border handling is REFLECT_101
		filter2D( img_cg_v[i+4], img_cg_v[i+4], CV_64F, filters_v[i] );
	}
	// Build back our final solution
	merge( img_cg_v, CHANNELS, *img_cg );

	// Shrink the image and mask to get 34 channels
	Mat* img_dc = shrink( *img_cg, mask, windowSize ); // Image double channels (shrunk)
	
	// Perform grabcut and save its time for future references
	int total_iters = perform_grabcut_on< double, double, 2*CHANNELS >( *img_dc, mask, 1, epsilon );

	// Resize mask back to original size
	expandShrunkMat( mask, output_mask, windowSize );

	delete img_cg;
	delete img_dc;
	return total_iters;
}

} //namespace cv