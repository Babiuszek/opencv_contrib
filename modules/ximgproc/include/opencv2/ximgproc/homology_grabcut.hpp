/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2014, Beat Kueng (beat-kueng@gmx.net), Lukas Vogel, Morten Lysgaard
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#ifndef __HOMOLOGY_GRABCUT_HPP__
#define __HOMOLOGY_GRABCUT_HPP__
#ifdef __cplusplus

#include <string>
#include <opencv2/core.hpp>

namespace cv
{
	/** @brief Performs 1-step (standard) grabcut using given image and mask.

	The function shrinks given mask and skeletizes it. It then randomly chooses a set amount of
	pixels from the shrunk, skeletized mask before enlarging it. Such prepared mask is then used
	by the standard grabcut algorythm until it meets the end condition (either convergeance
	given certain epsilon or maximum number of iterations).

	 */
	CV_EXPORTS_W int one_step_grabcut(InputArray img, InputArray mask, OutputArray output_mask,
		InputArray key, std::string& toLog, int iterCount=1, double epsilon=0.001);

	/** @brief Performs grabcut on shrunk image using given filters and mask.
	
	The function applies all filters given to seperate copies of grayscale transformed input
	image. The images and initial mask are then shrunk 10 times. Finally, a standard version
	of grabcut is used on the shrunk images, giving us shrunken answer which is then enlarged.
	The function returns the amount of iteration it took to found the answer.
	
	@param img Input image. Supported formats: CV_8UC1. Image size must match the one given by
	mask. The image is assumed to be in RGB format.
	
	@param mask Input mask. Supported formats: CV_8UC1. Image size must match the one given by
	input image. An array of values 0 and 255 for initialization, any other value will result
	in wrong answer. The image is assumed to be skeletized mask of input image. To transform
	out mask into next step mask use OpenCV threshold function with a threshold of 2.5.

	@param filters An array of filters for usage of algorythm. Each filter is seperately
	applied to its own version of grayscaled input image. May be empty.

	@param ground_truth Is the actual mask that answer found by the algorithm is compared to.
	
	@param output_mask The answer of the grabcut algorythm is stored here. An array of CV_8UC1
	values which are either GC_PR_FGD or GC_PR_BGD.

	@param SkelOccup Threshold value between 0.0 and 1.0. It controls how many points of input
	mask are taken into account, 1-SkelOccup points are ignored and assumed as background.

	@param seed Seed for initializing RNG class.
	
	@param iterCount The maximum amount of iterations. Each set of iterations uses the same GMMs.
	Default is 1, meaning GMMs are learned, used and immidiately discarded. Setting it to 0
	causes the program to iterate till convergeance with epsilon.

	@param epsilon The threshold value for changes. If accuracy difference between current and
	previous answer is lower than epsilon the algorythm ends.
	 */
	CV_EXPORTS_W int two_step_grabcut(InputArray img, InputArray mask, InputArray filters,
		OutputArray output_mask, InputArray key, std::string& toLog, std::string& path,
		int pcaCount=2048, int iterCount=1, double epsilon=0.001, int by=10);
	
	/** @brief Constructs 13 Schmid filters storing them in CV_32FC(13) Mat bank
	
	The function creates a bank of Schmid filters for further usage in algorythm. One square
	matrix of CV_32FC(13) format and given size is the output.
	
	@param filters Output filters material of CV_32FC(13) format.
	
	@param size of our output, square matrix. Default is 49, as given by original matlab code.
	 */
	CV_EXPORTS_W void create_filters(OutputArray filters, int size=49);

	CV_EXPORTS_W int grabCut(InputArray img, InputArray mask, OutputArray output_mask, double epsilon=0.001);
	
	CV_EXPORTS_W int roughCut(InputArray img, InputArray mask, OutputArray output_mask, double epsilon=0.001, int windowSize=10, InputArray filters=Mat());
}
#endif
#endif
