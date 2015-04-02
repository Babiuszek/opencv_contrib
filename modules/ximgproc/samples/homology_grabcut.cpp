#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc/homology_grabcut.hpp"

#include <time.h>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

ofstream logFile;

double calculateAccuracy(const cv::Mat& output, const cv::Mat& key, int verboseLevel, string& toLog)
{
	int tp = 0;
	int tn = 0;
	int fp = 0;
	int fn = 0;
	int wv = 0;
	for (int i = 0; i < output.rows; i++)
		for (int j = 0; j < output.cols; j++)
		{
			int value = (int)output.at<uchar>(i, j);
			int answer = (int)key.at<uchar>(i, j);
			
			if (value == GC_BGD || value == GC_PR_BGD)
				value = GC_PR_BGD;
			else value = GC_PR_FGD;
			answer = answer / 255 + 2;

			if (value == GC_PR_FGD && answer == GC_PR_FGD)
				tp++;
			else if (value == GC_PR_BGD && answer == GC_PR_BGD)
				tn++;
			else if (value == GC_PR_FGD && answer == GC_PR_BGD)
				fp++;
			else if (value == GC_PR_BGD && answer == GC_PR_FGD)
				fn++;
			else
			{
				wv++;
				if (verboseLevel > 2)
					std::cout << "Wrong value " << value << " or answer " << answer << "\n";
			}
		}
	double answer = (double)(tp+tn)/(tp+tn+fp+fn);
	if (verboseLevel > 0)
		std::cout << "Accuracy: " << answer << "\n";
	//ostringstream oss;
	//oss << "\t\tAccuracy=" << answer << "\n";
	//toLog = toLog + oss.str();
	if (verboseLevel > 1)
	{
		std::cout << "True positives: " << tp << "\n";
		std::cout << "True negatives: " << tn << "\n";
		std::cout << "False positives: " << fp << "\n";
		std::cout << "False negatives: " << fn << "\n";
		std::cout << "Wrong values: " << wv << "\n";
	}
	return answer;
}

void nextIter(const cv::Mat& image, const cv::Mat& image_mask, const cv::Mat& image_mask_skel, const cv::Mat& filters,
	cv::Mat& mask, double thresh, int iteration, int verboseLevel, string& toLog, double& accuracy, double& it_time)
{
    if( iteration > 0 )
	{
		Mat temp;
		mask.copyTo(temp);
		
		clock_t start = clock();
		homology_grabcut( image, temp, filters, mask, thresh, rand() );
		clock_t finish = clock();
		it_time = (((double)(finish - start)) / CLOCKS_PER_SEC);
		if (verboseLevel > 0)
			std::cout << "Answer for iteration " << iteration << " found in " << it_time << " seconds.\n";
		//ostringstream oss;
		//oss << "\t\tAnswer for iteration " << iteration << " found in " << it_time << " seconds.\n";
		//toLog = toLog + oss.str();

		accuracy = calculateAccuracy( mask, image_mask, verboseLevel, toLog );
	}
    else
    {
		clock_t start = clock();
		homology_grabcut( image, image_mask_skel, filters, mask, thresh, rand() );
		clock_t finish = clock();
		it_time = (((double)(finish - start)) / CLOCKS_PER_SEC);
		if (verboseLevel > 0)
			std::cout << "Answer for iteration " << iteration << " found in " << it_time << " seconds.\n";
		//ostringstream oss;
		//oss << "\t\tAnswer for iteration " << iteration << " found in " << it_time << " seconds.\n";
		//toLog = toLog + oss.str();

		accuracy = calculateAccuracy( mask, image_mask, verboseLevel, toLog );
	}
}

int main( int argc, char** argv )
{
	// Randomization init
	srand((unsigned int)time(NULL));

	// Initialize paths
	string out_path = "./answers/";

	// Load the image
	char* filename = argc >= 3 ? argv[2] : (char*)"grabcut_cow.png";
    Mat image = imread( filename, 1 );
    if( image.empty() )
    {
        cout << "\n Durn, couldn't read image filename " << filename << endl;
        return 1;
    }
	
	// Save original filename and begin log string
	string toLog = "";
	string original = "";
	//toLog = toLog + filename + "\n";
	original = original + filename;
	original.substr(0, original.length()-4);

	// Load the mask
	filename = argc >= 4 ? argv[3] : (char*)"grabcut_cow_mask.png";
	Mat image_mask = imread( filename, 1 );
    if( image_mask.empty() )
    {
        cout << "\n Durn, couldn't read image filename " << filename << endl;
        return 1;
    }
	cvtColor(image_mask, image_mask, COLOR_RGB2GRAY);
	threshold(image_mask, image_mask, 10, 255, THRESH_BINARY);

	// Load skel'd mask
	filename = argc >= 5 ? argv[4] : (char*)"grabcut_cow_mask_skel.png";
	Mat image_mask_skel = imread( filename, 1 );
    if( image_mask_skel.empty() )
    {
		skel( image_mask, image_mask_skel );
		imwrite( "grabcut_cow_mask_skel.png", image_mask_skel );
    }
	cvtColor(image_mask_skel, image_mask_skel, COLOR_RGB2GRAY);
	threshold(image_mask_skel, image_mask_skel, 10, 255, THRESH_BINARY);

	// Create filters
	Mat filters;
	create_filters(filters);

	// Output mask
	Mat mask;
	mask.create(image.rows, image.cols, CV_8UC1);

    for(int thresh = 1; thresh <= 10; ++thresh)
    {
		double accuracy, it_time, total_time;
		accuracy = it_time = total_time = 0.0;
		Mat answer, bin_mask;
		bin_mask.create( mask.size(), CV_8UC1 );
		
		//ostringstream oss;
		//oss << "\tThreshold=" << (double)thresh/10 << "\n";
		//toLog = toLog + oss.str();
		for (int i = 0; i < 2; i++)
		{
			// Perform iteration
			nextIter(image, image_mask, image_mask_skel, filters, mask, (double)thresh/10, i, 1, toLog, accuracy, it_time);
			cv::threshold(mask, mask, 2.5, 255, THRESH_BINARY);

			// Save answer
			total_time += it_time;
			bin_mask = mask & 1;
			image.copyTo( answer, bin_mask );
			//oss = ostringstream();
			//oss << "_thresh" << (double)thresh/10 << "_it" << i << "_ac" << accuracy << "_time" << total_time << ".png";
			string output_file = out_path + original + ".png"; //+ oss.str();
			imwrite( output_file, answer );
		}
		//oss = ostringstream();
		//oss << "\t\tTotal time=" << total_time << "\n";
		//toLog = toLog + oss.str();
    }
	// Initalize log file
	filename = argc >= 2 ? argv[1] : (char*)"log.txt";
	logFile.open( filename, ios::out | ios::ate | ios::app );
	for (unsigned int i = 0; i < toLog.length(); i++)
		logFile.write(&toLog.at(i), 1);
	logFile.close();

    return 0;
}
