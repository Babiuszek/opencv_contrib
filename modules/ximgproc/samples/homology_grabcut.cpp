#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc/homology_grabcut.hpp"

#include <iostream>
#include <time.h>
#include <fstream>

using namespace std;
using namespace cv;

fstream logFile;

enum MODE {
	ONE_STEP	= 0,
	TWO_STEP	= 1,
	END			= 2
};

int getID(string& line)
{
	// Get the first number (ID)
	int l = 0;
	while (line.at(l) != ';')
		++l;
	line = line.substr(0, l);

	// Transform it into an integer
	int Answer = 0;
	for (unsigned int i = 0; i < line.length(); ++i)
		Answer = Answer*10 + line.at(i) - 48;
	return Answer;
}

string toString(float value)
{
	// Init
	string Ans = "";
	char c;

	// Negative
	if (value < 0.f)
	{
		Ans = Ans + "-";
		value = -value;
	}

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

	// Fraction
	string fraction = "";
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
	cv::Mat& mask, double skelOccup, int iterCount, double epsilon, int verboseLevel, int mode,
	string& toLog, double& accuracy, double& it_time, int& total_iters)
{
	// Perform grabcut and measure it's time and number of iterations
	clock_t start = clock();
	if (mode == ONE_STEP)
		total_iters = one_step_grabcut( image, image_mask, image_mask, mask, skelOccup, rand(), iterCount, epsilon);
	else if (mode == TWO_STEP)
		total_iters = two_step_grabcut( image, image_mask, filters, image_mask, mask, skelOccup, rand(), iterCount, epsilon );
	clock_t finish = clock();
	it_time = (((double)(finish - start)) / CLOCKS_PER_SEC);

	// Calculate and save accuracy
	accuracy = calculateAccuracy( mask, image_mask, verboseLevel, toLog );
}

int main( int argc, char** argv )
{
	// Randomization init
	srand((unsigned int)time(NULL));

	// Initialize paths
	string out_path = "./answers/";
	int iterCount = 0;
	double epsilon = 0.001;
	int total_iters;

	// Load the image
	char* filename = argc >= 3 ? argv[2] : (char*)"grabcut_cow.png";
    Mat image = imread( filename, 1 );
    if( image.empty() )
    {
        cout << "\n Durn, couldn't read image filename " << filename << endl;
        return 1;
    }
	
	// Initialize values for logging
	string toLog = "";
	string original = "";
	original = original + filename;
	original = original.substr(0, original.length()-4);
	int id = 0;

	// Get the last id used in our csv file
	filename = argc >= 2 ? argv[1] : (char*)"log.csv";
	logFile.open( filename, ios::in );
	string line;
	getline( logFile, line );
	while (!line.empty())
	{
		id = getID( line ) + 1;
		getline( logFile, line );
	}
	logFile.close();

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

    for(int skelOccup = 1; skelOccup <= 5; ++skelOccup)
    {
		double accuracy, it_time;
		accuracy = it_time = 0.0;
		Mat bin_mask;
		bin_mask.create( mask.size(), CV_8UC1 );

		for (int mode = ONE_STEP; mode < END; mode++)
		{
			// Perform iteration
			nextIter(image, image_mask, image_mask_skel, filters,
				mask, (double)skelOccup/10, iterCount, epsilon, 1, mode,
				toLog, accuracy, it_time, total_iters);
			cv::threshold(mask, mask, 2.5, 255.0, THRESH_BINARY);

			// Save calculated image mask
			string output_file = out_path + original + "_" + toString((float)id) + ".png";
			imwrite( output_file, mask );
			// Update log string
			toLog = toLog + toString((float)id) + ";" + toString((float)skelOccup/10) + ";" + toString((float)mode) + ";" +
				toString((float)accuracy) + ";" + toString((float)total_iters) + ";" +
				toString((float)it_time) + ";" + output_file + "\n";
			++id;
		}
    }
	// Initalize log file
	filename = argc >= 2 ? argv[1] : (char*)"log.csv";
	logFile.open( filename, ios::out | ios::ate | ios::app );
	for (unsigned int i = 0; i < toLog.length(); i++)
		logFile.write(&toLog.at(i), 1);
	logFile.close();

    return 0;
}
