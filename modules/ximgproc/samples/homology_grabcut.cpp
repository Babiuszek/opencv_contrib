#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc/homology_grabcut.hpp"

#include <time.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/thread.hpp>
#ifdef _WINDOWS
	#include <Windows.h>
#else
	#include <pthread.h>
	#include <sys/types.h>
	#include <dirent.h>
	#include <errno.h>
#endif

using namespace std;
using namespace cv;

//boost::mutex mtx;

enum MODE {
	ONE_STEP	= 0,
	TWO_STEP	= 1,
	END			= 2
};

/* Returns a list of files in a directory (except the ones that begin with a dot) */

void GetFilesInDirectory(std::vector<std::string> &out, const std::string &directory)
{
#ifdef _WINDOWS
    HANDLE dir;
    WIN32_FIND_DATA file_data;

    if ((dir = FindFirstFile((directory + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE)
        return; /* No files found */

    do {
        const string file_name = file_data.cFileName;
        const string full_file_name = directory + "/" + file_name;
        const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

        if (file_name[0] == '.')
            continue;

        if (is_directory)
            continue;

        out.push_back(full_file_name);
    } while (FindNextFile(dir, &file_data));

    FindClose(dir);
#else
	DIR* dirp = opendir(directory.c_str());
	struct dirent *dp;
	while ((dp = readdir(dirp)) != NULL)
	{
		if (dp->d_name[0] == '.')
			continue;
		if (dp->d_type == DT_DIR)
			continue;
		out.push_back(dp->d_name);
	}
	(void)closedir(dirp);
#endif
} // GetFilesInDirectory

std::string getFileType(const std::string &file)
{
	std::string ans = "";
	for (unsigned int i = file.length()-1; i > 0; --i)
		if (file.at(i) != '.')
			ans = file.at(i) + ans;
		else break;
	return ans;
}
std::string getFileName(const std::string &file)
{
	std::string ans = "";
	unsigned int i;
	// Find the dot
	for (i = file.length()-1; i > 0; --i)
		if (file.at(i) == '.')
			break;
	// Save the filename until '/' or '\'
	for (i = i-1; i > 0; --i) //skip the dot
	{
		if ((file.at(i) == '\\') || file.at(i) == '/')
			break;
		else ans = file.at(i) + ans;
	}
	return ans;
}
bool isTypeGraphic(const std::string &file)
{
	if ((file.compare("bmp") == 0) || (file.compare("jpg") == 0) || (file.compare("png") == 0))
		return true;
	return false;
}

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
	if (value == 0.f)
		return "0.0";
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

void nextIter(const cv::Mat& image, const cv::Mat& image_mask, const cv::Mat& filters,
	cv::Mat& mask, double skelOccup, int iterCount, double epsilon, int verboseLevel, int mode,
	string& toLog, double& accuracy, double& total_time, double& it_time1, double& it_time2, int& total_iters)
{
	// Perform grabcut and measure it's time and number of iterations
	clock_t start = clock();
	if (mode == ONE_STEP)
		total_iters = one_step_grabcut( image, image_mask, image_mask, mask, skelOccup, rand(), iterCount, epsilon);
	else if (mode == TWO_STEP)
		total_iters = two_step_grabcut( image, image_mask, filters, image_mask, mask, it_time1, it_time2,
										skelOccup, rand(), iterCount, epsilon );
	clock_t finish = clock();
	total_time = (((double)(finish - start)) / CLOCKS_PER_SEC);

	// Calculate and save accuracy
	accuracy = calculateAccuracy( mask, image_mask, verboseLevel, toLog );
}

class Worker {
public:
	Worker( const std::string &_logFileName, const std::string &_source, const std::string &_mask,
		const std::string &_out_path, const int _start_id) :
		logFileName(_logFileName), source(_source), mask(_mask), out_path(_out_path), start_id(_start_id) {};

	void operator()() {
		// Randomization init
		srand((unsigned int)time(NULL));

		// Load the images
		Mat image = imread( source, 1 );
		Mat image_mask = imread( mask, 1 );

		// Transform mask to one channel binary image
		cvtColor(image_mask, image_mask, COLOR_RGB2GRAY);
		threshold(image_mask, image_mask, 1.0, 255, THRESH_BINARY);
	
		// Initialize values for program and for logging
		int iterCount = 4;
		double epsilon = 0.05;
		int total_iters;

		fstream logFile;
		string toLog = "";
		string original = "";
		original = original + getFileName( source );
		int id = 0;

		// Create filters
		Mat filters;
		create_filters(filters);

		// Output mask
		Mat mask;
		mask.create(image.rows, image.cols, CV_8UC1);

		for(int skelOccup = 2; skelOccup < 6; ++skelOccup)
		{
			double accuracy, it_time1, it_time2, total_time;
			accuracy = total_time = it_time1 = it_time2 = 0.0;
			Mat bin_mask;
			bin_mask.create( mask.size(), CV_8UC1 );

			for (int mode = ONE_STEP; mode < END; mode++)
			{
				// Perform iteration
				cout << "Begining loop for " << original << " with " << skelOccup << ", " << mode << endl;
				nextIter(image, image_mask, filters,
					mask, (double)skelOccup/10, iterCount, epsilon, 1, mode,
					toLog, accuracy, total_time, it_time1, it_time2, total_iters);
				cv::threshold(mask, mask, 2.5, 255.0, THRESH_BINARY);

				// Save calculated image mask
				string output_file = out_path + "/" + original + (mode == ONE_STEP ? "_ONE" : (mode == TWO_STEP ? "_TWO" : "_HOM"))
					+ "_so" + toString((float)skelOccup/10.f) + ".png";
				imwrite( output_file, mask );
				// Update log string
				if (mode == ONE_STEP)
					toLog = toLog + toString((float)id) + ";" + toString((float)skelOccup/10) + ";" + toString((float)mode) + ";" +
						toString((float)accuracy) + ";" + toString((float)total_iters) + ";" +
						toString((float)total_time) + ";" + toString(0.f) + ";" + toString(0.f) + ";" + output_file + "\n";
				else
					toLog = toLog + toString((float)id) + ";" + toString((float)skelOccup/10) + ";" + toString((float)mode) + ";" +
						toString((float)accuracy) + ";" + toString((float)total_iters) + ";" +
						toString((float)total_time) + ";" + toString(it_time1) + ";" + toString(it_time2) + ";" + output_file + "\n";
				++id;
			}
		}
		// Initalize log file
		{
			//boost::lock_guard<boost::mutex> lock(mtx);
			logFile.open( logFileName.c_str(), ios::out | ios::ate | ios::app );
			for (unsigned int i = 0; i < toLog.length(); i++)
				logFile.write(&toLog.at(i), 1);
			logFile.close();
		}
	}

private:
	const std::string logFileName;
	const std::string source;
	const std::string mask;
	const std::string out_path;
	const int start_id;
};

int main( int argc, char** argv )
{
	// Initialize paths
	char* log_path = argc >= 2 ? argv[1] : (char*)"log.csv";
	
	// Load images from given (or default) source folder
	char* filename = argc >= 3 ? argv[2] : (char*)"./bin/images/sources";
	std::vector<std::string> sources;
	GetFilesInDirectory(sources, filename);
	for (std::vector<std::string>::iterator i = sources.begin(); i != sources.end(); i)
	{
		if (!isTypeGraphic( getFileType( *i )))
			i = sources.erase(i);
		else ++i;
	}
	
	// Load images from given (or default) mask folder
	filename = argc >= 4 ? argv[3] : (char*)"./bin/images/ground_truths";
	std::vector<std::string> masks;
	GetFilesInDirectory(masks, filename);
	for (std::vector<std::string>::iterator i = masks.begin(); i != masks.end(); i)
	{
		if (!isTypeGraphic( getFileType( *i )))
			i = masks.erase(i);
		else ++i;
	}

	// And last but not least the out path
	char* out_path = argc >= 5 ? argv[4] : (char*)"./answers";
	
	// Pairing up sources with their masks
	std::vector<std::pair<int, int> > pairs;
	for (unsigned int i = 0; i < sources.size(); ++i)
	{
		std::string source = getFileName( sources.at(i) );
		for (unsigned int j = 0; j < masks.size(); ++j)
			if (source.compare( getFileName( masks.at(j) ) ) == 0)
			{
				pairs.push_back( std::pair<int, int>(i, j) );
				break;
			}
			else if (j == masks.size()-1)
			{
				cout << "\n Could not find a pairing mask for image " << sources.at(i) << endl;
				cout << "Make sure the mask has the same filename as the image and it's extension is lower case.\n";
				return 2;
			}
	}
	
	//Enlarging database images
	for (std::vector<std::pair<int, int> >::iterator i = pairs.begin(); i != pairs.end(); ++i)
	{
		Mat A = imread( sources.at(i->first) ,1 );
		resize( A, A, A.size()*5, 0, 0, 1 );
		imwrite( "bin/images/sources_enlarged/" + getFileName( sources.at(i->first) ) + ".png", A );
		
		Mat B = imread( masks.at(i->second) ,1 );
		resize( B, B, B.size()*5, 0, 0, 1 );
		imwrite( "bin/images/ground_truths_enlarged/" + getFileName( masks.at(i->second) ) + ".png", B );
	}
	// Create threads
	/*int id = 1;
	for (std::vector<std::pair<int, int> >::iterator i = pairs.begin(); i != pairs.end(); ++i)
	{
		Worker w( log_path, sources.at( i->first ), masks.at( i->second ), out_path, id );
		w();
		id += 10;
	}*/
	/*boost::thread_group threads;
	for (std::vector<std::pair<int, int> >::iterator i = pairs.begin(); i != pairs.end(); ++i)
	{
		Worker w( log_path, sources.at( i->first ), masks.at( i->second ), out_path, id );
		boost::thread thread( w );
		thread.join();
	}*/
	/*for (std::vector<std::pair<int, int> >::iterator i = pairs.begin(); i != pairs.end(); ++i)
	{
		Worker w( log_path, sources.at( i->first ), masks.at( i->second ), out_path, id );
		threads.create_thread( w );
		id += 10;
	}
	threads.join_all();*/

    return 0;
}
