#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc/homology_grabcut.hpp"

#include <time.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/thread.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#ifdef _WINDOWS
	#include <Windows.h>
#else
	#include <sys/types.h>
	#include <dirent.h>
	#include <errno.h>
#endif

using namespace std;
using namespace cv;

boost::mutex mtx;
boost::interprocess::interprocess_semaphore sem(30);

enum MODE {
	ONE_STEP	= 0,
	TWO_STEP	= 1,
	HOMOLOGY	= 2,
	THREE_STEP	= 3,
	END			= 4
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
		out.push_back(directory + "/" + dp->d_name);
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
	if ((file.compare("bmp") == 0) || (file.compare("jpg") == 0) || (file.compare("png") == 0) || (file.compare("tif") == 0))
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
			if (answer == GC_BGD || answer == GC_PR_BGD)
				answer = GC_PR_BGD;
			else answer = GC_PR_FGD;

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
double calculateFMeasure(const cv::Mat& output, const cv::Mat& key, int verboseLevel)
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
			if (answer == GC_BGD || answer == GC_PR_BGD)
				answer = GC_PR_BGD;
			else answer = GC_PR_FGD;

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
	double precision = (double)(tp)/(tp+fp);
	double recall = (double)(tp)/(tp+fn);
	double answer = 2.0/(1.0/precision + 1.0/recall);
	//std::cout << tp << "tp " << fp << "fp " << tn << "tn " << fn << "fn\n";
	//std::cout << precision << " " << recall << " " << answer << "\n";
	return answer;
}

void nextIter(const cv::Mat& image, const cv::Mat& image_mask, const cv::Mat& filters,
	cv::Mat& mask, double skelOccup, int iterCount, double epsilon, int verboseLevel, int mode,
	string& toLog, double& accuracy, double& total_time, double& it_time1, double& it_time2, int& total_iters,
	string test, int by)
{
	// Perform grabcut and measure it's time and number of iterations
	clock_t start = clock();

	if (mode == ONE_STEP)
		total_iters = one_step_grabcut( image, image_mask, mask, iterCount, epsilon);
	else if (mode == TWO_STEP)
		total_iters = two_step_grabcut( image, image_mask, filters,
						mask, it_time1, it_time2, test,
						2048, iterCount/2, epsilon, by );
	else if (mode == HOMOLOGY)
		total_iters = homology_grabcut( image, image_mask,
						mask, it_time1, it_time2,
						iterCount/2, epsilon, by );

	clock_t finish = clock();
	total_time = (((double)(finish - start)) / CLOCKS_PER_SEC);

	// Calculate and save accuracy
	//accuracy = calculateAccuracy( mask, image_mask, verboseLevel, toLog );
	accuracy = calculateFMeasure( mask, image_mask, verboseLevel );
}

void swapToInput( Mat& mask )
{
	Point p;
	for( p.y = 0; p.y < mask.rows; p.y++ )
	{
		for( p.x = 0; p.x < mask.cols; p.x++ )
		{
			if ( (int)mask.at<uchar>(p) == 0)
				mask.at<uchar>(p) = GC_PR_BGD;
			else mask.at<uchar>(p) = GC_FGD;
		}
	}
	if (mask.at<uchar>(0, 0) == GC_PR_BGD)
		mask.at<uchar>(0, 0) = GC_BGD;
	if (mask.at<uchar>(mask.rows-1, 0) == GC_PR_BGD)
		mask.at<uchar>(mask.rows-1, 0) = GC_BGD;
	if (mask.at<uchar>(0, mask.cols-1) == GC_PR_BGD)
		mask.at<uchar>(0, mask.cols-1) = GC_BGD;
	if (mask.at<uchar>(mask.rows-1, mask.cols-1) == GC_PR_BGD)
		mask.at<uchar>(mask.rows-1, mask.cols-1) = GC_BGD;
}
void swapToValues( Mat& mask )
{
	Point p;
	for( p.y = 0; p.y < mask.rows; p.y++ )
	{
		for( p.x = 0; p.x < mask.cols; p.x++ )
		{
			if ( (int)mask.at<uchar>(p) == GC_PR_BGD || (int)mask.at<uchar>(p) == GC_BGD )
				mask.at<uchar>(p) = 0;
			else mask.at<uchar>(p) = 255;
		}
	}
}

// Read mask from Weizmann's database
void readMask( Mat& mask )
{
	Point p;
	for( p.y = 0; p.y < mask.rows; p.y++ )
	{
		for( p.x = 0; p.x < mask.cols; p.x++ )
		{
			Vec3b& v = mask.at<Vec3b>(p);
			if (v[0] == 255 && v[1] == 0 && v[2] == 0) // Red, is mask
				v[0] = v[1] = v[2] = 255;
			else if (v[0] == 0 && v[1] == 0 && v[2] == 255) // Blue, is mask
				v[0] = v[1] = v[2] = 255;
			else v[0] = v[1] = v[2] = 0;
		}
	}
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
		//readMask( image_mask );

		// Save image data and enlarge image/mask if needed be
		int scale = 5;
		std::string original_size = toString(image.rows) + "x" + toString(image.cols);
		if (scale > 1)
		{
			resize(image, image, image.size()*scale);
			resize(image_mask, image_mask, image_mask.size()*scale, 0, 0, 1);
		}

		// Transform mask to one channel binary image
		cvtColor(image_mask, image_mask, COLOR_RGB2GRAY);
		threshold(image_mask, image_mask, 1.0, 255, THRESH_BINARY);
		int mask_count = countNonZero( image_mask );
		swapToInput( image_mask );

		// Initialize values for program and for logging
		int iterCount = 4;
		double epsilon = 0.05;
		int total_iters;

		fstream logFile;
		string toLog = "";
		string original = "";
		original = original + getFileName( source );
		int id = start_id;

		// Create filters
		Mat filters;
		create_filters(filters);

		// Output mask
		Mat mask;
		mask.create(image.rows, image.cols, CV_8UC1);

		for(int skelOccup = 5; skelOccup > 0; --skelOccup)
		{
			// Erode the mask to achieve % of original mask pixels
			swapToValues( image_mask );
			while ((double)countNonZero( image_mask )/mask_count > (double)skelOccup/10)
				erode( image_mask, image_mask, Mat(), Point(-1, -1), 1 );
			//std::cout << "SkelOccup: " << toString( (double)countNonZero( image_mask )/mask_count ) << "\n";
			swapToInput( image_mask );

			double accuracy, it_time1, it_time2, total_time;
			accuracy = total_time = it_time1 = it_time2 = 0.0;

			for (int mode = ONE_STEP; mode < HOMOLOGY; ++mode)
			{
				// Perform iteration
				cout << "Begining loop for " << original << " with " << skelOccup << ", " << mode << endl;
				nextIter(image, image_mask, filters,
					mask, (double)skelOccup/10, iterCount, epsilon, 0, mode,
					toLog, accuracy, total_time, it_time1, it_time2, total_iters,
					out_path + "_t/" + original + "_TWO_so" + toString((float)skelOccup/10.0));
				swapToValues( mask );

				// Save calculated image mask
				string output_file = out_path + "/" + original + (mode == ONE_STEP ? "_ONE" : (mode == TWO_STEP ? "_TWO" : (mode == HOMOLOGY ? "_HOM" : "_THREE")))
					+ "_so" + toString((float)skelOccup/10.0) + ".png";
				imwrite( output_file, mask );
				// Update log string
				if (mode == ONE_STEP)
					toLog = toLog + toString((float)id) + ";" + toString((float)skelOccup/10) + ";" + toString((float)mode) + ";" +
						toString((float)accuracy) + ";" + toString((float)total_iters) + ";" +
						toString((float)total_time) + ";0.0;0.0;" +
						getFileName(source) + ";" + original_size + ";" + toString(scale) + ";" + output_file + "\n";
				else
					toLog = toLog + toString((float)id) + ";" + toString((float)skelOccup/10) + ";" + toString((float)mode) + ";" +
						toString((float)accuracy) + ";" + toString((float)total_iters) + ";" +
						toString((float)total_time) + ";" + toString((float)it_time1) + ";" + toString((float)it_time2) + ";" +
						getFileName(source) + ";" + original_size + ";" + toString(scale) + ";" + output_file + "\n";
				++id;
			}
		}
		// Initalize log file
		{
			boost::lock_guard<boost::mutex> lock(mtx);
			logFile.open( logFileName.c_str(), ios::out | ios::ate | ios::app );
			for (unsigned int i = 0; i < toLog.length(); i++)
				logFile.write(&toLog.at(i), 1);
			logFile.close();
		}
		sem.post();
	}

private:
	const std::string logFileName;
	const std::string source;
	const std::string mask;
	const std::string out_path;
	const int start_id;
};
class WorkerMats {
public:
	WorkerMats( const std::string &_logFileName, const std::string &_out_path, Mat& _image, Mat& _mask,
		const int _start_id, const int _scale, const int _by) : logFileName(_logFileName), out_path(_out_path), image(_image), image_mask(_mask),
		start_id(_start_id), scale(_scale), by(_by) {};

	void operator()() {
		// Transform mask to one channel binary image
		int mask_count = countNonZero( image_mask );
		swapToInput( image_mask );

		// Initialize values for program and for logging
		int iterCount = 4;
		double epsilon = 0.05;
		int total_iters;

		fstream logFile;
		string toLog = "";
		string original = "";
		int id = start_id;

		// Create filters
		Mat filters;
		create_filters(filters);

		// Output mask
		Mat mask;
		mask.create(image.rows, image.cols, CV_8UC1);

		for(int skelOccup = 5; skelOccup > 0; --skelOccup)
		{
			// Erode the mask to achieve % of original mask pixels
			swapToValues( image_mask );
			while ((double)countNonZero( image_mask )/mask_count > (double)skelOccup/10)
				erode( image_mask, image_mask, Mat(), Point(-1, -1), 1 );
			//std::cout << "SkelOccup: " << toString( (double)countNonZero( image_mask )/mask_count ) << "\n";
			swapToInput( image_mask );

			double accuracy, it_time1, it_time2, total_time;
			accuracy = total_time = it_time1 = it_time2 = 0.0;

			for (int mode = ONE_STEP; mode < THREE_STEP; ++mode)
			{
				// Set original filename to image id
				original = toString( id );

				// Perform iteration
				cout << "Begining loop for " << original << " with " << skelOccup << ", " << mode << endl;
				nextIter(image, image_mask, filters,
					mask, (double)skelOccup/10, iterCount, epsilon, 0, mode,
					toLog, accuracy, total_time, it_time1, it_time2, total_iters,
					out_path + "_t/" + original + "_TWO_so" + toString((float)skelOccup/10.0), by);
				swapToValues( mask );

				// Save calculated image mask
				string output_file = out_path + "/" + original + (mode == ONE_STEP ? "_ONE" : (mode == TWO_STEP ? "_TWO" : (mode == HOMOLOGY ? "_HOM" : "_THREE")))
					+ "_so" + toString((float)skelOccup/10.0) + ".png";
				imwrite( output_file, mask );

				// Update log string
				std::string original_size = toString(image.cols) + "x" + toString(image.rows);
				if (mode == ONE_STEP)
					toLog = toLog + toString((float)id) + ";" + toString((float)skelOccup/10) + ";" + toString((float)mode) + ";" +
						toString((float)accuracy) + ";" + toString((float)total_iters) + ";" +
						toString((float)total_time) + ";0.0;0.0;" +
						original_size + ";" + toString(scale) + ";" + output_file + "\n";
				else
					toLog = toLog + toString((float)id) + ";" + toString((float)skelOccup/10) + ";" + toString((float)mode) + ";" +
						toString((float)accuracy) + ";" + toString((float)total_iters) + ";" +
						toString((float)total_time) + ";" + toString((float)it_time1) + ";" + toString((float)it_time2) + ";" +
						original_size + ";" + toString(scale) + ";" + output_file + "\n";
				++id;
			}
		}
		// Initalize log file
		{
			boost::lock_guard<boost::mutex> lock(mtx);
			logFile.open( logFileName.c_str(), ios::out | ios::ate | ios::app );
			for (unsigned int i = 0; i < toLog.length(); i++)
				logFile.write(&toLog.at(i), 1);
			logFile.close();
		}
		sem.post();
	}

private:
	const std::string logFileName;
	const std::string out_path;
	const int start_id;
	const int scale;
	const int by;
	Mat& image;
	Mat& image_mask;
};

class ImageGenerator {
private:
	std::vector< std::string > shapes;
	std::vector< std::string > textures;
	
public:
	ImageGenerator(std::string _shapes, std::string _textures)
	{
		// Load shapes database
		GetFilesInDirectory(shapes, _shapes);
		for (std::vector<std::string>::iterator i = shapes.begin(); i != shapes.end(); i)
		{
			if (!isTypeGraphic( getFileType( *i )))
				i = shapes.erase(i);
			else ++i;
		}

		// Load texture database
		GetFilesInDirectory(textures, _textures);
		for (std::vector<std::string>::iterator i = textures.begin(); i != textures.end(); i)
		{
			if (!isTypeGraphic( getFileType( *i )))
				i = textures.erase(i);
			else ++i;
		}
	}

	// pair< Image, Mask >
	std::pair< Mat, Mat > generateImage( int width, int height, int object_count, int seed)
	{
		// Make sure random works correctly and initialize vectors
		RNG rng( seed );
		std::vector< Mat > objects;
		std::vector< Mat > object_textures;
		std::vector< unsigned int > texture_ids;

		// Create our image...
		Mat image;
		image.create( height, width, CV_8UC3 );
		image.setTo( Scalar(0, 0, 0) );

		// ...and fill it with objects and textures
		for (int i = 0; i < object_count; ++i)
		{
			// Randomize shape
			int current = rng.next()%shapes.size();
			Mat object = imread( shapes[current] );

			// Get affine transformation
			Point2f src[3] = { Point2f(0.f, 0.f), Point2f( object.cols-1, 0.f ), Point2f( 0.f, object.rows-1 ) };
			Point2f dst[3];
			// Create points so that the object is between 1/16 and 1/4 of the size of the entire image
			dst[0] = Point2f( rng.next()%(image.rows*3/4), rng.next()%(image.cols*3/4) );
			// Do ming our images are square, so the following random number can be used for both, width and height
			float added = max(rng.next()%image.rows/2, image.rows/4);
			dst[1] = Point2f( dst[0].x + added, dst[0].y );
			dst[2] = Point2f( dst[0].x, dst[0].y + added );

			Mat transform = cv::getAffineTransform(src, dst);
			Mat object_warped;
			warpAffine( object, object_warped, transform, image.size() );
			threshold( object_warped, object_warped, 125.0, i+1, THRESH_BINARY );

			// Save created object matrix
			objects.push_back( object_warped );
		}

		// Create the mask and randomize mask id
		Mat mask;
		mask.create( height, width, CV_8UC1 );
		mask.setTo( Scalar(0) );
		unsigned int selected_object = objects.size()-1;

		// Write our objects into the image, taking care to pick the one on top
		Point p;
		for (p.y = 0; p.y < image.rows; ++p.y)
			for (p.x = 0; p.x < image.cols; ++p.x)
			{
				Vec3b& max = image.at<Vec3b>(p);
				for (int i = 0; i < object_count; ++i)
				{
					Vec3b val = objects[i].at<Vec3b>(p);
					// Only the first one is needed as only this one will be checked
					max[0] = max( max[0], val[0] );
				}
				// 0 = background. Textures start from 1
				if ((unsigned int)max[0] == (selected_object+1))
					mask.at<uchar>(p) = 255;
			}

		// Randomize background texture
		int current = rng.next()%textures.size();
		Mat texture = imread( textures[current] );
		object_textures.push_back( texture );
		texture_ids.push_back( current );

		// Fill our vector with randomized textures
		for (int i = 0; i < object_count; ++i)
		{
			// Randomize texture. It cannot be any of the already previously randomized
			bool check = true;
			while (check)
			{
				current = rng.next()%textures.size();
				check = false;
				for (int j = 0; j < texture_ids.size(); ++j)
					if (texture_ids[j] == current)
						check = true;
			}

			// Get the new texture and save it along with its id
			texture = imread( textures[current] );
			object_textures.push_back( texture );
			texture_ids.push_back( current );
		}

		// Finish creating our image by properly copying textures into their designated places
		for (p.y = 0; p.y < image.rows; ++p.y)
			for (p.x = 0; p.x < image.cols; ++p.x)
			{
				Vec3b& max = image.at<Vec3b>(p);
				Vec3b val = object_textures[max[0]].at<Vec3b>(p.y % object_textures[max[0]].rows,
																p.x % object_textures[max[0]].cols);
				max[0] = val[0];
				max[1] = val[1];
				max[2] = val[2];
			}

		// Done, simply return the data
		return std::pair< Mat, Mat >(image, mask); 
	}
};

int main( int argc, char** argv )
{
	/*
	char* path = argc >= 3 ? argv[2] : (char*)"./bin/databases/tools";
	std::vector<std::string> sources;
	GetFilesInDirectory(sources, path);
	for (std::vector<std::string>::iterator i = sources.begin(); i != sources.end(); i)
	{
		if (!isTypeGraphic( getFileType( *i )))
			i = sources.erase(i);
		else ++i;
	}
	for (std::vector<std::string>::iterator i = sources.begin(); i != sources.end(); ++i)
	{
		Mat img = imread( *i );
		Point p;
		for (p.y = 0; p.y < img.rows; p.y++)
			for (p.x = 0; p.x < img.cols; p.x++)
			{
				Vec3b& val = img.at<Vec3b>(p);
				if (val[0] == 0)
				{
					val[0] = 255;
					val[1] = 255;
					val[2] = 255;
				}
				else
				{
					val[0] = 0;
					val[1] = 0;
					val[2] = 0;
				}
			}
		std::string filename = "./bin/databases/shapes/" + getFileName( *i ) + ".png";
		imwrite( filename, img );
	}
	*/

	char* log_path = argc >= 2 ? argv[1] : (char*)"./bin/log.csv";
	char* shapes = argc >= 3 ? argv[2] : (char*)"./bin/databases/shapes";
	char* textures = argc >= 4 ? argv[3] : (char*)"./bin/databases/textures";
	char* out_path = argc >= 5 ? argv[4] : (char*)"./bin/test";

	ImageGenerator generator( shapes, textures );

	srand(time(NULL));
	pair< Mat, Mat > data = generator.generateImage( 1024, 1024, 5, rand() );

	imwrite( "./bin/test/test.png", data.first );
	imwrite( "./bin/test/test_mask.png", data.second );

	WorkerMats w( log_path, out_path, data.first, data.second, 0, 1, 8);
	w();

	/*
	// Initialize paths
	char* log_path = argc >= 2 ? argv[1] : (char*)"./bin/log.csv";
	
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
	char* out_path = argc >= 5 ? argv[4] : (char*)"./bin/answers";
	
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

	// Creating work threads
	int id = 0;
	boost::thread_group threads;
	for (std::vector<std::pair<int, int> >::iterator i = pairs.begin(); i != pairs.end(); ++i)
	{
		sem.wait();
		Worker w( log_path, sources.at( i->first ), masks.at( i->second ), out_path, id );
		threads.create_thread( w );
		//w();
		id += 15;
	}
	threads.join_all();
	*/
	return 0;
}
