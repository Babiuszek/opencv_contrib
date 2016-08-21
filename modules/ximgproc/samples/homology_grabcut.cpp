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

#define M_PI           3.14159265358979323846  /* pi */

using namespace std;
using namespace cv;

boost::mutex mtx;
boost::interprocess::interprocess_semaphore sem(30);
int progress = 0;
int total = 0;

int STARTING_SKEL_OCCUP = 1;
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

void nextIter(const cv::Mat& image, const cv::Mat& image_mask, const cv::Mat& key, const cv::Mat& filters,
	cv::Mat& mask, double skelOccup, int iterCount, double epsilon, int verboseLevel, int mode,
	string& toLog, double& accuracy, double& total_time, int& total_iters,
	string test, int by)
{
	// Perform grabcut and measure it's time and number of iterations
	clock_t start = clock();

	if (mode == ONE_STEP) {
		total_iters = one_step_grabcut( image, image_mask, mask, key, toLog, iterCount, epsilon);
	}
	else {
		total_iters = two_step_grabcut( image, image_mask, filters,
						mask, key, toLog, test,
						2048, iterCount/2, epsilon, by );
	}

	clock_t finish = clock();
	total_time = (((double)(finish - start)) / CLOCKS_PER_SEC);

	// Calculate and save accuracy
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

class WorkerMats {
private:
	const std::string logFileName;
	const std::string out_path;
	const int start_id;
	const int scale;
	const int objects;
	const int by;
	std::string description;
	Mat image;
	Mat image_mask;

public:
	WorkerMats( const std::string &_logFileName, const std::string &_out_path, Mat& _image, Mat& _mask,
		const std::string _description, const int _start_id, const int _scale, const int _objects, const int _by) :
		logFileName(_logFileName), out_path(_out_path),
		start_id(_start_id), scale(_scale), objects(_objects), by(_by)
	{
		description = _description;
		_image.copyTo( image );
		_mask.copyTo( image_mask );
	}
	WorkerMats(WorkerMats& wm) :
		logFileName(wm.logFileName), out_path(wm.out_path),
		start_id(wm.start_id), scale(wm.scale), objects(wm.objects), by(wm.by) 
	{
		description = wm.description;
		wm.image.copyTo( image );
		wm.image_mask.copyTo( image_mask );
	}

	void operator()() {
		// Transform mask to one channel binary image
		int mask_count = countNonZero( image_mask );
		swapToInput( image_mask );

		// Create key for testing
		cv::Mat key;
		image_mask.copyTo( key );

		// Initialize values for program and for logging
		int iterCount = 4;
		double epsilon = 0.05;
		int total_iters;

		int id = start_id;
		fstream logFile;
		string toLog = "";
		string original = toString( id );

		// Create filters
		Mat filters;
		create_filters(filters);

		// Output mask
		Mat mask;
		mask.create(image.rows, image.cols, CV_8UC1);

		string output_file = out_path + "/" + original + "img.png";
		imwrite( output_file, image );
		output_file = out_path + "/" + original + "mask.png";
		swapToValues( image_mask );
		imwrite( output_file, image );
		swapToInput( image_mask );

		for(int skelOccup = STARTING_SKEL_OCCUP; skelOccup > 0; --skelOccup)
		{
			// Erode the mask to achieve % of original mask pixels
			swapToValues( image_mask );
			while ((double)countNonZero( image_mask )/mask_count > (double)skelOccup/10)
				erode( image_mask, image_mask, Mat(), Point(-1, -1), 1 );
			//std::cout << "SkelOccup: " << toString( (double)countNonZero( image_mask )/mask_count ) << "\n";
			swapToInput( image_mask );

			double accuracy, it_time1, it_time2, total_time;
			accuracy = total_time = it_time1 = it_time2 = 0.0;

			for (int mode = ONE_STEP; mode < END; ++mode)
			{
				// Set original filename to image id
				original = toString( id );

				// Write progress
				float current_progress = 0.0;
				{
					boost::lock_guard<boost::mutex> lock(mtx);
					current_progress = (float)progress / total;
				}
				cout.setf(ios::fixed);
				cout << "[" << setprecision(2) << current_progress*100.f << "%] - Begining loop for " << original << " with " << skelOccup << ", " << mode << endl;

				// Perform iteration
				toLog += toString(scale) + ";" + toString(objects) + ";" + description +
						toString(by) + ";" + toString(mode) + ";";
				nextIter(image, image_mask, key, filters,
					mask, (double)skelOccup/10, iterCount, epsilon, 0, mode,
					toLog, accuracy, total_time, total_iters,
					out_path + "_t/" + original + "_TWO_so" + toString((float)skelOccup/10.0), by);
				swapToValues( mask );

				// Save calculated image mask
				output_file = out_path + "/" + original + (mode == ONE_STEP ? "_ONE" : "_TWO")
					+ "_so" + toString((float)skelOccup/10.0) + ".png";
				imwrite( output_file, mask );

				// Update log string
				toLog += toString((float)total_time) + ";" + toString((float)skelOccup/10.0) + ";" + toString(id) + "\n";
				
				++id;
				{
					boost::lock_guard<boost::mutex> lock(mtx);
					++progress;
				}
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
};

class ImageGenerator {
private:
	std::vector< std::string > shapes;
	std::vector< std::string > textures;
	
public:
	struct Data {
		Mat image;
		Mat mask;
		std::string description;

		Data(Mat _image, Mat _mask, std::string _description) : image(_image), mask(_mask) {description = _description;}
	};

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
	Data generateImage( const int width, const int height, const int object_count, const int seed)
	{
		// Make sure random works correctly and initialize vectors
		RNG rng( seed );
		std::vector< Mat > objects;
		std::vector< Mat > object_textures;
		std::vector< Mat > object_transformations;
		std::vector< unsigned int > texture_ids;
		std::string description = "";

		// Select a single object that will be the foreground
		unsigned int selected_object = object_count-1;

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

			// Points:
			// 0 1
			// 2 -
			// Get affine transformation
			Point2f src[3] = { Point2f(0.f, 0.f), Point2f( object.cols-1, 0.f ), Point2f( 0.f, object.rows-1 ) };
			Point2f dst[3] = { Point2f(-0.1*image.cols, -0.1*image.rows),
				Point2f( 1.1*image.cols, -0.1*image.rows ), Point2f( -0.1*image.cols, 1.1*image.rows ) };

			// Randomize transformation
			float scale_x = (80.f + (float)(rng.next()%21))/100.f;
			float scale_y = (80.f + (float)(rng.next()%21))/100.f;
			float angle = (rng.next()%360)*M_PI/180.f;
			float translation_x = (int)(rng.next()%image.cols*1/2) - image.cols/4;
			float translation_y = (int)(rng.next()%image.cols*1/2) - image.rows/4;
			
			// Save object transformation
			if (i == selected_object)
				description = description + shapes[current] + ";" + toString(scale_x) + ";" + toString(scale_y) + ";" +
					toString(angle) + ";" + toString(translation_x) + ";" + toString(translation_y) + ";";
			
			// Handle scale
			dst[0].x += (1.0f - scale_x) * image.cols * 3/4;
			dst[0].y += (1.0f - scale_y) * image.rows * 3/4;
			dst[1].x -= (1.0f - scale_x) * image.cols * 3/4;
			dst[1].y += (1.0f - scale_y) * image.rows * 3/4;
			dst[2].x += (1.0f - scale_x) * image.cols * 3/4;
			dst[2].y -= (1.0f - scale_y) * image.rows * 3/4;
			
			// Save location
			if (i == selected_object)
				description = description + toString(dst[0].x + translation_x) + ";" + toString(dst[0].y + translation_y) + ";";

			// Handle rotation - substract pivot point => rotate => add pivot point
			for (int j = 0; j < 3; ++j)
			{
				dst[j].x -= image.cols/2;
				dst[j].y -= image.rows/2;
			}
			for (int j = 0; j < 3; ++j)
			{
				float p_x = cos(angle)*dst[j].x - sin(angle)*dst[j].y;
				float p_y = sin(angle)*dst[j].x + cos(angle)*dst[j].y;
				dst[j].x = p_x;
				dst[j].y = p_y;
			}
			for (int j = 0; j < 3; ++j)
			{
				dst[j].x += image.cols/2;
				dst[j].y += image.rows/2;
			}
			
			// Handle translation
			dst[0].x += translation_x;
			dst[0].y += translation_y;
			dst[1].x += translation_x;
			dst[1].y += translation_y;
			dst[2].x += translation_x;
			dst[2].y += translation_y;

			Mat transform = cv::getAffineTransform(src, dst);
			Mat object_warped;
			warpAffine( object, object_warped, transform, image.size() );
			threshold( object_warped, object_warped, 125.0, i+1, THRESH_BINARY );

			// Save created object matrix
			objects.push_back( object_warped );
		}

		// Create the mask
		Mat mask;
		mask.create( height, width, CV_8UC1 );
		mask.setTo( Scalar(0) );

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
					if (max[0] < val[0])
						max[0] = val[0];
					//max[0] = std::max( max[0], val[0] );
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

			// Save our texture
			if ( i == selected_object )
				description = textures[current] + ";" + description;
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
		return Data(image, mask, description);
	}
};

int main( int argc, char** argv )
{
	char* log_path = argc >= 2 ? argv[1] : (char*)"./bin/log.csv";
	char* shapes = argc >= 3 ? argv[2] : (char*)"./bin/databases/shapes";
	char* textures = argc >= 4 ? argv[3] : (char*)"./bin/databases/textures";
	char* out_path = argc >= 5 ? argv[4] : (char*)"./bin/answers";

	ImageGenerator generator( shapes, textures );

	// Make sure we get correctly random seed
	srand(time(NULL));

	std::map<int, int> cellSizes;
	cellSizes[512] = 8;
	cellSizes[1024] = 16;
	cellSizes[2048] = 32;
	cellSizes[4096] = 64;

	// Set totals: id_step * sizes * square_sizes * images
	total = 10*4*1*100;

	// Create tests for image size 512, 1024, 2048, 4096
	int id = 0;
	boost::thread_group threads;
	for (int i = 1; i < 16; i*=2)
	{
		// For each data set test 100 images
		for (int l = 0; l < 100; ++l)
		{
			// Randomize the amount of objects
			int j = rand()%8 + 1;

			// First generate the actual image
			ImageGenerator::Data data = generator.generateImage( 512*i, 512*i, j, rand() );

			// Then create a thread for it
			sem.wait();
			WorkerMats w( log_path, out_path, data.image, data.mask, data.description, id, 512*i, j, cellSizes[512*i]);
			threads.create_thread( w );
			id += 2;
		}
	}
	threads.join_all();

	return 0;
}
