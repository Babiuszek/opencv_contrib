#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc/homology_grabcut.hpp"

#include <time.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

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

class ImageGenerator {
private:
	std::vector< std::string > shapes;
	std::vector< std::string > textures;
	
public:
	struct Data {
		Mat image;
		Mat mask;

		Data(Mat _image, Mat _mask) : image(_image), mask(_mask) {}
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
			
			// Handle scale
			dst[0].x += (1.0f - scale_x) * image.cols * 3/4;
			dst[0].y += (1.0f - scale_y) * image.rows * 3/4;
			dst[1].x -= (1.0f - scale_x) * image.cols * 3/4;
			dst[1].y += (1.0f - scale_y) * image.rows * 3/4;
			dst[2].x += (1.0f - scale_x) * image.cols * 3/4;
			dst[2].y -= (1.0f - scale_y) * image.rows * 3/4;
			
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
		return Data(image, mask);
	}

private:
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
};

class RCApplication
{
public:
	RCApplication();
	void destroy();
	void setAndShowImage(Mat* mat, int which);
	void setGeneratorShapes(char* _shapes);
	void setGeneratorTextures(char* _textures);

	void generateImageAndGroundTruth(int height, int width, int objects, int seed);
	void generateSeed();
	void performRoughCut(double epsilon, int windowSize);
	void calculateF1Score();
	
	void status();
	void reset();
	void enableImShow();
	void disableImShow();
	int getKey();
private:
	static const int size = 4;
	string windowNames[size];
	Mat* images[size];
	// image, ground truth, seed, segmentation

	char* shapes;
	char* textures;
	ImageGenerator* generator;

	Mat* filters;
	
	double f1Score;
	bool imShowEnabled;
	
	void swapToInput(Mat* mat);
	void swapToValues(Mat* mat);
};

RCApplication::RCApplication()
{
	windowNames[0] = "Image";
	windowNames[1] = "Ground Truth";
	windowNames[2] = "Seed";
	windowNames[3] = "Segmentation";
	for (int i = 0; i < size; ++i) {
		images[i] = NULL;
	}

	shapes = NULL;
	textures = NULL;
	generator = NULL;
	
	filters = new Mat();
	create_filters(*filters);
	f1Score = 0.0;

	imShowEnabled = true;
	for (int i = 0; i < size; ++i) {
		namedWindow( windowNames[i], WINDOW_AUTOSIZE );
	}
}

void RCApplication::destroy()
{
	if (imShowEnabled) {
		for (int i = 0; i < size; ++i) {
			 destroyWindow( windowNames[i] );
		}
	}
	for (int i = 0; i < size; ++i) {
		if (images[i] != NULL)
			delete images[i];
	}
	
	if (generator != NULL)
		delete generator;
	if (shapes != NULL)
		delete[] shapes;
	if (textures != NULL)
		delete[] textures;

	delete filters;
}

void RCApplication::setAndShowImage(Mat* mat, int which)
{
	if (images[which] != NULL)
		delete images[which];
	images[which] = mat;
	if (imShowEnabled) {
		imshow( windowNames[which], *images[which] );
	}
}

void RCApplication::setGeneratorShapes(char* _shapes)
{
	shapes = _shapes;
	if (textures != NULL)
		generator = new ImageGenerator( shapes, textures );
}

void RCApplication::setGeneratorTextures(char* _textures)
{
	textures = _textures;
	if (shapes != NULL)
		generator = new ImageGenerator( shapes, textures );
}

void RCApplication::generateImageAndGroundTruth(int height, int width, int objects, int seed)
{
	if (generator == NULL)
		return;

	ImageGenerator::Data data = (*generator).generateImage( height, width, objects, seed );

	Mat* imageP = new Mat();
	data.image.copyTo( *imageP );
	setAndShowImage( imageP, 0 );
	
	Mat* maskP = new Mat();
	data.mask.copyTo( *maskP );
	setAndShowImage( maskP, 1 );
}

void RCApplication::generateSeed()
{
	if (images[1] == NULL)
		return;
	
	cv::Mat* seed = new Mat();
	(*images[1]).copyTo(*seed);
	const int maskCount = countNonZero( *images[1] );
	
	while ((double)countNonZero( *seed )/(double)maskCount > 0.1)
		erode( *seed, *seed, Mat(), Point(-1, -1), 1 );
	
	setAndShowImage( seed, 2 );
}

void RCApplication::performRoughCut(double epsilon, int windowSize)
{
	if (images[2] == NULL)
		return;

	Mat* segmentation = new Mat();
	swapToInput(images[2]);

	//TODO Get rid of filters, move them inside, maybe as additional parameter used only if given
	roughCut( *images[0], *images[2], *filters, *segmentation);

	swapToValues(images[2]);
	swapToValues(segmentation);
	setAndShowImage( segmentation, 3 );
}

void RCApplication::calculateF1Score()
{
	if (images[3] == NULL)
		return;

	swapToInput(images[1]);
	swapToInput(images[3]);

	int tp = 0;
	int tn = 0;
	int fp = 0;
	int fn = 0;
	int wv = 0;
	for (int i = 0; i < (*images[3]).rows; i++)
		for (int j = 0; j < (*images[3]).cols; j++)
		{
			int value = (int)(*images[3]).at<uchar>(i, j);
			int answer = (int)(*images[1]).at<uchar>(i, j);
			
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
				std::cout << "Wrong value " << value << " or answer " << answer << "\n";
			}
		}
	double precision = (double)(tp)/(tp+fp);
	double recall = (double)(tp)/(tp+fn);
	f1Score = 2.0/(1.0/precision + 1.0/recall);
	
	swapToValues(images[1]);
	swapToValues(images[3]);
}

void RCApplication::status()
{
	cout << "Application status:\n";
	if (images[0] != NULL)
		cout << "\tImage:\t\tOK\n";
	else cout << "\tImage:\t\t-\n";
	if (images[1] != NULL)
		cout << "\tGround truth:\tOK\n";
	else cout << "\tGround truth:\t-\n";
	if (images[2] != NULL)
		cout << "\tSeed:\t\tOK\n";
	else cout << "\tSeed:\t\t-\n";
	if (images[3] != NULL)
		cout << "\tSegmentation:\tOK\n";
	else cout << "\tSegmentation:\t-\n";
	if (f1Score != 0.0)
		cout << "\tF1Score:\t" << f1Score << "\n";
	else cout << "\tF1Score:\t-\n";
}

void RCApplication::reset()
{
	Mat disp(512, 512, CV_8UC3, Scalar(0, 0, 0));
	for (int i = 0; i < size; ++i) {
		if (imShowEnabled) {
			imshow( windowNames[i], disp );
		}
		if (images[i] != NULL) {
			delete images[i];
			images[i] = NULL;
		}
	}
	f1Score = 0.0;
}

void RCApplication::swapToInput(Mat* mat)
{
	Point p;
	for( p.y = 0; p.y < (*mat).rows; p.y++ )
	{
		for( p.x = 0; p.x < (*mat).cols; p.x++ )
		{
			if ( (int)(*mat).at<uchar>(p) == 0)
				(*mat).at<uchar>(p) = GC_PR_BGD;
			else (*mat).at<uchar>(p) = GC_FGD;
		}
	}
}
void RCApplication::swapToValues(Mat* mat)
{
	Point p;
	for( p.y = 0; p.y < (*mat).rows; p.y++ )
	{
		for( p.x = 0; p.x < (*mat).cols; p.x++ )
		{
			if ( (int)(*mat).at<uchar>(p) == GC_PR_BGD || (int)(*mat).at<uchar>(p) == GC_BGD )
				(*mat).at<uchar>(p) = 0;
			else (*mat).at<uchar>(p) = 255;
		}
	}
}

void RCApplication::enableImShow()
{
	if (imShowEnabled)
		return;

	imShowEnabled = true;
	for (int i = 0; i < size; ++i) {
		namedWindow( windowNames[i], WINDOW_AUTOSIZE );
		if (images[i] != NULL) {
			imshow( windowNames[i], *images[i] );
		}
	}
}

void RCApplication::disableImShow()
{
	if (!imShowEnabled)
		return;

	imShowEnabled = false;
	for (int i = 0; i < size; ++i) {
		 destroyWindow( windowNames[i] );
	}
}

int RCApplication::getKey()
{
	if (imShowEnabled) {
		return cv::waitKey(0);
	} else {
		cin.clear();
		cin.sync();
        return getchar();
	}
}

static void help(RCApplication& app)
{
    cout << "\nThis program demonstrates RoughCut algorithm\n"
            "Provide an image and its seed for RoughCut to perform segmentation on it.\n"
        "\nHot keys: \n"
        "\te/q - quit the program\n"
        "\ti - select image\n"
		"\tg - select ground truth\n"
		"\ts - set generator shapes\n"
		"\tt - set generator textures\n"
        "\t1 - generate random image/ground truth\n"
        "\t2 - generate seed from ground truth\n"
        "\t3 - perform rough cut\n"
        "\t4 - calculate F1 Score\n"
		"\t9 - enable image show\n"
		"\t0 - disable image show\n" << endl;
	app.status();
}

string getFilePath() {
	cin.clear();
	cin.sync();
	string input_line;
	getline(cin, input_line);
	return input_line;
}

int main( int argc, char** argv )
{
	char* shapes = "./bin/databases/shapes";
	char* textures = "./bin/databases/textures";

	RCApplication app;
	
	app.setGeneratorShapes( shapes );
	app.setGeneratorTextures( textures );

	help(app);

	// Make sure we get correctly random seed
	srand(time(NULL));

    for(;;)
    {
		string path = "";
		char* cstr;

		int c = app.getKey();

        switch( (char) c )
        {
        case 'e':
            cout << "Exiting ..." << endl;
            goto exit_main;
        case 'q':
            cout << "Exiting ..." << endl;
            goto exit_main;
        case 'i':
			cout << "Enter image path: ";
			path = getFilePath();
			cout << "Loading file...\n";
			app.setAndShowImage(&imread( path, 1 ), 0);
            break;
        case 'g':
			cout << "Enter ground truth path: ";
			path = getFilePath();
			cout << "Loading file...\n";
			app.setAndShowImage(&imread( path, 1 ), 1);
            break;
		case '1':
			app.reset();
			cout << "Generating image and ground truth..." << endl;
			app.generateImageAndGroundTruth(512, 512, 5, rand());
			break;
		case '2':
			cout << "Generating seed..." << endl;
			app.generateSeed();
			break;
		case '3':
			cout << "Performing RoughCut..." << endl;
			app.performRoughCut(0.05, 4);
			app.calculateF1Score();
			break;
        }
		system("cls");
		help(app);
    }

exit_main:
	return 0;
}