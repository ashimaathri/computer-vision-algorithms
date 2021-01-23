#include <stdio.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat computeScaling(Mat image, Mat homography);
Mat changePerspective(Mat image, Mat homography);
Mat computeHomography(std::vector <Point2f> from, std::vector <Point2f> to);
void write(char* nameOfOriginal, Mat image, string suffix =
    string("-corrected"));
void write(string nameOfOriginal, Mat image, std::string suffix = "");
void display(Mat image);
void receivePointCorrespondence(std::vector <Point2f> &points, const
    std::string &type);
Mat readHomogeneousCoordinates();
Mat getNullSpace(void* linearSystem, int numRows, int numCols,
    int finalRows = 3);
void printVector(vector <float> v);
Mat normalize(Mat coordinates);
Mat choleskyDecomposition(Mat matrix);
Mat choleskyUpperDecomp(Mat matrix);
Mat fullDecomp(Mat symmetricMat);
Mat getConjugate(Mat point);
Mat complexMatMul(Mat matrix1, Mat matrix2);
Mat solveConicSystem(vector<Mat> points);
Mat getConic();
Mat intersectLineAndConic(Mat conic, Mat P, Mat Q);
Mat transformConic(Mat pointTransform, Mat conic);
vector<tuple<KeyPoint, vector<uint8_t>>> harris_stephens_corners(Mat image, float k, float threshold);
Mat create_composite_image(Mat image1, Mat image2);
Point2f transform_point(Mat homography, float x, float y);
Vec3b interpolate(Mat image, Point2f point);
vector<tuple<KeyPoint, KeyPoint>> getMatches(
    vector<tuple<KeyPoint, vector<uint8_t>>> corners1,
    vector<tuple<KeyPoint, vector<uint8_t>>> corners2);
