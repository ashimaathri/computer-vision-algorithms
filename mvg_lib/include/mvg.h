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
vector<KeyPoint> harris_stephens_corners(Mat image, float k, float threshold);
Mat create_composite_image(Mat image1, Mat image2);
float ssd(Mat image1, Point2f point1, Mat image2, Point2f point2);
vector<tuple<KeyPoint, KeyPoint>> get_matches(
    Mat image1,
    vector<KeyPoint> corners1,
    Mat image2,
    vector<KeyPoint> corners2,
    float(*distance)(Mat, Point2f, Mat, Point2f));
