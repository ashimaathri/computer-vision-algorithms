#include <mvg.h>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

void receivePointCorrespondences(std::vector <Point2f> &points1,
    std::vector <Point2f> &points2,
    std::vector <Point2f> &points3,
    std::vector <Point2f> &points4) {
  receivePointCorrespondence(points1, "first");
  receivePointCorrespondence(points2, "second");
  receivePointCorrespondence(points3, "third");
  receivePointCorrespondence(points4, "fourth");
}

void copyPerspectively(Mat &to, Mat from, Mat homography, std::vector <Point2f> points) {
  int toHeight = to.size().height;
  int toWidth = to.size().width;
  int fromHeight = from.size().height;
  int fromWidth = from.size().width;

  Mat world = Mat(toHeight, toWidth, to.type());
  Mat mapX = Mat(toHeight, toWidth, CV_32FC1);
  Mat mapY = Mat(toHeight, toWidth, CV_32FC1);
  for(int x = 0; x < toWidth; x++) {
    for(int y = 0; y < toHeight; y++) {
      const float homogeneousPoint[3] = {(float)x, (float)y, 1};
      Mat imagePoint = homography * Mat(3, 1, CV_32FC1, (void*)&homogeneousPoint);
      float imageZ = imagePoint.at<float>(2, 0);
      float imageX = imagePoint.at<float>(0, 0)/imageZ;
      float imageY = imagePoint.at<float>(1, 0)/imageZ;
      if(imageX > 0 && imageY > 0 && imageX < toWidth && imageY < toHeight) {
        mapX.at<float>(y, x) = imageX;
        mapY.at<float>(y, x) = imageY;
      } else {
        // Setting the x and y values in the maps to a value outside the image
        // causes them to be treated as "outliers" in the image and the default
        // BORDER_CONSTANT flag maps outliers to 0
        mapX.at<float>(y, x) = toWidth + 1;
        mapY.at<float>(y, x) = toHeight + 1;
      }
    }
  }
  remap(from, world, mapX, mapY, INTER_LINEAR);
  Mat mask(toHeight, toWidth, CV_8UC1, cv::Scalar(0));
  vector<vector<Point>> contours;
  contours.push_back(vector<Point>());
  contours[0].push_back(points[0]);
  contours[0].push_back(points[1]);
  contours[0].push_back(points[3]);
  contours[0].push_back(points[2]);
  drawContours(mask, contours, 0, Scalar(255), FILLED, 8);
  world.copyTo(to, mask);
}

int main(int argc, char** argv )
{
  if (argc != 5) {
    printf("usage: DisplayImage.out <Image_Path1> <Image_Path2> <Image_Path3> <Image_Path4>\n");
    return -1;
  }

  Mat image1 = imread(argv[1], 1);
  Mat image2 = imread(argv[2], 1);
  Mat image3 = imread(argv[3], 1);
  Mat image4 = imread(argv[4], 1);
  if(!image1.data || !image2.data || !image3.data || !image4.data) {
    printf("No image data \n");
    return -1;
  }

  std::vector <Point2f> points1;
  std::vector <Point2f> points2;
  std::vector <Point2f> points3;
  std::vector <Point2f> points4;
  receivePointCorrespondences(points1, points2, points3, points4);

  Mat homography12 = computeHomography(points2, points1);
  Mat homography23 = computeHomography(points3, points2);
  Mat homography41 = computeHomography(points1, points4);
  Mat homography42 = computeHomography(points2, points4);
  Mat homography43 = computeHomography(points3, points4);

  //Mat correctedImage13 = changePerspective(image1, homography23 * homography12);

  //display(correctedImage13);
  //write(argv[1], correctedImage13);
  copyPerspectively(image1, image4, homography41, points1);
  write(argv[1], image1);
  copyPerspectively(image2, image4, homography42, points2);
  write(argv[2], image2);
  copyPerspectively(image3, image4, homography43, points3);
  write(argv[3], image3);
  return 0;
}
