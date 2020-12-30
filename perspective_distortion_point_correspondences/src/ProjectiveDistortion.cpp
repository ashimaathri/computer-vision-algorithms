#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <mvg.h>
using namespace cv;
using namespace std;

void receivePointCorrespondences(std::vector <Point2f> &srcPoints, std::vector <Point2f> &dstPoints) {
  receivePointCorrespondence(srcPoints, "original");
  receivePointCorrespondence(dstPoints, "correctedImage");
}

int main(int argc, char** argv )
{
  if (argc != 2) {
    printf("usage: DisplayImage.out <Image_Path>\n");
    return -1;
  }

  Mat image = imread(argv[1], 1);
  if(!image.data) {
    printf("No image data \n");
    return -1;
  }

  std::vector <Point2f> srcPoints;
  std::vector <Point2f> dstPoints;
  receivePointCorrespondences(srcPoints, dstPoints);

  Mat homography = computeHomography(dstPoints, srcPoints);
  /* Note: Scaling only works in some cases as there is no guarantee that the
   * mapping of the extent of the image into the world plane will contain *all*
   * mapped pixels. This can be observed, for instance, if we apply the
   * homography to the ground plane instead of the wall plane.
   * Mat scaling = computeScaling(image, homography);
   * Mat correctedImage = changePerspective(image, homography * scaling);
   */ 
  Mat correctedImage = changePerspective(image, homography);

  display(correctedImage);
  write(argv[1], correctedImage);
  return 0;
}
