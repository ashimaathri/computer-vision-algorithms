#include <mvg.h>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("usage: match_features <Image1> <Image2>\n");
    return -1;
  }

  Mat image1 = imread(argv[1]);
  Mat image2 = imread(argv[2]);

  Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();

  vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;
  detector->detectAndCompute(image1, noArray(), keypoints1, descriptors1);
  detector->detectAndCompute(image2, noArray(), keypoints2, descriptors2);

  Mat result;
  drawKeypoints(image1, keypoints1, result, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS); 
  write(argv[1], result, "-sift-corners");
  drawKeypoints(image2, keypoints2, result, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS); 
  write(argv[2], result, "-sift-corners");

  BFMatcher matcher = BFMatcher(NORM_L2, true);
  vector<DMatch> matches;
  matcher.match(descriptors1, descriptors2, matches);

  cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, result);

  char name[5] = "sift";
  display(result);
  write(name, result, "-matches");
  return 0;
}
