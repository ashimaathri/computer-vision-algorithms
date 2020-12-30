#include <mvg.h> 
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/mat.hpp>
#include "math.h"
using namespace cv;
using namespace std;

Mat rectifyUptoAffinity(Mat image, Mat vanishingLine) {
  cout << "Calculating matrix that rectifies image upto an affinity..." << endl;
  Mat transform = (Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0,
      vanishingLine.at<float>(0, 0),
      vanishingLine.at<float>(0, 1),
      vanishingLine.at<float>(0, 2));
  Mat rectification = transform.inv();
  return rectification * computeScaling(image, rectification);
}

Mat solveOrthogonalSystem(vector<Mat> lines) {
  int numLines = lines.size();

  assert(numLines % 4 == 0);
  int numRows = numLines / 2;
  int numCols = 3;

  float linearSystem[numRows][3];

  for(int i = 0; i < numRows / 2; i++) {
    const float subSystem[2][3] = {
      {
        lines[i + 0].at<float>(0, 0) * lines[i + 1].at<float>(0, 0),
        lines[i + 0].at<float>(0, 0) * lines[i + 1].at<float>(1, 0) + lines[i + 0].at<float>(1, 0) * lines[i + 1].at<float>(0, 0),
        lines[i + 0].at<float>(1, 0) * lines[i + 1].at<float>(1, 0)
      },
      {
        lines[i + 2].at<float>(0, 0) * lines[i + 3].at<float>(0, 0),
        lines[i + 2].at<float>(0, 0) * lines[i + 3].at<float>(1, 0) + lines[i + 2].at<float>(1, 0) * lines[i + 3].at<float>(0, 0),
        lines[i + 2].at<float>(1, 0) * lines[i + 3].at<float>(1, 0)
      },
    };
    int sizeOfSubMatrix = 2 * 3 * sizeof(float);
    memcpy(linearSystem + i * sizeOfSubMatrix, subSystem, sizeOfSubMatrix);
  }
  return getNullSpace((void*)&linearSystem, numRows, numCols);
}

Mat rectifyUptoSimilarity(Mat image, Mat S, Mat (*decomp)(Mat), string name) {
  cout << "Rectifying image upto a similarity using " << name << " decomposition..." << endl;
  Mat K = decomp(S);
  Mat homography = (Mat_<float>(3, 3) << K.at<float>(0, 0), K.at<float>(0, 1),
      0, K.at<float>(1, 0), K.at<float>(1, 1), 0, 0, 0, 1);
  return changePerspective(image, homography * computeScaling(image, homography));
}

void readOrthogonalLines(vector<vector<float>> &orthogonalLines) {
  cout << "Enter orthogonal line coordinates from the affinely rectified image" << endl;
  for(int i = 0; i < 4; i++) {
    orthogonalLines.push_back(readHomogeneousCoordinates());
  }
}

void getOrthogonalLines(vector<Mat> &orthogonalLines, Mat homography) {
  cout << "Enter point coordinates of orthogonal line pairs from the original non-rectified image" << endl;

  vector<Mat> points;
  for(int i = 0; i < 6; i++) {
    points.push_back(homography.inv() * readHomogeneousCoordinates());
  }

  orthogonalLines.push_back(normalize(points[0].cross(points[1])));
  orthogonalLines.push_back(normalize(points[1].cross(points[2])));
  orthogonalLines.push_back(normalize(points[3].cross(points[4])));
  orthogonalLines.push_back(normalize(points[4].cross(points[5])));
}

int main(int argc, char** argv ) {
  if (argc != 2) {
    printf("usage: %s <Image_Path>\n", argv[0]);
    return -1;
  }

  Mat image = imread(argv[1], 1);
  if(!image.data) {
    printf("No image data \n");
    return -1;
  }

  cout << "Enter vanishing line coordinates" << endl;
  Mat vanishingLine = readHomogeneousCoordinates();

  // Step 1
  Mat affineRectification = rectifyUptoAffinity(image, vanishingLine);
  Mat affinelyRectified = changePerspective(image, affineRectification);
  write(argv[1], affinelyRectified, std::string("-two-step-affine"));

  // Step 2
  vector<Mat> orthogonalLines;
  getOrthogonalLines(orthogonalLines, affineRectification);
  Mat S = solveOrthogonalSystem(orthogonalLines);
  Mat S_matrix = (Mat_<float>(2, 2) << S.at<float>(0, 0), S.at<float>(0, 1),
      S.at<float>(0, 1), S.at<float>(0, 2));

  // There's always more than one way to do something...
  write(argv[1], rectifyUptoSimilarity(affinelyRectified, S_matrix,
        choleskyDecomposition, "lower cholesky"),
      string("-two-step-cholesky-lower"));

  write(argv[1], rectifyUptoSimilarity(affinelyRectified, S_matrix, fullDecomp,
        "svd"),
      string("-two-step-svd"));

  write(argv[1], rectifyUptoSimilarity(affinelyRectified, S_matrix,
        choleskyUpperDecomp, "upper cholesky"),
      string("-two-step-cholesky-upper"));

  return 0;
}
