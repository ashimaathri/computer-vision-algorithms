#include "homography.hpp"
#include <opencv2/core.hpp>
using namespace cv;
using namespace std;

void getOrthogonalLines(vector<Mat> &orthogonalLines) {
  cout << "Enter points of 5 pairs of orthogonal lines" << endl;

  for(int i = 0; i < 10; i++) {
    Mat point1 = readHomogeneousCoordinates();
    Mat point2 = readHomogeneousCoordinates();
    orthogonalLines.push_back(normalize(point1.cross(point2)));
  }
}

Mat solveOrthogonalSystem(vector<Mat> lines) {
  int numLines = lines.size();

  assert(numLines % 10 == 0);
  int numSets = numLines / 10;
  int numRows = 5 * numSets;
  int numCols = 6;

  float linearSystem[numRows][numCols];

  for(int k = 0; k < numSets; k++) {
    float subSystem[5][6];
    for(int i = 0, row = 0; i < 10; i += 2, row++) {
      const float l1 = lines[i].at<float>(0, 0);
      const float l2 = lines[i].at<float>(1, 0);
      const float l3 = lines[i].at<float>(2, 0);
      const float m1 = lines[i + 1].at<float>(0, 0);
      const float m2 = lines[i + 1].at<float>(1, 0);
      const float m3 = lines[i + 1].at<float>(2, 0);
      subSystem[row][0] = l1 * m1;
      subSystem[row][1] = (l1 * m2 + l2 * m1) / 2;
      subSystem[row][2] = l2 * m2;
      subSystem[row][3] = (l1 * m3 + l3 * m1) / 2;
      subSystem[row][4] = (l2 * m3 + m3 * m2) / 2;
      subSystem[row][5] = l3 * m3;
    }
    int sizeOfSubMatrix = 5 * 6 * sizeof(float);
    memcpy(linearSystem + k * sizeOfSubMatrix, subSystem, sizeOfSubMatrix);
  }
  return getNullSpace((void*)&linearSystem, numRows, numCols, 1);
}

Mat rectify(Mat image, Mat KKt, Mat KKtv, Mat (*decomp)(Mat), string name) {
  cout << "Performing one step rectification using " << name << " decomposition" << endl;
  Mat K = decomp(KKt);
  Mat Ktv = K.inv() * KKtv;

  Mat homography = (Mat_<float>(3, 3) << K.at<float>(0, 0), K.at<float>(0, 1),
      0, K.at<float>(1, 0), K.at<float>(1, 1), 0, Ktv.at<float>(0, 0),
      Ktv.at<float>(1, 0), 1);
  return changePerspective(image, homography * computeScaling(image, homography));
}

Mat makePositiveDefinite(Mat conicCoefficients) {
  float a = conicCoefficients.at<float>(0, 0);
  float b = conicCoefficients.at<float>(0, 1) / 2;
  float c = conicCoefficients.at<float>(0, 2);
  float d = conicCoefficients.at<float>(0, 3) / 2;
  float e = conicCoefficients.at<float>(0, 4) / 2;
  float f = conicCoefficients.at<float>(0, 5);

  Mat conic = (Mat_<float>(3, 3) << a, b, d, b, c, e, d, e, f);

  bool condition0 = a > 0;
  bool condition1 = a * c > b * b;
  bool condition2 = determinant(conic) > 0;

  bool isPositiveDefinite = condition0 && condition1 && condition2;

  if(isPositiveDefinite) {
    return conicCoefficients;
  } else {
    if(!condition1 || !(determinant(-conic) > 0)) {
      std::cerr << "Conic can not be made positive definite by negation and the algorithm will not work! " << endl;
      abort();
    }
    cout << "Negating coefficients to ensure positive definiteness..." << endl;
    return -conicCoefficients;
  }
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

  vector<Mat> orthogonalLines;
  getOrthogonalLines(orthogonalLines);

  Mat S = makePositiveDefinite(solveOrthogonalSystem(orthogonalLines));

  float a = S.at<float>(0, 0);
  float b = S.at<float>(0, 1) / 2;
  float c = S.at<float>(0, 2);
  float d = S.at<float>(0, 3) / 2;
  float e = S.at<float>(0, 4) / 2;
  float f = S.at<float>(0, 5);

  Mat KKt = (Mat_<float>(2, 2) << a, b, b, c);
  Mat KKtv = (Mat_<float>(2, 1) << d, e);

  // What is your favorite decomposition?
  write(argv[1], rectify(image, KKt, KKtv, choleskyDecomposition, "lower cholesky"),
      string("-one-step-cholesky-lower"));
  write(argv[1], rectify(image, KKt, KKtv, fullDecomp, "svd"),
      string("-one-step-svd"));
  write(argv[1], rectify(image, KKt, KKtv, choleskyUpperDecomp, "upper cholesky"),
      string("-one-step-cholesky-upper"));
  return 0;
}
