#include <iostream>
#include <vector>
#include <filesystem>
#include <cmath>

#include <nurbs_rt/nurbs_rt.h>

int main() {
  std::vector<LiteMath::float3> pointsFlatten = {
    {-1.000, 1.000, -0.395}, {-0.600, 1.000, 0.070}, {-0.200, 1.000, 0.070}, {0.200, 1.000, 0.070}, {0.600, 1.000, 0.070}, {1.000, 1.000, 0.070},
    {-1.000, 0.600, -0.070}, {-0.600, 0.600, 1.000}, {-0.200, 0.600, 1.000}, {0.200, 0.600, -0.070}, {0.600, 0.600, -0.070}, {1.000, 0.600, -0.070},
    {-1.000, 0.200, -0.070}, {-0.600, 0.200, 1.000}, {-0.200, 0.200, 1.000}, {0.200, 0.200, -0.070}, {0.600, 0.200, -0.791}, {1.000, 0.200, -1.000},
    {-1.000, -0.200, -0.070}, {-0.600, -0.200, -0.070}, {-0.200, -0.200, -1.000}, {0.200, -0.200, -0.070}, {0.600, -0.200, -0.070}, {1.000, -0.200, -0.070},
    {-1.000, -0.600, -0.070}, {-0.600, -0.600, -0.070}, {-0.200, -0.600, -0.070}, {0.200, -0.600, 0.163}, {0.600, -0.600, -1.000}, {1.000, -0.600, -0.070},
    {-1.000, -1.000, 0.628}, {-0.600, -1.000, -0.070}, {-0.200, -1.000, 1.000}, {0.200, -1.000, -0.070}, {0.600, -1.000, -0.442}, {1.000, -1.000, -0.628}
  };
  std::vector<float> weightsFlatten = {
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1
  };
  std::vector<float> uKnots = {0.0f, 0.333f, 0.666f, 1.0f};
  std::vector<uint32_t> uMults = {4, 1, 1, 4};
  std::vector<float> vKnots = {0.0f, 0.333f, 0.666f, 1.0f};
  std::vector<uint32_t> vMults = {4, 1, 1, 4};

  nurbs_rt::Matrix<LiteMath::float3> points(6, 6, pointsFlatten.data());
  nurbs_rt::Matrix<float> weights(6, 6, weightsFlatten.data());

  nurbs_rt::NurbsSurface surface(points, weights, uKnots, uMults, vKnots, vMults);

  LiteImage::Image2D<uint32_t> image(1024, 1024);
  LiteMath::float3 cameraPos = nurbs_rt::float3(1, 1, -3);
  LiteMath::float4x4 lookAt = LiteMath::lookAt(cameraPos, nurbs_rt::float3(0, 0, 0), nurbs_rt::float3(0, 1, 0));
  LiteMath::float4x4 projection = LiteMath::perspectiveMatrix(45.0f, 1.0f, 0.001f, 100.0f);
  LiteMath::float4x4 viewProj = projection * lookAt;

  nurbs_rt::drawUniformSamples(surface, image, 1000, 1000, viewProj);  
  std::filesystem::path savePathUniformSamples = std::filesystem::current_path() / "uniform_samples.bmp";
  std::cout << "Saving image to " << savePathUniformSamples.string() << std::endl;
  LiteImage::SaveImage(savePathUniformSamples.string().c_str(), image);

  image.clear(0);
  for (uint32_t i = 0; i < 10; ++i) {
    nurbs_rt::drawNewtonStochastic(surface, image, lookAt, projection, i);
  }
  std::filesystem::path savePath = std::filesystem::current_path() / "newton.bmp";
  std::cout << "Saving image to " << savePath.string() << std::endl;
  LiteImage::SaveImage(savePath.string().c_str(), image);
  
  return 0;
}