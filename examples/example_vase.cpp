#include <iostream>
#include <vector>
#include <filesystem>
#include <cmath>

#include <nurbs_rt/nurbs_rt.h>

int main() {
  std::vector<nurbs_rt::float3> pointsFlatten = {
    {1.0000, 1.0000, 0.0000}, {1.0000, 1.0000, 1.0000}, {0.0000, 1.0000, 1.0000}, {-1.0000, 1.0000, 1.0000}, {-1.0000, 1.0000, 0.0000}, {-1.0000, 1.0000, -1.0000}, {0.0000, 1.0000, -1.0000}, {1.0000, 1.0000, -1.0000}, {1.0000, 1.0000, 0.0000},
    {0.1667, -0.3846, 0.0000}, {0.1667, -0.3846, 0.1667}, {0.0000, -0.3846, 0.1667}, {-0.1667, -0.3846, 0.1667}, {-0.1667, -0.3846, 0.0000}, {-0.1667, -0.3846, -0.1667}, {0.0000, -0.3846, -0.1667}, {0.1667, -0.3846, -0.1667}, {0.1667, -0.3846, 0.0000},
    {0.3333, -1.0000, 0.0000}, {0.3333, -1.0000, 0.3333}, {0.0000, -1.0000, 0.3333}, {-0.3333, -1.0000, 0.3333}, {-0.3333, -1.0000, 0.0000}, {-0.3333, -1.0000, -0.3333}, {0.0000, -1.0000, -0.3333}, {0.3333, -1.0000, -0.3333}, {0.3333, -1.0000, 0.0000}
  };
  std::vector<float> weightsFlatten = {
    1, 0.707107, 1, 0.707107, 1, 0.707107, 1, 0.707107, 1, 
    1, 0.707107, 1, 0.707107, 1, 0.707107, 1, 0.707107, 1, 
    1, 0.707107, 1, 0.707107, 1, 0.707107, 1, 0.707107, 1
  };
  std::vector<float> uKnots = {3, 3.5, 4};
  std::vector<uint32_t> uMults = {2, 1, 2};
  std::vector<float> vKnots = {0, 0.25, 0.5, 0.75, 1};
  std::vector<uint32_t> vMults = {3, 2, 2, 2, 3};

  nurbs_rt::Matrix<nurbs_rt::float3> points(9, 3, pointsFlatten.data());
  nurbs_rt::Matrix<float> weights(9, 3, weightsFlatten.data());

  nurbs_rt::NurbsSurface surface(points, weights, uKnots, uMults, vKnots, vMults);

  LiteImage::Image2D<uint32_t> image(1024, 1024);
  LiteMath::float3 cameraPos = nurbs_rt::float3(0, 2, 3);
  LiteMath::float4x4 lookAt = LiteMath::lookAt(cameraPos, nurbs_rt::float3(0, 0, 0), nurbs_rt::float3(0, 1, 0));
  LiteMath::float4x4 projection = LiteMath::perspectiveMatrix(45.0f, 1.0f, 0.001f, 100.0f);
  LiteMath::float4x4 viewProj = projection * lookAt;

  LiteMath::float4x4 transform = LiteMath::rotate4x4Z(3.14f/8);
  surface.transform(transform);

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