#include <iostream>
#include <vector>
#include <filesystem>
#include <cmath>
#include <random>

#include <nurbs_rt/nurbs_rt.h>

std::ostream &operator<<(std::ostream &os, const LiteMath::float3 &v) {
  os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const LiteMath::float4 &v) {
  os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const LiteMath::float2 &v) {
  os << "(" << v.x << ", " << v.y << ")";
  return os;
}

int main() {
  std::vector<LiteMath::float3> pointsFlatten = {
    {1.0000, 1.0000, 0.0000}, {1.0000, 1.0000, 1.0000}, {0.0000, 1.0000, 1.0000}, {-1.0000, 1.0000, 1.0000}, {-1.0000, 1.0000, 0.0000}, {-1.0000, 1.0000, -1.0000}, {0.0000, 1.0000, -1.0000}, {1.0000, 1.0000, -1.0000}, {1.0000, 1.0000, 0.0000},
    {0.1667, -0.3846, 0.0000}, {0.1667, -0.3846, 0.1667}, {0.0000, -0.3846, 0.1667}, {-0.1667, -0.3846, 0.1667}, {-0.1667, -0.3846, 0.0000}, {-0.1667, -0.3846, -0.1667}, {0.0000, -0.3846, -0.1667}, {0.1667, -0.3846, -0.1667}, {0.1667, -0.3846, 0.0000},
    {0.3333, -1.0000, 0.0000}, {0.3333, -1.0000, 0.3333}, {0.0000, -1.0000, 0.3333}, {-0.3333, -1.0000, 0.3333}, {-0.3333, -1.0000, 0.0000}, {-0.3333, -1.0000, -0.3333}, {0.0000, -1.0000, -0.3333}, {0.3333, -1.0000, -0.3333}, {0.3333, -1.0000, 0.0000}
  };
  std::vector<float> weightsFlatten = {
    1, 0.707107, 1, 0.707107, 1, 0.707107, 1, 0.707107, 1, 
    1, 0.707107, 1, 0.707107, 1, 0.707107, 1, 0.707107, 1, 
    1, 0.707107, 1, 0.707107, 1, 0.707107, 1, 0.707107, 1
  };
  std::vector<float> uKnots = {0, 0.5, 1};
  std::vector<uint32_t> uMults = {2, 1, 2};
  std::vector<float> vKnots = {0, 0.25, 0.5, 0.75, 1};
  std::vector<uint32_t> vMults = {3, 2, 2, 2, 3};

  nurbs_rt::Matrix<LiteMath::float3> points(9, 3, pointsFlatten.data());
  nurbs_rt::Matrix<float> weights(9, 3, weightsFlatten.data());

  nurbs_rt::NurbsSurface surface(points, weights, uKnots, uMults, vKnots, vMults);

  LiteImage::Image2D<uint32_t> image(1024, 1024);
  LiteMath::float3 cameraPos = nurbs_rt::float3(0, 2, 3);
  LiteMath::float4x4 lookAt = LiteMath::lookAt(cameraPos, nurbs_rt::float3(0, 0, 0), nurbs_rt::float3(0, 1, 0));
  LiteMath::float4x4 projection = LiteMath::perspectiveMatrix(45.0f, 1.0f, 0.001f, 100.0f);
  LiteMath::float4x4 viewProj = projection * lookAt;

  LiteMath::float4x4 transform = LiteMath::rotate4x4Z(3.14f/8);
  surface.transform(transform);
  surface.reparameterizeU({3.0f, 8.0f});

  {
    float u = 3.234212455f;
    float v = 0.6;

    std::cout << "u: " << u << " v: " << v << std::endl;
    
    auto point = surface.eval(u, v);
    auto point4 = surface.eval4(u, v);
    std::cout << "point: " << point << std::endl;
    std::cout << "point4: " << point4 << std::endl;

    auto uDerivative = surface.uDerivative(u, v);
    auto uDerivativeFaster = surface.uDerivative(u, v, point4);
    auto vDerivative = surface.vDerivative(u, v);
    auto vDerivativeFaster = surface.vDerivative(u, v, point4);
    std::cout << "uderivative: " << uDerivative << std::endl;
    std::cout << "uderivative faster: " << uDerivativeFaster << std::endl;
    std::cout << "vderivative: " << vDerivative << std::endl;
    std::cout << "vderivative faster: " << vDerivativeFaster << std::endl;
    std::cout << std::endl;

    nurbs_rt::NewtonParameters params = {};
    params.maxIterations = 5;
    params.eps = 1e-6f;
    params.initialGuess = {0.5f, 0.5f};

    LiteMath::float3 rayPos = cameraPos;
    LiteMath::float3 rayDir = LiteMath::normalize(LiteMath::to_float3(point4/point4.w) - cameraPos);
    std::cout << "rayPos: " << rayPos << std::endl;
    std::cout << "rayDir: " << rayDir << std::endl;
    // let's try to find initial guess 
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    nurbs_rt::HitInfo hit = {};
    while(!hit.hitten()) {
      auto uBounds = surface.uParamsRange();
      auto vBounds = surface.vParamsRange();
      
      float u = LiteMath::lerp(uBounds.x, uBounds.y, dist(rng));
      float v = LiteMath::lerp(vBounds.x, vBounds.y, dist(rng));

      params.initialGuess = {u, v};
      hit = surface.intersect(rayPos, rayDir, params);
      std::cout << "initGuess: (" << u << ", " << v << "), hit: " << std::boolalpha << hit.hitten() << std::endl;
      if (hit.hitten()) {
        std::cout << "\t t = " << hit.t << std::endl;
        std::cout << "\t normal = " << hit.normal << std::endl;
        std::cout << "\t uv = " << hit.uv << std::endl;
      }
    }
    std::cout << std::endl;
  }

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