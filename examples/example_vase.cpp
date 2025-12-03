#include <iostream>
#include <vector>
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
  std::vector<float> uKnots = {0, 0.5, 1};
  std::vector<uint32_t> uMults = {2, 1, 2};
  std::vector<float> vKnots = {0, 0.25, 0.5, 0.75, 1};
  std::vector<uint32_t> vMults = {3, 2, 2, 2, 3};

  nurbs_rt::Matrix<nurbs_rt::float3> points(9, 3, pointsFlatten.data());
  nurbs_rt::Matrix<float> weights(9, 3, weightsFlatten.data());

  nurbs_rt::NurbsSurface surface(points, weights, uKnots, uMults, vKnots, vMults);

  nurbs_rt::float3 evalPoint = surface.eval(0.2, 0.75);
  std::cout << evalPoint.x << " " << evalPoint.y << " " << evalPoint.z << std::endl;
  
  return 0;
}