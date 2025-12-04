#pragma once

#include <Image2d.h>
#include <LiteMath.h>

namespace nurbs_rt {
template <typename T> using Matrix = LiteImage::Image2D<T>;
using LiteMath::BBox3f;
using LiteMath::float2, LiteMath::float3, LiteMath::float4;
using LiteMath::float4x4;
using index2 = LiteMath::uint2;

struct HitInfo {
  float t = std::numeric_limits<float>::max();
  float3 normal;
  float2 uv;
  bool hitten() const noexcept { return t < std::numeric_limits<float>::max(); }
};

struct NewtonParameters {
  float2 initialGuess = {0.5f, 0.5f};
  float eps = 1e-6f;
  uint32_t maxIterations = 5;
};

class NurbsSurface {
public:
  NurbsSurface(const Matrix<float3> &controlPoints,
               const Matrix<float> &weights, const std::vector<float> &uKnots,
               const std::vector<uint32_t> &uMults,
               const std::vector<float> &vKnots,
               const std::vector<uint32_t> &vMults);
  float4 eval4(float u, float v) const;
  float3 eval(float u, float v) const;
  float3 uDerivative(float u, float v) const;
  float3 vDerivative(float u, float v) const;
  float3 uDerivative(float u, float v, float4 point4) const;
  float3 vDerivative(float u, float v, float4 point4) const;
  uint32_t uDegree() const noexcept;
  uint32_t vDegree() const noexcept;
  BBox3f boundingBox() const noexcept;
  HitInfo intersect(const float3 &origin, const float3 &direction,
                    const NewtonParameters &params = NewtonParameters{}) const;
  float2 uParamsRange() const;
  float2 vParamsRange() const;
  void reparametrizeU(float2 newRange);
  void reparametrizeV(float2 newRange);
  void transform(const float4x4 &transformMatrix);

private:
  Matrix<float4> m_controlPointsWeighted;
  std::vector<float> m_uKnots;
  std::vector<float> m_vKnots;
  BBox3f m_boundingBox;
};

constexpr uint32_t MAX_NURBS_DEGREE = 10;

void drawNewtonStochastic(const NurbsSurface &surface,
                          LiteImage::Image2D<uint32_t> &image,
                          const float4x4 &view, const float4x4 &projection,
                          uint32_t seed = 0);

void drawUniformSamples(const NurbsSurface &surface,
                        LiteImage::Image2D<uint32_t> &image,
                        uint32_t uSamplesCount, uint32_t vSamplesCount,
                        const float4x4 &worldViewProj);

} // namespace nurbs_rt
