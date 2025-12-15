#include <nurbs_rt/nurbs_rt.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <random>

namespace nurbs_rt {

bool checkKnotsCount(uint32_t knotsCount, uint32_t pointsCount) {
  return (knotsCount >= (pointsCount + 1) + 1) // min degree >= 1
         && (knotsCount <= (pointsCount + 1) + MAX_NURBS_DEGREE);
}

bool checkKnotsMults(const std::vector<uint32_t> &mults, uint32_t degree) {
  bool valid = true;

  valid &= (mults.front() == degree + 1);
  valid &= (mults.back() == degree + 1);
  for (uint32_t i = 1; i + 1 < mults.size(); ++i) {
    valid &= (mults[i] >= 1) && (mults[i] <= degree);
  }

  return valid;
}

bool checkKnotsIsDifferent(const std::vector<float> &knots) {
  return std::adjacent_find(knots.begin(), knots.end()) == knots.end();
}

bool checkKnotsOrder(const std::vector<float> &knots) {
  return std::is_sorted(knots.begin(), knots.end());
}

bool checkWeightsArePositive(const Matrix<float> &weights) {
  return (*std::min_element(weights.vector().begin(), weights.vector().end())) >
         0;
}

auto flattenKnots(const std::vector<float> &knots,
                  const std::vector<uint32_t> &mults) {
  uint32_t knotsTotal = std::accumulate(mults.begin(), mults.end(), 0);
  std::vector<float> flatKnots(knotsTotal);
  for (uint32_t i = 0, currentPos = 0; i < mults.size(); ++i) {
    std::fill_n(flatKnots.begin() + currentPos, mults[i], knots[i]);
    currentPos += mults[i];
  }

  return flatKnots;
}

uint32_t nurbs_rt::NurbsSurface::uDegree() const noexcept {
  return uint32_t(m_uKnots.size() - 1 - m_controlPointsWeighted.height());
}

uint32_t nurbs_rt::NurbsSurface::vDegree() const noexcept {
  return uint32_t(m_vKnots.size() - 1 - m_controlPointsWeighted.width());
}

BBox3f nurbs_rt::NurbsSurface::boundingBox() const noexcept {
  return m_boundingBox;
}

nurbs_rt::NurbsSurface::NurbsSurface(const Matrix<float3> &controlPoints,
                                     const Matrix<float> &weights,
                                     const std::vector<float> &uKnots,
                                     const std::vector<uint32_t> &uMults,
                                     const std::vector<float> &vKnots,
                                     const std::vector<uint32_t> &vMults) {
  auto flattenUknots = flattenKnots(uKnots, uMults);
  auto flattenVknots = flattenKnots(vKnots, vMults);

  assert(
      checkKnotsCount(flattenUknots.size(), controlPoints.height()) &&
      "uKnots count is not correct. Supported degrees: [1, MAX_NURBS_DEGREE]");
  assert(
      checkKnotsCount(flattenVknots.size(), controlPoints.width()) &&
      "vKnots count is not correct. Supported degrees: [1, MAX_NURBS_DEGREE]");
  assert(checkKnotsMults(uMults,
                         flattenUknots.size() - 1 - controlPoints.height()) &&
         "uMults is not correct");
  assert(checkKnotsMults(vMults,
                         flattenVknots.size() - 1 - controlPoints.width()) &&
         "vMults is not correct");
  assert(checkKnotsIsDifferent(uKnots) &&
         "uKnots must be different. Equal knots are encoded via mults "
         "(multiplicity) array");
  assert(checkKnotsIsDifferent(vKnots) &&
         "uKnots must be different. Equal knots are encoded via mults "
         "(multiplicity) array");
  assert(checkKnotsOrder(uKnots) && "uKnots are not ordered");
  assert(checkKnotsOrder(vKnots) && "vKnots are not ordered");
  assert(checkWeightsArePositive(weights) &&
         "weights must be positive. Negative weights are not supported");

  m_boundingBox.boxMin = float3(std::numeric_limits<float>::max());
  m_boundingBox.boxMax = float3(std::numeric_limits<float>::min());
  for (auto &point : controlPoints.vector()) {
    m_boundingBox.boxMin = LiteMath::min(m_boundingBox.boxMin, point);
    m_boundingBox.boxMax = LiteMath::max(m_boundingBox.boxMax, point);
  }

  m_controlPointsWeighted.resize(controlPoints.width(), controlPoints.height());
  for (uint32_t ui = 0; ui < controlPoints.height(); ui++) {
    for (uint32_t vi = 0; vi < controlPoints.width(); vi++) {
      auto idx = index2{vi, ui};
      m_controlPointsWeighted[idx] =
          LiteMath::to_float4(controlPoints[idx], 1.0f) * weights[idx];
    }
  }
  m_uKnots = flattenUknots;
  m_vKnots = flattenVknots;
}

uint32_t knotSpan(float t, const float *knots, uint32_t knotsCount,
                  uint32_t degree) {
  assert(knots[0] <= t && t <= knots[knotsCount - 1]);

  uint32_t n = knotsCount - degree - 2;

  if (t == knots[n + 1]) {
    return n;
  }

  uint32_t spanId = std::upper_bound(knots, knots + knotsCount, t) - knots - 1;
  spanId = std::clamp(spanId, 0u, n);

  return spanId;
}

void calculateBSplineBasis(float t, uint32_t spanId, const float *knots,
                           uint32_t degree, float *basis) {
  // The NURBS Book, 2-nd edition, ALGORITHM A2.2, p. 70

  uint32_t p = degree;
  std::array<float, MAX_NURBS_DEGREE + 1> left = {}, right = {};

  basis[0] = 1.0f;
  for (uint32_t j = 1; j <= p; j++) {
    left[j] = t - knots[spanId + 1 - j];
    right[j] = knots[spanId + j] - t;
    float saved = 0.0f;
    for (uint32_t r = 0; r < j; r++) {
      float temp = basis[r] / (right[r + 1] + left[j - r]);
      basis[r] = saved + temp * right[r + 1];
      saved = temp * left[j - r];
    }
    basis[j] = saved;
  }
}

template <typename T, typename F>
T evalNurbsCurve(float t, F pointsGenerator, const float *knots,
                 uint32_t knotsCount, uint32_t degree) {
  assert(degree >= 0 && degree <= MAX_NURBS_DEGREE);

  std::array<float, MAX_NURBS_DEGREE + 1> basis = {};
  uint32_t spanId = knotSpan(t, knots, knotsCount, degree);
  calculateBSplineBasis(t, spanId, knots, degree, basis.data());

  T result = T{};
  for (uint32_t i = 0; i <= degree; i++) {
    result += basis[i] * pointsGenerator(spanId - degree + i);
  }
  return result;
}

template <typename T, typename F>
T evalNurbsCurveDerivative(float t, F pointsGenerator, const float *knots,
                           uint32_t knotsCount, uint32_t degree) {
  assert(degree >= 0 && degree <= MAX_NURBS_DEGREE);

  auto derivativePointsGenerator = [&](uint32_t i) -> T {
    float p = static_cast<float>(degree);
    float denom = knots[i + degree + 1] - knots[i + 1];
    float coeff = (denom != 0.0f) ? p / denom : 0.0f;
    return coeff * (pointsGenerator(i + 1) - pointsGenerator(i));
  };

  return evalNurbsCurve<T>(t, derivativePointsGenerator, knots + 1,
                           knotsCount - 2, degree - 1);
}

float4 nurbs_rt::NurbsSurface::eval4(float u, float v) const {
  auto evalCurveV = [this, v](uint32_t uId) {
    return evalNurbsCurve<float4>(
        v,
        [&](uint32_t vId) { return m_controlPointsWeighted[index2{vId, uId}]; },
        m_vKnots.data(), m_vKnots.size(), vDegree());
  };

  float4 res = evalNurbsCurve<float4>(u, evalCurveV, m_uKnots.data(),
                                      m_uKnots.size(), uDegree());

  return res;
}

float3 nurbs_rt::NurbsSurface::eval(float u, float v) const {
  float4 res4 = eval4(u, v);
  return to_float3(res4 / res4.w);
}

float3 nurbs_rt::NurbsSurface::uDerivative(float u, float v,
                                           float4 point4) const {
  auto evalCurveV = [this, v](uint32_t uId) {
    return evalNurbsCurve<float4>(
        v,
        [&](uint32_t vId) { return m_controlPointsWeighted[index2{vId, uId}]; },
        m_vKnots.data(), m_vKnots.size(), vDegree());
  };

  float4 res4 = evalNurbsCurveDerivative<float4>(u, evalCurveV, m_uKnots.data(),
                                                 m_uKnots.size(), uDegree());

  return to_float3((res4 * point4.w - point4 * res4.w) / (point4.w * point4.w));
}

float3 nurbs_rt::NurbsSurface::uDerivative(float u, float v) const {
  float4 point4 = eval4(u, v);
  return uDerivative(u, v, point4);
}

float3 nurbs_rt::NurbsSurface::vDerivative(float u, float v,
                                           float4 point4) const {
  auto evalCurveDerV = [this, v](uint32_t uId) {
    return evalNurbsCurveDerivative<float4>(
        v,
        [&](uint32_t vId) { return m_controlPointsWeighted[index2{vId, uId}]; },
        m_vKnots.data(), m_vKnots.size(), vDegree());
  };

  float4 res4 = evalNurbsCurve<float4>(u, evalCurveDerV, m_uKnots.data(),
                                       m_uKnots.size(), uDegree());

  return to_float3((res4 * point4.w - point4 * res4.w) / (point4.w * point4.w));
}

float3 nurbs_rt::NurbsSurface::vDerivative(float u, float v) const {
  float4 point4 = eval4(u, v);
  return vDerivative(u, v, point4);
}

inline float2 project2planes(float4 P1, float4 P2, float4 point) {
  return float2{dot(P1, point), dot(P2, point)};
}

HitInfo
nurbs_rt::NurbsSurface::intersect(const float3 &origin, const float3 &dir,
                                  const NewtonParameters &params) const {
  HitInfo hit = {};
  float2 uv = params.initialGuess;

  auto boxIntersection =
      m_boundingBox.Intersection(origin, 1.0f / dir, 0.000f, 1e16f);
  if (boxIntersection.t1 >= boxIntersection.t2) {
    return hit; // no hit
  }

  float3 absDir = LiteMath::abs(dir);
  float3 ortho_dir1 = (absDir.x > absDir.y && absDir.x > absDir.z)
                          ? float3{-dir.y, dir.x, 0}
                          : float3{0, -dir.z, dir.y};
  float3 ortho_dir2 = normalize(cross(ortho_dir1, dir));
  ortho_dir1 = normalize(cross(dir, ortho_dir2));

  float4 P1 = to_float4(ortho_dir1, -dot(ortho_dir1, origin));
  float4 P2 = to_float4(ortho_dir2, -dot(ortho_dir2, origin));

  float4 point4 = eval4(uv.x, uv.y);
  float4 point = point4 / point4.w;
  float2 D = project2planes(P1, P2, point);

  uint32_t steps_left = params.maxIterations - 1;
  while (length(D) > params.eps && steps_left > 0) {
    --steps_left;
    float2 J[2] = {
        project2planes(
            P1, P2, LiteMath::to_float4(uDerivative(uv.x, uv.y, point4), 0.0f)),
        project2planes(
            P1, P2,
            LiteMath::to_float4(vDerivative(uv.x, uv.y, point4), 0.0f))};

    float det = J[0][0] * J[1][1] - J[0][1] * J[1][0];

    float2 J_inversed[2] = {{J[1][1] / det, -J[0][1] / det},
                            {-J[1][0] / det, J[0][0] / det}};

    uv = uv - (J_inversed[0] * D[0] +
               J_inversed[1] * D[1]); // mul2x2x2(J_inversed, D);
    uv.x = LiteMath::clamp(uv.x, m_uKnots.front(), m_uKnots.back());
    uv.y = LiteMath::clamp(uv.y, m_vKnots.front(), m_vKnots.back());

    point4 = eval4(uv.x, uv.y);
    point = point4 / point4.w;
    float2 new_D = project2planes(P1, P2, point);

    if (length(new_D) > length(D))
      return hit; // no hit

    D = new_D;
  }

  if (length(D) > params.eps)
    return hit; // no hit

  float3 uder = uDerivative(uv.x, uv.y, point4);
  float3 vder = vDerivative(uv.x, uv.y, point4);

  float3 normal = normalize(cross(uder, vder));
  if (dot(normal, origin - LiteMath::to_float3(point)) < 0) {
    normal *= -1.0f;
  }

  hit.t = dot(dir, LiteMath::to_float3(point) - origin);
  if (hit.t < 0) {
    hit.t = std::numeric_limits<float>::max();
  }
  hit.normal = normal;
  hit.uv = uv;
  return hit;
}

LiteMath::float2 nurbs_rt::NurbsSurface::uParamsRange() const {
  return {m_uKnots.front(), m_uKnots.back()};
}

LiteMath::float2 nurbs_rt::NurbsSurface::vParamsRange() const {
  return {m_vKnots.front(), m_vKnots.back()};
}

void drawUniformSamples(const NurbsSurface &surface,
                        LiteImage::Image2D<uint32_t> &image,
                        uint32_t uSamplesCount, uint32_t vSamplesCount,
                        const float4x4 &worldViewProj) {
  auto uParams = surface.uParamsRange();
  auto vParams = surface.vParamsRange();

  for (uint32_t ui = 0; ui < uSamplesCount; ++ui) {
    for (uint32_t vi = 0; vi < vSamplesCount; ++vi) {
      float relU = (float)(ui + 0.5f) / uSamplesCount;
      float relV = (float)(vi + 0.5f) / vSamplesCount;
      float u = LiteMath::lerp(uParams[0], uParams[1], relU);
      float v = LiteMath::lerp(vParams[0], vParams[1], relV);

      float4 point = LiteMath::to_float4(surface.eval(u, v), 1.0f);
      point = worldViewProj * point;
      point /= point.w;

      int x = (int)((point.x + 1) / 2 * image.width());
      int y = (int)((point.y + 1) / 2 * image.height());

      if (!(0 <= x && x < (int)image.width()) ||
          !(0 <= y && y < (int)image.height())) {
        continue;
      }

      float4 outputColor = {relU, relV, 0.0f, 1.0f};
      image[index2{uint32_t(x), uint32_t(image.height() - y - 1)}] =
          LiteMath::color_pack_rgba(outputColor);
    }
  }
}

void drawNewtonStochastic(const NurbsSurface &surface,
                          LiteImage::Image2D<uint32_t> &image,
                          const float4x4 &view, const float4x4 &proj,
                          uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float4x4 projInv = inverse4x4(proj);
  float4x4 viewInv = inverse4x4(view);
  for (uint32_t y = 0; y < image.height(); ++y) {
    for (uint32_t x = 0; x < image.width(); ++x) {
      float4 rayDir4 = EyeRayDir4f(x + 0.5f, y + 0.5f, image.width(),
                                   image.height(), projInv);
      rayDir4.w = 0.0f;
      rayDir4 = viewInv * rayDir4;

      float3 rayDir = normalize(to_float3(rayDir4));
      float3 cameraPos = to_float3(viewInv.col(3));

      auto uRange = surface.uParamsRange();
      auto vRange = surface.vParamsRange();
      float u = LiteMath::lerp(uRange.x, uRange.y, dist(rng));
      float v = LiteMath::lerp(vRange.x, vRange.y, dist(rng));

      NewtonParameters params = {};
      params.initialGuess = {u, v};

      HitInfo hit = surface.intersect(cameraPos, rayDir, params);
      if (!hit.hitten()) {
        continue;
      }

      float relU = (hit.uv.x - uRange.x) / (uRange.y - uRange.x);
      float relV = (hit.uv.y - vRange.x) / (vRange.y - vRange.x);

      float4 outputColor = {relU, relV, 0.0f, 1.0f};
      image[index2{x, image.height() - y - 1}] =
          LiteMath::color_pack_rgba(outputColor);
    }
  }
}

void nurbs_rt::NurbsSurface::reparameterizeU(float2 newRange) {
  auto prevRange = uParamsRange();
  for (auto &knot : m_uKnots) {
    float relU = (knot - prevRange.x) / (prevRange.y - prevRange.x);
    knot = LiteMath::lerp(newRange.x, newRange.y, relU);
  }
}

void nurbs_rt::NurbsSurface::reparameterizeV(float2 newRange) {
  auto prevRange = vParamsRange();
  for (auto &knot : m_vKnots) {
    float relV = (knot - prevRange.x) / (prevRange.y - prevRange.x);
    knot = LiteMath::lerp(newRange.x, newRange.y, relV);
  }
}

void nurbs_rt::NurbsSurface::transform(const float4x4 &transformMatrix) {
  uint32_t w = m_controlPointsWeighted.width();
  uint32_t h = m_controlPointsWeighted.height();
  m_boundingBox.boxMin = float3{std::numeric_limits<float>::min()};
  m_boundingBox.boxMax = float3{std::numeric_limits<float>::max()};
  for (uint32_t flatId = 0; flatId < w * h; ++flatId) {
    auto &point = m_controlPointsWeighted.data()[flatId];
    float weight = point.w;

    point /= point.w;
    point = transformMatrix * point;
    point /= point.w;

    m_boundingBox.boxMin =
        LiteMath::min(m_boundingBox.boxMin, LiteMath::to_float3(point));
    m_boundingBox.boxMax =
        LiteMath::max(m_boundingBox.boxMax, LiteMath::to_float3(point));

    point *= weight;
  }
}

} // namespace nurbs_rt