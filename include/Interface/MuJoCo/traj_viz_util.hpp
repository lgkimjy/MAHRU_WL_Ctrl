#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <mujoco/mujoco.h>

namespace mujoco
{
class TrajVizUtil
{
public:
    using Color = std::array<float, 4>;

    void clear()
    {
        std::lock_guard<std::mutex> lock(marker_mutex_);
        markers_.clear();
    }

    void remove(const std::string& name)
    {
        std::lock_guard<std::mutex> lock(marker_mutex_);
        markers_.erase(name);
    }

    void clearPrefix(const std::string& prefix)
    {
        std::lock_guard<std::mutex> lock(marker_mutex_);
        for (auto it = markers_.begin(); it != markers_.end();) {
            if (it->first.rfind(prefix, 0) == 0) {
                it = markers_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void sphere(const std::string& name,
                const Eigen::Vector3d& pos,
                double radius,
                const Color& color)
    {
        Marker marker;
        marker.kind = MarkerKind::Sphere;
        marker.points = {pos};
        marker.size = radius;
        marker.color = color;
        setMarker(name, marker);
    }

    void cylinder(const std::string& name,
                  const Eigen::Vector3d& pos,
                  double radius,
                  double height,
                  const Color& color)
    {
        Marker marker;
        marker.kind = MarkerKind::Cylinder;
        marker.points = {pos};
        marker.size = radius;
        marker.axis_length = height;
        marker.color = color;
        setMarker(name, marker);
    }

    void label(const std::string& name,
               const Eigen::Vector3d& pos,
               const std::string& text,
               const Color& color = {0.0f, 0.0f, 1.0f, 1.0f})
    {
        Marker marker;
        marker.kind = MarkerKind::Label;
        marker.points = {pos};
        marker.color = color;
        marker.label = text;
        setMarker(name, marker);
    }

    void frame(const std::string& name,
               const Eigen::Vector3d& pos,
               const Eigen::Matrix3d& rot,
               double axis_length,
               double axis_radius = 0.005)
    {
        Marker marker;
        marker.kind = MarkerKind::Frame;
        marker.points = {pos};
        marker.rot = rot;
        marker.axis_length = axis_length;
        marker.axis_radius = axis_radius;
        setMarker(name, marker);
    }

    void line(const std::string& name,
              const Eigen::Vector3d& from,
              const Eigen::Vector3d& to,
              double width,
              const Color& color)
    {
        Marker marker;
        marker.kind = MarkerKind::Line;
        marker.points = {from, to};
        marker.size = width;
        marker.color = color;
        setMarker(name, marker);
    }

    void traj(const std::string& name,
              const std::vector<Eigen::Vector3d>& points,
              double width,
              const Color& color)
    {
        Marker marker;
        marker.kind = MarkerKind::Trajectory;
        marker.points = points;
        marker.size = width;
        marker.color = color;
        setMarker(name, marker);
    }

    void horizon(const std::string& name,
                 const std::vector<Eigen::Vector3d>& points,
                 double width,
                 const Color& color)
    {
        traj(name, points, width, color);
    }

    void mpcHorizon(const std::string& name,
                    const Eigen::VectorXd& stacked_states,
                    int horizon_len,
                    int state_dim,
                    int position_offset,
                    double width,
                    const Color& color)
    {
        std::vector<Eigen::Vector3d> points;
        points.reserve(std::max(0, horizon_len));

        for (int i = 0; i < horizon_len; ++i) {
            const int idx = i * state_dim + position_offset;
            if (idx + 2 >= stacked_states.size()) break;
            points.emplace_back(stacked_states.segment<3>(idx));
        }

        horizon(name, points, width, color);
    }

    void pushTrail(const std::string& name,
                   const Eigen::Vector3d& pos,
                   size_t max_len,
                   double width,
                   const Color& color)
    {
        std::lock_guard<std::mutex> lock(marker_mutex_);

        Marker& marker = markers_[name];
        marker.kind = MarkerKind::Trajectory;
        marker.size = width;
        marker.color = color;
        marker.points.push_back(pos);

        trimTrail(marker, max_len);
    }

    void pushTrail(const std::string& name,
                   const Eigen::Vector3d& pos,
                   const Eigen::Matrix3d& rot,
                   size_t max_len,
                   double width,
                   const Color& color,
                   size_t frame_stride = 50,
                   double axis_length = 0.12,
                   double axis_radius = 0.003)
    {
        std::lock_guard<std::mutex> lock(marker_mutex_);

        Marker& marker = markers_[name];
        marker.kind = MarkerKind::TrajectoryWithFrames;
        marker.size = width;
        marker.color = color;
        marker.frame_stride = frame_stride;
        marker.axis_length = axis_length;
        marker.axis_radius = axis_radius;
        marker.points.push_back(pos);
        marker.rotations.push_back(rot);

        trimTrail(marker, max_len);
    }

    void update(mjvScene& scn)
    {
        std::unordered_map<std::string, Marker> markers;
        {
            std::lock_guard<std::mutex> lock(marker_mutex_);
            markers = markers_;
        }

        for (const auto& [name, marker] : markers) {
            switch (marker.kind) {
            case MarkerKind::Sphere:
                drawSphere(marker, scn);
                break;
            case MarkerKind::Cylinder:
                drawCylinder(marker, scn);
                break;
            case MarkerKind::Label:
                drawLabel(marker, scn);
                break;
            case MarkerKind::Frame:
                drawFrame(marker, scn);
                break;
            case MarkerKind::Line:
                drawLine(marker, scn);
                break;
            case MarkerKind::Trajectory:
                drawTrajectory(marker, scn);
                break;
            case MarkerKind::TrajectoryWithFrames:
                drawTrajectoryWithFrames(marker, scn);
                break;
            }
        }
    }

private:
    enum class MarkerKind {
        Sphere,
        Cylinder,
        Label,
        Frame,
        Line,
        Trajectory,
        TrajectoryWithFrames
    };

    struct Marker {
        MarkerKind kind = MarkerKind::Sphere;
        std::vector<Eigen::Vector3d> points;
        std::vector<Eigen::Matrix3d> rotations;
        Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
        double size = 0.01;
        double axis_length = 0.1;
        double axis_radius = 0.005;
        size_t frame_stride = 50;
        Color color = {1.0f, 1.0f, 1.0f, 1.0f};
        std::string label;
    };

    std::unordered_map<std::string, Marker> markers_;
    std::mutex marker_mutex_;

    void setMarker(const std::string& name, const Marker& marker)
    {
        std::lock_guard<std::mutex> lock(marker_mutex_);
        markers_[name] = marker;
    }

    static void trimTrail(Marker& marker, size_t max_len)
    {
        while (marker.points.size() > max_len) {
            marker.points.erase(marker.points.begin());
            if (!marker.rotations.empty()) {
                marker.rotations.erase(marker.rotations.begin());
            }
        }
    }

    static void copyColor(float dst[4], const Color& src)
    {
        for (int i = 0; i < 4; ++i) {
            dst[i] = src[i];
        }
    }

    static void copyPos(mjtNum dst[3], const Eigen::Vector3d& src)
    {
        for (int i = 0; i < 3; ++i) {
            dst[i] = src(i);
        }
    }

    static void copyRot(mjtNum dst[9], const Eigen::Matrix3d& src)
    {
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                dst[row * 3 + col] = src(row, col);
            }
        }
    }

    static bool canAddGeom(const mjvScene& scn, int count = 1)
    {
        return scn.ngeom + count <= scn.maxgeom;
    }

    void drawSphere(const Marker& marker, mjvScene& scn)
    {
        if (marker.points.empty() || !canAddGeom(scn)) return;

        mjtNum pos[3];
        mjtNum size[1] = {marker.size};
        float color[4];
        copyPos(pos, marker.points.front());
        copyColor(color, marker.color);

        mjvGeom* geom = scn.geoms + scn.ngeom++;
        mjv_initGeom(geom, mjGEOM_SPHERE, size, pos, nullptr, color);
    }

    void drawCylinder(const Marker& marker, mjvScene& scn)
    {
        if (marker.points.empty() || !canAddGeom(scn)) return;

        const Eigen::Vector3d center = marker.points.front();
        const Eigen::Vector3d half_height(0.0, 0.0, 0.5 * marker.axis_length);

        mjtNum from[3];
        mjtNum to[3];
        copyPos(from, center - half_height);
        copyPos(to, center + half_height);

        mjvGeom* geom = scn.geoms + scn.ngeom++;
        mjv_initGeom(geom, mjGEOM_CYLINDER, nullptr, nullptr, nullptr, nullptr);
        mjv_connector(geom, mjGEOM_CYLINDER, marker.size, from, to);
        copyColor(geom->rgba, marker.color);
    }

    void drawLabel(const Marker& marker, mjvScene& scn)
    {
        if (marker.points.empty() || !canAddGeom(scn)) return;

        mjtNum pos[3];
        float color[4];
        copyPos(pos, marker.points.front());
        copyColor(color, marker.color);

        mjvGeom* geom = scn.geoms + scn.ngeom++;
        mjv_initGeom(geom, mjGEOM_LABEL, nullptr, pos, nullptr, color);
        std::strncpy(geom->label, marker.label.c_str(), sizeof(geom->label));
        geom->label[sizeof(geom->label) - 1] = '\0';
    }

    void drawFrame(const Marker& marker, mjvScene& scn)
    {
        if (marker.points.empty()) return;
        drawFrameAt(marker.points.front(), marker.rot, marker.axis_length, marker.axis_radius, scn);
    }

    void drawFrameAt(const Eigen::Vector3d& frame_pos,
                     const Eigen::Matrix3d& frame_rot,
                     double axis_length,
                     double axis_radius,
                     mjvScene& scn)
    {
        if (!canAddGeom(scn, 3)) return;

        mjtNum pos[3];
        mjtNum rot[9];
        copyPos(pos, frame_pos);
        copyRot(rot, frame_rot);

        for (int axis_idx = 0; axis_idx < 3; ++axis_idx) {
            mjvGeom* geom = scn.geoms + scn.ngeom++;
            mjv_initGeom(geom, mjGEOM_CYLINDER, nullptr, nullptr, nullptr, nullptr);

            mjtNum axis[3] = {0.0, 0.0, 0.0};
            mjtNum vec[3];
            mjtNum to[3];
            axis[axis_idx] = axis_length;
            mju_mulMatVec(vec, rot, axis, 3, 3);
            mju_add3(to, pos, vec);
            mjv_connector(geom, mjGEOM_CYLINDER, axis_radius, pos, to);

            geom->rgba[0] = axis_idx == 0 ? 0.9f : 0.0f;
            geom->rgba[1] = axis_idx == 1 ? 0.9f : 0.0f;
            geom->rgba[2] = axis_idx == 2 ? 0.9f : 0.0f;
            geom->rgba[3] = 1.0f;
        }
    }

    void drawLine(const Marker& marker, mjvScene& scn)
    {
        if (marker.points.size() < 2 || !canAddGeom(scn)) return;

        mjtNum from[3];
        mjtNum to[3];
        float color[4];
        copyPos(from, marker.points[0]);
        copyPos(to, marker.points[1]);
        copyColor(color, marker.color);

        mjvGeom* geom = scn.geoms + scn.ngeom++;
        mjv_initGeom(geom, mjGEOM_SPHERE, nullptr, nullptr, nullptr, color);
        mjv_connector(geom, mjGEOM_LINE, marker.size, from, to);
    }

    void drawTrajectory(const Marker& marker, mjvScene& scn)
    {
        if (marker.points.size() < 2) return;

        float color[4];
        copyColor(color, marker.color);

        for (size_t i = 1; i < marker.points.size(); ++i) {
            if (!canAddGeom(scn)) return;

            mjtNum from[3];
            mjtNum to[3];
            copyPos(from, marker.points[i - 1]);
            copyPos(to, marker.points[i]);

            mjvGeom* geom = scn.geoms + scn.ngeom++;
            mjv_initGeom(geom, mjGEOM_SPHERE, nullptr, nullptr, nullptr, color);
            mjv_connector(geom, mjGEOM_LINE, marker.size, from, to);
        }
    }

    void drawTrajectoryWithFrames(const Marker& marker, mjvScene& scn)
    {
        drawTrajectory(marker, scn);

        if (marker.points.empty() || marker.rotations.empty()) return;

        const size_t stride = std::max<size_t>(1, marker.frame_stride);
        const size_t count = std::min(marker.points.size(), marker.rotations.size());

        for (size_t i = 0; i < count; i += stride) {
            drawFrameAt(marker.points[i], marker.rotations[i], marker.axis_length, marker.axis_radius, scn);
        }

        if ((count - 1) % stride != 0) {
            drawFrameAt(marker.points[count - 1], marker.rotations[count - 1], marker.axis_length, marker.axis_radius, scn);
        }
    }
};
}
