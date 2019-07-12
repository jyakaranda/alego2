/**
 * @file LM.cpp
 * @author heng zhang (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2019-07-11
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include "alego2/utility.h"
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

using namespace std;
using namespace gtsam;

class LM : public rclcpp::Node
{
private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_surf_last_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_corner_last_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_outlier_last_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_laser_odom_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_surround_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_aft_mapped_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_keyposes_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_history_keyframes_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_icp_keyframes_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_recent_keyframes_;

  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr srv_save_map_;

  tf2_ros::TransformBroadcaster *tf_broad_;

  NonlinearFactorGraph gtSAMgraph_;
  Values init_estimate_;
  Values opt_estimate_;
  ISAM2 *isam_;
  Values isam_cur_estimate_;
  noiseModel::Diagonal::shared_ptr prior_noise_;
  noiseModel::Diagonal::shared_ptr odom_noise_;
  noiseModel::Diagonal::shared_ptr constraint_noise_;

  // 回环检测相关
  bool loop_closed_;
  int latest_history_frame_id_;
  int closest_history_frame_id_;
  PointCloudT::Ptr latest_keyframe_;
  PointCloudT::Ptr near_history_keyframes_;

  std::mutex mtx_;
  std::thread loop_thread_;
  std::thread main_thread_;
  std::thread visualize_thread_;

  float history_search_radius_; // 回环检测参数
  int history_search_num_;
  float history_fitness_score_;
  bool loop_closure_enabled_;

  PointCloudT::Ptr surf_last_;
  PointCloudT::Ptr corner_last_;
  PointCloudT::Ptr outlier_last_;

  vector<PointCloudT::Ptr> corner_frames_;
  vector<PointCloudT::Ptr> surf_frames_;
  vector<PointCloudT::Ptr> outlier_frames_;

  PointCloudT::Ptr cloud_keyposes_3d_;
  pcl::PointCloud<PointTypePose>::Ptr cloud_keyposes_6d_;

  PointCloudT::Ptr corner_from_map_;
  PointCloudT::Ptr surf_from_map_;
  // PointCloudT::Ptr outlier_from_map_;
  PointCloudT::Ptr corner_from_map_ds_;
  PointCloudT::Ptr surf_from_map_ds_;

  PointCloudT::Ptr laser_corner_;
  PointCloudT::Ptr laser_surf_;
  PointCloudT::Ptr laser_outlier_;
  PointCloudT::Ptr laser_surf_total_;
  PointCloudT::Ptr laser_corner_ds_;
  PointCloudT::Ptr laser_surf_ds_;
  PointCloudT::Ptr laser_outlier_ds_;
  PointCloudT::Ptr laser_surf_total_ds_;
  double time_laser_corner_;
  double time_laser_surf_;
  double time_laser_outlier_;
  double time_laser_odom_;
  bool new_laser_corner_;
  bool new_laser_surf_;
  bool new_laser_outlier_;
  bool new_laser_odom_;

  pcl::KdTreeFLANN<PointT>::Ptr kd_corner_map_;
  pcl::KdTreeFLANN<PointT>::Ptr kd_surf_map_;
  pcl::KdTreeFLANN<PointT>::Ptr kd_keyposes_;
  vector<int> point_idx_;
  vector<float> point_dist_;

  pcl::VoxelGrid<PointT> ds_corner_;
  pcl::VoxelGrid<PointT> ds_surf_;
  pcl::VoxelGrid<PointT> ds_outlier_;
  pcl::VoxelGrid<PointT> ds_keyposes_;
  pcl::VoxelGrid<PointT> ds_history_keyframes_;

  double min_keyframe_dist_;

  // 回环检测使用
  deque<PointCloudT::Ptr> recent_corner_keyframes_;
  deque<PointCloudT::Ptr> recent_surf_keyframes_;
  deque<PointCloudT::Ptr> recent_outlier_keyframes_;
  int recent_keyframe_search_num_;
  int latest_frame_id_;
  Eigen::Matrix4d correction_;

  // 无回环检测使用
  PointCloudT::Ptr surround_keyposes_;
  PointCloudT::Ptr surround_keyposes_ds_;
  vector<int> surround_exist_keypose_id_;
  vector<PointCloudT::Ptr> surround_corner_keyframes_;
  vector<PointCloudT::Ptr> surround_surf_keyframes_;
  vector<PointCloudT::Ptr> surround_outlier_keyframes_;
  double surround_keyframe_search_radius_;

  double params_[6];
  Eigen::Vector3d t_map2odom_;
  Eigen::Quaterniond q_map2odom_;
  Eigen::Vector3d t_odom2laser_;
  Eigen::Quaterniond q_odom2laser_;
  Eigen::Vector3d t_map2laser_;
  Eigen::Quaterniond q_map2laser_;

  std::shared_ptr<Node> self_;

public:
  LM() : Node("LM")
  {
    onInit();
  }
  void onInit()
  {
    TicToc t_init;

    RCLCPP_INFO(this->get_logger(), "--------- LaserMapping init --------------");

    surf_last_.reset(new PointCloudT);
    corner_last_.reset(new PointCloudT);
    outlier_last_.reset(new PointCloudT);
    cloud_keyposes_3d_.reset(new PointCloudT);
    cloud_keyposes_6d_.reset(new pcl::PointCloud<PointTypePose>);
    corner_from_map_.reset(new PointCloudT);
    surf_from_map_.reset(new PointCloudT);
    // outlier_from_map_.reset(new PointCloudT);
    corner_from_map_ds_.reset(new PointCloudT);
    surf_from_map_ds_.reset(new PointCloudT);
    laser_corner_.reset(new PointCloudT);
    laser_surf_.reset(new PointCloudT);
    laser_outlier_.reset(new PointCloudT);
    laser_surf_total_.reset(new PointCloudT);
    laser_corner_ds_.reset(new PointCloudT);
    laser_surf_ds_.reset(new PointCloudT);
    laser_outlier_ds_.reset(new PointCloudT);
    laser_surf_total_ds_.reset(new PointCloudT);
    kd_surf_map_.reset(new pcl::KdTreeFLANN<PointT>);
    kd_corner_map_.reset(new pcl::KdTreeFLANN<PointT>);
    kd_keyposes_.reset(new pcl::KdTreeFLANN<PointT>);

    new_laser_surf_ = new_laser_corner_ = new_laser_outlier_ = new_laser_corner_ = false;

    ds_corner_.setLeafSize(0.4, 0.4, 0.4);
    ds_surf_.setLeafSize(0.8, 0.8, 0.8);
    ds_outlier_.setLeafSize(1.0, 1.0, 1.0);
    ds_keyposes_.setLeafSize(1.0, 1.0, 1.0);
    ds_history_keyframes_.setLeafSize(0.4, 0.4, 0.4);

    min_keyframe_dist_ = 1.0;

    surround_keyposes_.reset(new PointCloudT);
    surround_keyposes_ds_.reset(new PointCloudT);

    recent_keyframe_search_num_ = 50;
    surround_keyframe_search_radius_ = 50.;
    latest_frame_id_ = -1;

    for (int i = 0; i < 6; ++i)
    {
      params_[i] = 0.;
    }
    t_map2odom_.setZero();
    q_map2odom_.setIdentity();
    t_odom2laser_.setZero();
    q_odom2laser_.setIdentity();
    t_map2laser_.setZero();
    q_map2laser_.setIdentity();

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam_ = new ISAM2(parameters);
    gtsam::Vector Vector6(6);
    Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
    prior_noise_ = noiseModel::Diagonal::Variances(Vector6);
    odom_noise_ = noiseModel::Diagonal::Variances(Vector6);

    loop_closed_ = false;
    latest_history_frame_id_ = -1;
    latest_keyframe_.reset(new PointCloudT);
    near_history_keyframes_.reset(new PointCloudT);
    history_search_radius_ = 10.;
    history_search_num_ = 25;
    history_fitness_score_ = 0.3;
    loop_closure_enabled_ = true;
    correction_.setIdentity();

    pub_cloud_surround_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/laser_cloud_surround");
    pub_odom_aft_mapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom_aft_mapped");
    pub_keyposes_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/keyposes");
    pub_recent_keyframes_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/recent_keyframes");
    pub_history_keyframes_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/history_keyframes");
    pub_icp_keyframes_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/icp_keyframes");
    srv_save_map_ = this->create_service<std_srvs::srv::Empty>("/save_map", std::bind(&LM::saveMapCB, this, std::placeholders::_1, std::placeholders::_2));

    sub_surf_last_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/surf_last", std::bind(&LM::surfLastHandler, this, std::placeholders::_1));
    sub_corner_last_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/corner_last", std::bind(&LM::cornerLastHandler, this, std::placeholders::_1));
    sub_outlier_last_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/outlier", std::bind(&LM::outlierLastHandler, this, std::placeholders::_1));
    sub_laser_odom_ = this->create_subscription<nav_msgs::msg::Odometry>("/odom/lidar", std::bind(&LM::laserOdomHandler, this, std::placeholders::_1));

    loop_thread_ = std::thread(&LM::loopClosureThread, this);
    visualize_thread_ = std::thread(&LM::visualizeGlobalMapThread, this);

    RCLCPP_INFO(this->get_logger(), "LaserMapping onInit end: %.3fms", t_init.toc());
  }

  void mainLoop(std::shared_ptr<LM> lm)
  {
    self_ = lm;
    tf_broad_ = new tf2_ros::TransformBroadcaster(self_);
    rclcpp::Rate rate(100);
    while (rclcpp::ok())
    {
      if (new_laser_surf_ && new_laser_corner_ && new_laser_outlier_ && new_laser_odom_ &&
          std::abs(time_laser_surf_ - time_laser_corner_) < 0.005 && std::abs(time_laser_surf_ - time_laser_outlier_) < 0.005 && std::abs(time_laser_surf_ - time_laser_odom_) < 0.005)
      {
        new_laser_surf_ = new_laser_corner_ = new_laser_outlier_ = new_laser_odom_ = false;
        static int frame_cnt = 0;
        if (frame_cnt % 2 == 0)
        {
          std::lock_guard<std::mutex> lock(mtx_);
          TicToc t_whole;
          RCLCPP_INFO(this->get_logger(), "transformAssociateToMap");
          TicToc t_tatm;
          transformAssociateToMap();
          RCLCPP_INFO(this->get_logger(), "transformAssociateToMap: %.3fms", t_tatm.toc());
          RCLCPP_INFO(this->get_logger(), "extractSurroundingKeyFrames");
          TicToc t_eskf;
          extractSurroundingKeyFrames();
          RCLCPP_INFO(this->get_logger(), "extractSurroundingKeyFrames: %.3fms", t_eskf.toc());
          RCLCPP_INFO(this->get_logger(), "downsampleCurrentScan");
          TicToc t_dcs;
          downsampleCurrentScan();
          RCLCPP_INFO(this->get_logger(), "downsampleCurrentScan: %.3fms", t_dcs.toc());
          RCLCPP_INFO(this->get_logger(), "scan2MapOptimization");
          TicToc t_s2mo;
          scan2MapOptimization();
          RCLCPP_INFO(this->get_logger(), "scan2MapOptimization: %.3fms", t_s2mo.toc());
          saveKeyFramesAndFactor();
          correctPoses();
          transformUpdate();
          publish();
          RCLCPP_INFO(this->get_logger(), "mapping whole time: %.3fms", t_whole.toc());
        }
        ++frame_cnt;
      }
      rate.sleep();
      rclcpp::spin_some(self_);
    }
  }

  void surfLastHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    laser_surf_->clear();
    pcl::fromROSMsg(*msg, *laser_surf_);
    time_laser_surf_ = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    new_laser_surf_ = true;
  }
  void cornerLastHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    laser_corner_->clear();
    pcl::fromROSMsg(*msg, *laser_corner_);
    time_laser_corner_ = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    new_laser_corner_ = true;
  }
  void outlierLastHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    laser_outlier_->clear();
    pcl::fromROSMsg(*msg, *laser_outlier_);
    time_laser_outlier_ = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    new_laser_outlier_ = true;
  }
  void laserOdomHandler(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    time_laser_odom_ = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    new_laser_odom_ = true;
    t_odom2laser_(0) = msg->pose.pose.position.x;
    t_odom2laser_(1) = msg->pose.pose.position.y;
    t_odom2laser_(2) = msg->pose.pose.position.z;
    q_odom2laser_.w() = msg->pose.pose.orientation.w;
    q_odom2laser_.x() = msg->pose.pose.orientation.x;
    q_odom2laser_.y() = msg->pose.pose.orientation.y;
    q_odom2laser_.z() = msg->pose.pose.orientation.z;
    t_map2laser_ = q_map2odom_ * t_odom2laser_ + t_map2odom_;
    q_map2laser_ = q_map2odom_ * q_odom2laser_;
    if (pub_odom_aft_mapped_->get_publisher_handle() > 0)
    {
      nav_msgs::msg::Odometry::SharedPtr msg(new nav_msgs::msg::Odometry);
      msg->header.stamp = msg->header.stamp;
      msg->header.frame_id = "map";
      msg->child_frame_id = "/laser";
      msg->pose.pose.position.x = t_map2laser_.x();
      msg->pose.pose.position.y = t_map2laser_.y();
      msg->pose.pose.position.z = t_map2laser_.z();
      msg->pose.pose.orientation.w = q_map2laser_.w();
      msg->pose.pose.orientation.x = q_map2laser_.x();
      msg->pose.pose.orientation.y = q_map2laser_.y();
      msg->pose.pose.orientation.z = q_map2laser_.z();
      pub_odom_aft_mapped_->publish(msg);
    }
    geometry_msgs::msg::TransformStamped msg_m2o;
    msg_m2o.header.stamp = msg->header.stamp;
    msg_m2o.header.frame_id = "map";
    msg_m2o.child_frame_id = "/odom";
    msg_m2o.transform.rotation.w = q_map2odom_.w();
    msg_m2o.transform.rotation.x = q_map2odom_.x();
    msg_m2o.transform.rotation.y = q_map2odom_.y();
    msg_m2o.transform.rotation.z = q_map2odom_.z();
    msg_m2o.transform.translation.x = t_map2odom_.x();
    msg_m2o.transform.translation.y = t_map2odom_.y();
    msg_m2o.transform.translation.z = t_map2odom_.z();
    tf_broad_->sendTransform(msg_m2o);
  }

  void transformAssociateToMap()
  {
    t_map2laser_ = q_map2odom_ * t_odom2laser_ + t_map2odom_;
    q_map2laser_ = q_map2odom_ * q_odom2laser_;
  }

  void extractSurroundingKeyFrames()
  {
    TicToc t_data;
    RCLCPP_INFO(this->get_logger(), "extractSurroundingKeyFrames");
    surf_from_map_->clear();
    corner_from_map_->clear();
    surf_from_map_ds_->clear();
    corner_from_map_ds_->clear();
    if (cloud_keyposes_3d_->empty())
    {
      return;
    }
    if (loop_closure_enabled_)
    {
      if (recent_corner_keyframes_.size() < recent_keyframe_search_num_)
      {
        recent_corner_keyframes_.clear();
        recent_surf_keyframes_.clear();
        recent_outlier_keyframes_.clear();
        for (int i = cloud_keyposes_3d_->size() - 1; i >= 0; --i)
        {
          int frame_id = int(cloud_keyposes_3d_->points[i].intensity);
          recent_corner_keyframes_.push_front(transformPointCloud(corner_frames_[frame_id], cloud_keyposes_6d_->points[frame_id]));
          recent_surf_keyframes_.push_front(transformPointCloud(surf_frames_[frame_id], cloud_keyposes_6d_->points[frame_id]));
          recent_outlier_keyframes_.push_front(transformPointCloud(outlier_frames_[frame_id], cloud_keyposes_6d_->points[frame_id]));
          if (recent_corner_keyframes_.size() >= recent_keyframe_search_num_)
          {
            break;
          }
        }
      }
      else
      {
        if (latest_frame_id_ != cloud_keyposes_3d_->size() - 1)
        {
          recent_corner_keyframes_.pop_front();
          recent_surf_keyframes_.pop_front();
          recent_outlier_keyframes_.pop_front();
          latest_frame_id_ = cloud_keyposes_3d_->size() - 1;
          recent_corner_keyframes_.push_back(transformPointCloud(corner_frames_[latest_frame_id_], cloud_keyposes_6d_->points[latest_frame_id_]));
          recent_surf_keyframes_.push_back(transformPointCloud(surf_frames_[latest_frame_id_], cloud_keyposes_6d_->points[latest_frame_id_]));
          recent_outlier_keyframes_.push_back(transformPointCloud(outlier_frames_[latest_frame_id_], cloud_keyposes_6d_->points[latest_frame_id_]));
        }
      }
      for (int i = 0; i < recent_corner_keyframes_.size(); ++i)
      {
        *corner_from_map_ += *recent_corner_keyframes_[i];
        *surf_from_map_ += *recent_surf_keyframes_[i];
        *surf_from_map_ += *recent_outlier_keyframes_[i];
      }
    }
    else
    {
      surround_keyposes_->clear();
      surround_keyposes_ds_->clear();
      PointT cur_laser_pose;
      cur_laser_pose.x = t_map2laser_.x();
      cur_laser_pose.y = t_map2laser_.y();
      cur_laser_pose.z = t_map2laser_.z();
      kd_keyposes_->setInputCloud(cloud_keyposes_3d_);
      kd_keyposes_->radiusSearch(cur_laser_pose, surround_keyframe_search_radius_, point_idx_, point_dist_);
      for (int i = 0; i < point_idx_.size(); ++i)
      {
        surround_keyposes_->points.push_back(cloud_keyposes_3d_->points[point_idx_[i]]);
      }
      ds_keyposes_.setInputCloud(surround_keyposes_);
      ds_keyposes_.filter(*surround_keyposes_ds_);

      int len = surround_keyposes_ds_->size();
      // TODO: 感觉这两步没啥必要，也就是节省了 transformPointCloud 的次数，直接用 surround_keyposes_ds_ 中的不行么
      // surrounding 中保存的是附近位姿和点云，当做局部的 target 地图来匹配
      // 1. 对于不在当前采样的附近位姿的 surroundingExistingKeyPoses，将其剔除
      for (int i = 0; i < surround_exist_keypose_id_.size(); ++i)
      {
        bool exist_flag = false;
        for (int j = 0; j < len; ++j)
        {
          if (surround_exist_keypose_id_[i] == int(surround_keyposes_ds_->points[i].intensity))
          {
            exist_flag = true;
            break;
          }
        }
        if (false == exist_flag)
        {
          surround_exist_keypose_id_.erase(surround_exist_keypose_id_.begin() + i);
          surround_corner_keyframes_.erase(surround_corner_keyframes_.begin() + i);
          surround_surf_keyframes_.erase(surround_surf_keyframes_.begin() + i);
          surround_outlier_keyframes_.erase(surround_outlier_keyframes_.begin() + i);
          --i;
        }
      }
      // 2. 对于当前采样的附近位姿，如果 surroundingExistingKeyPoses 中没有，就将其添加进去
      for (int i = 0; i < len; ++i)
      {
        bool exist_flag = false;
        for (auto iter = surround_exist_keypose_id_.begin(); iter != surround_exist_keypose_id_.end(); ++iter)
        {
          if ((*iter) == int(surround_keyposes_ds_->points[i].intensity))
          {
            exist_flag = true;
            break;
          }
        }
        if (false == exist_flag)
        {
          int frame_id = int(surround_keyposes_ds_->points[i].intensity);
          surround_exist_keypose_id_.push_back(frame_id);
          surround_surf_keyframes_.push_back(transformPointCloud(surf_frames_[frame_id], cloud_keyposes_6d_->points[frame_id]));
          surround_corner_keyframes_.push_back(transformPointCloud(corner_frames_[frame_id], cloud_keyposes_6d_->points[frame_id]));
          surround_outlier_keyframes_.push_back(transformPointCloud(outlier_frames_[frame_id], cloud_keyposes_6d_->points[frame_id]));
        }
      }
      for (int i = 0; i < surround_exist_keypose_id_.size(); ++i)
      {
        *surf_from_map_ += *surround_surf_keyframes_[i];
        *corner_from_map_ += *surround_corner_keyframes_[i];
        *surf_from_map_ += *surround_outlier_keyframes_[i];
      }
    }

    TicToc t_ds;
    ds_surf_.setInputCloud(surf_from_map_);
    ds_surf_.filter(*surf_from_map_ds_);
    ds_corner_.setInputCloud(corner_from_map_);
    ds_corner_.filter(*corner_from_map_ds_);
    RCLCPP_INFO(this->get_logger(), "before ds, surf points: %d, corner points: %d, time: %.3fms", surf_from_map_->size(), corner_from_map_->size());
    RCLCPP_INFO(this->get_logger(), "surf points: %d, corner points: %d, time: %.3fms", surf_from_map_ds_->size(), corner_from_map_ds_->size(), t_data.toc());
    RCLCPP_INFO(this->get_logger(), "downsize time: %.3fms", t_ds.toc());
  }

  void downsampleCurrentScan()
  {
    TicToc t_ds;
    laser_corner_ds_->clear();
    ds_corner_.setInputCloud(laser_corner_);
    ds_corner_.filter(*laser_corner_ds_);
    laser_surf_ds_->clear();
    ds_surf_.setInputCloud(laser_surf_);
    ds_surf_.filter(*laser_surf_ds_);
    laser_outlier_ds_->clear();
    ds_outlier_.setInputCloud(laser_outlier_);
    ds_outlier_.filter(*laser_outlier_ds_);
    laser_surf_total_->clear();
    laser_surf_total_ds_->clear();
    *laser_surf_total_ += *laser_surf_ds_;
    *laser_surf_total_ += *laser_outlier_ds_;
    ds_surf_.setInputCloud(laser_surf_total_);
    ds_surf_.filter(*laser_surf_total_ds_);
    RCLCPP_INFO(this->get_logger(), "downsampleCurrentScan: %.3fms", t_ds.toc());
    RCLCPP_INFO(this->get_logger(), "before downsample: corner size %d, surf size %d", laser_corner_->points.size(), laser_surf_total_->points.size() + laser_outlier_->points.size());
    RCLCPP_INFO(this->get_logger(), "after downsample: corner size %d, surf size %d", laser_corner_ds_->points.size(), laser_surf_total_ds_->points.size());
  }

  void scan2MapOptimization()
  {
    if (laser_corner_ds_->points.size() < 10 || laser_surf_total_->points.size() < 100 || corner_from_map_ds_->size() < 10)
    {
      RCLCPP_WARN(this->get_logger(), "few feature points to registration");
      return;
    }
    TicToc t_opt, t_tree;
    kd_corner_map_->setInputCloud(corner_from_map_ds_);
    kd_surf_map_->setInputCloud(surf_from_map_ds_);
    RCLCPP_INFO(this->get_logger(), "build kdtree time: %.3fms", t_tree.toc());

    for (int iter_cnt = 0; iter_cnt < 2; ++iter_cnt)
    {
      TicToc t_data;
      ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
      ceres::Problem::Options problem_options;
      ceres::Problem problem(problem_options);
      // problem.AddParameterBlock(parameters, 4, q_parameterization);
      problem.AddParameterBlock(params_, 6);

      int corner_correspondace = 0, surf_correnspondance = 0;
      int test_p = 0;
      for (int i = 0; i < laser_corner_ds_->size(); ++i)
      {
        PointT point_sel;
        pointAssociateToMap(laser_corner_ds_->points[i], point_sel);
        kd_corner_map_->nearestKSearch(point_sel, 5, point_idx_, point_dist_);
        if (point_dist_[4] < 1.0)
        {
          std::vector<Eigen::Vector3d> nearCorners;
          Eigen::Vector3d center(0., 0., 0.);
          for (int j = 0; j < 5; j++)
          {
            Eigen::Vector3d tmp(corner_from_map_ds_->points[point_idx_[j]].x,
                                corner_from_map_ds_->points[point_idx_[j]].y,
                                corner_from_map_ds_->points[point_idx_[j]].z);
            center = center + tmp;
            nearCorners.push_back(tmp);
          }
          center = center / 5.0;

          Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
          for (int j = 0; j < 5; j++)
          {
            Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
            covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
          }

          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

          // if is indeed line feature
          // note Eigen library sort eigenvalues in increasing order
          Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
          Eigen::Vector3d cp(laser_corner_ds_->points[i].x, laser_corner_ds_->points[i].y, laser_corner_ds_->points[i].z);
          if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
          {
            Eigen::Vector3d point_on_line = center;
            Eigen::Vector3d lpj = 0.1 * unit_direction + point_on_line;
            Eigen::Vector3d lpl = -0.1 * unit_direction + point_on_line;
            problem.AddResidualBlock(new LidarEdgeCostFunction(cp, lpj, lpl),
                                     loss_function, params_);
            ++corner_correspondace;
          }
          else
          {
            ++test_p;
          }
        }
      }
      RCLCPP_INFO(this->get_logger(), "test_p: %d", test_p);
      for (int i = 0; i < laser_surf_total_ds_->size(); ++i)
      {
        PointT point_sel;
        pointAssociateToMap(laser_surf_total_ds_->points[i], point_sel);
        kd_surf_map_->nearestKSearch(point_sel, 5, point_idx_, point_dist_);
        Eigen::Matrix<double, 5, 3> matA0;
        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
        if (point_dist_[4] < 1.0)
        {
          for (int j = 0; j < 5; j++)
          {
            matA0(j, 0) = surf_from_map_ds_->points[point_idx_[j]].x;
            matA0(j, 1) = surf_from_map_ds_->points[point_idx_[j]].y;
            matA0(j, 2) = surf_from_map_ds_->points[point_idx_[j]].z;
          }
          // find the norm of plane
          Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
          double negative_OA_dot_norm = 1 / norm.norm();
          norm.normalize();

          // Here n(pa, pb, pc) is unit norm of plane
          bool planeValid = true;
          for (int j = 0; j < 5; j++)
          {
            // if OX * n > 0.2, then plane is not fit well
            if (fabs(norm(0) * surf_from_map_ds_->points[point_idx_[j]].x +
                     norm(1) * surf_from_map_ds_->points[point_idx_[j]].y +
                     norm(2) * surf_from_map_ds_->points[point_idx_[j]].z + negative_OA_dot_norm) > 0.2)
            {
              planeValid = false;
              RCLCPP_WARN_ONCE(this->get_logger(), "plane is not fit well");
              break;
            }
          }
          if (planeValid)
          {
            Eigen::Vector3d cp(laser_surf_total_ds_->points[i].x, laser_surf_total_ds_->points[i].y, laser_surf_total_ds_->points[i].z);
            problem.AddResidualBlock(new LidarPlaneCostFunction(cp, norm, negative_OA_dot_norm),
                                     loss_function, params_);
            // TODO: 先解决 corner 数量过少的问题，少了十倍
            ++surf_correnspondance;
          }
        }
      }

      RCLCPP_INFO(this->get_logger(), "mapping data assosiation time %.3fms", t_data.toc());
      RCLCPP_INFO(this->get_logger(), "corner_correspondance: %d, surf_correspondance: %d", corner_correspondace, surf_correnspondance);

      TicToc t_solver;
      ceres::Solver::Options options;
      options.linear_solver_type = ceres::DENSE_QR;
      options.max_num_iterations = 20;
      options.minimizer_progress_to_stdout = false;
      options.check_gradients = false;
      options.gradient_check_relative_precision = 1e-4;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      RCLCPP_INFO(this->get_logger(), "mapping solver time %.3fms", t_solver.toc());
      cout << summary.BriefReport() << endl;
    }
  }

  void transformUpdate()
  {
    q_map2laser_ = Eigen::AngleAxisd(params_[5], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(params_[4], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(params_[3], Eigen::Vector3d::UnitX());
    t_map2laser_.x() = params_[0];
    t_map2laser_.y() = params_[1];
    t_map2laser_.z() = params_[2];
    q_map2odom_ = q_map2laser_ * q_odom2laser_.inverse();
    t_map2odom_ = t_map2laser_ - q_map2odom_ * t_odom2laser_;
  }

  bool saveKeyFramesAndFactor()
  {
    if (cloud_keyposes_3d_->points.empty())
    {
      gtSAMgraph_.add(PriorFactor<Pose3>(0, Pose3(Rot3::Quaternion(q_map2laser_.w(), q_map2laser_.x(), q_map2laser_.y(), q_map2laser_.z()), Point3(t_map2laser_.x(), t_map2laser_.y(), t_map2laser_.z())), prior_noise_));
      init_estimate_.insert(0, Pose3(Rot3::Quaternion(q_map2laser_.w(), q_map2laser_.x(), q_map2laser_.y(), q_map2laser_.z()), Point3(t_map2laser_.x(), t_map2laser_.y(), t_map2laser_.z())));
    }
    else
    {
      const auto &pre_pose = cloud_keyposes_6d_->points[cloud_keyposes_3d_->points.size() - 1];
      if (std::pow(t_map2laser_.x() - pre_pose.x, 2) +
              std::pow(t_map2laser_.y() - pre_pose.y, 2) +
              std::pow(t_map2laser_.z() - pre_pose.z, 2) <
          min_keyframe_dist_)
      {
        // too close
        return false;
      }

      gtsam::Pose3 pose_from = Pose3(Rot3::RzRyRx(pre_pose.roll, pre_pose.pitch, pre_pose.yaw), Point3(pre_pose.x, pre_pose.y, pre_pose.z));
      gtsam::Pose3 pose_to = Pose3(Rot3::Quaternion(q_map2laser_.w(), q_map2laser_.x(), q_map2laser_.y(), q_map2laser_.z()), Point3(t_map2laser_.x(), t_map2laser_.y(), t_map2laser_.z()));
      gtSAMgraph_.add(BetweenFactor<Pose3>(cloud_keyposes_3d_->points.size() - 1, cloud_keyposes_3d_->points.size(), pose_from.between(pose_to), odom_noise_));
      init_estimate_.insert(cloud_keyposes_3d_->points.size(), Pose3(Rot3::Quaternion(q_map2laser_.w(), q_map2laser_.x(), q_map2laser_.y(), q_map2laser_.z()), Point3(t_map2laser_.x(), t_map2laser_.y(), t_map2laser_.z())));
    }

    isam_->update(gtSAMgraph_, init_estimate_);
    isam_->update();

    gtSAMgraph_.resize(0);
    init_estimate_.clear();

    PointT this_pose_3d;
    PointTypePose this_pose_6d;
    gtsam::Pose3 latest_estimate;
    isam_cur_estimate_ = isam_->calculateEstimate();
    latest_estimate = isam_cur_estimate_.at<gtsam::Pose3>(isam_cur_estimate_.size() - 1);

    this_pose_6d.x = this_pose_3d.x = latest_estimate.translation().x();
    this_pose_6d.y = this_pose_3d.y = latest_estimate.translation().y();
    this_pose_6d.z = this_pose_3d.z = latest_estimate.translation().z();
    this_pose_6d.intensity = this_pose_3d.intensity = cloud_keyposes_3d_->points.size() + 0.1;
    this_pose_6d.roll = latest_estimate.rotation().roll();
    this_pose_6d.pitch = latest_estimate.rotation().pitch();
    this_pose_6d.yaw = latest_estimate.rotation().yaw();
    this_pose_6d.time = time_laser_corner_;
    cloud_keyposes_3d_->points.push_back(this_pose_3d);
    cloud_keyposes_6d_->points.push_back(this_pose_6d);

    params_[0] = this_pose_6d.x;
    params_[1] = this_pose_6d.y;
    params_[2] = this_pose_6d.z;
    params_[3] = this_pose_6d.roll;
    params_[4] = this_pose_6d.pitch;
    params_[5] = this_pose_6d.yaw;

    PointCloudT::Ptr this_corner_ds(new PointCloudT);
    PointCloudT::Ptr this_surf_ds(new PointCloudT);
    PointCloudT::Ptr this_outlier_ds(new PointCloudT);

    pcl::copyPointCloud(*laser_corner_ds_, *this_corner_ds);
    pcl::copyPointCloud(*laser_surf_ds_, *this_surf_ds);
    pcl::copyPointCloud(*laser_outlier_ds_, *this_outlier_ds);

    corner_frames_.push_back(this_corner_ds);
    surf_frames_.push_back(this_surf_ds);
    outlier_frames_.push_back(this_outlier_ds);

    return true;
  }

  void correctPoses()
  {
    if (loop_closed_)
    {
      recent_surf_keyframes_.clear();
      recent_corner_keyframes_.clear();
      recent_outlier_keyframes_.clear();
      RCLCPP_WARN(this->get_logger(), "correctPoses");
      int num_poses = isam_cur_estimate_.size();
      RCLCPP_INFO(this->get_logger(), "num_poses: %d", num_poses);
      for (int i = 0; i < num_poses; ++i)
      {
        cloud_keyposes_6d_->points[i].x = cloud_keyposes_3d_->points[i].x = isam_cur_estimate_.at<gtsam::Pose3>(i).translation().x();
        cloud_keyposes_6d_->points[i].y = cloud_keyposes_3d_->points[i].y = isam_cur_estimate_.at<gtsam::Pose3>(i).translation().y();
        cloud_keyposes_6d_->points[i].z = cloud_keyposes_3d_->points[i].z = isam_cur_estimate_.at<gtsam::Pose3>(i).translation().z();
        cloud_keyposes_6d_->points[i].roll = isam_cur_estimate_.at<gtsam::Pose3>(i).rotation().roll();
        cloud_keyposes_6d_->points[i].pitch = isam_cur_estimate_.at<gtsam::Pose3>(i).rotation().pitch();
        cloud_keyposes_6d_->points[i].yaw = isam_cur_estimate_.at<gtsam::Pose3>(i).rotation().yaw();
      }
      q_map2odom_ = correction_.block<3, 3>(0, 0) * q_map2odom_.toRotationMatrix();
      t_map2odom_.matrix() = correction_.block<3, 3>(0, 0) * t_map2odom_.matrix() + correction_.block<3, 1>(0, 3);

      loop_closed_ = false;
    }
  }

  void publish()
  {
    if (pub_keyposes_->get_publisher_handle() > 0)
    {
      sensor_msgs::msg::PointCloud2::SharedPtr msg(new sensor_msgs::msg::PointCloud2);
      pcl::toROSMsg(*cloud_keyposes_3d_, *msg);
      msg->header.stamp.sec = int(time_laser_odom_);
      msg->header.stamp.nanosec = (time_laser_odom_ - msg->header.stamp.sec) * 1e9;
      msg->header.frame_id = "map";
      pub_keyposes_->publish(msg);
    }
  }

  void visualizeGlobalMapThread()
  {
    rclcpp::Rate rate(0.2);
    while (rclcpp::ok)
    {
      if (pub_cloud_surround_->get_publisher_handle() > 0)
      {
        sensor_msgs::msg::PointCloud2::SharedPtr msg(new sensor_msgs::msg::PointCloud2);
        PointCloudT::Ptr pc_map(new PointCloudT);
        for (int i = 0; i < cloud_keyposes_3d_->size(); ++i)
        {
          *pc_map += *transformPointCloud(surf_frames_[i], cloud_keyposes_6d_->points[i]);
          *pc_map += *transformPointCloud(corner_frames_[i], cloud_keyposes_6d_->points[i]);
          *pc_map += *transformPointCloud(outlier_frames_[i], cloud_keyposes_6d_->points[i]);
        }
        pcl::toROSMsg(*pc_map, *msg);
        msg->header.stamp.sec = int(time_laser_odom_);
        msg->header.stamp.nanosec = (time_laser_odom_ - msg->header.stamp.sec) * 1e9;
        msg->header.frame_id = "map";
        pub_cloud_surround_->publish(msg);
      }
      if (pub_recent_keyframes_->get_publisher_handle() > 0)
      {
        sensor_msgs::msg::PointCloud2::SharedPtr msg(new sensor_msgs::msg::PointCloud2);
        PointCloudT::Ptr pc_map(new PointCloudT);
        *pc_map += *corner_from_map_ds_;
        *pc_map += *surf_from_map_ds_;
        pcl::toROSMsg(*pc_map, *msg);
        msg->header.stamp.sec = int(time_laser_odom_);
        msg->header.stamp.nanosec = (time_laser_odom_ - msg->header.stamp.sec) * 1e9;
        msg->header.frame_id = "map";
        pub_recent_keyframes_->publish(msg);
      }
      rate.sleep();
    }
  }

  void loopClosureThread()
  {
    if (!loop_closure_enabled_)
    {
      return;
    }
    rclcpp::Rate rate(1);
    while (rclcpp::ok())
    {
      RCLCPP_INFO(this->get_logger(), "performLoopClosure");
      TicToc t_plc;
      performLoopClosure();
      RCLCPP_INFO(this->get_logger(), "performLoopClosure: %.3fms", t_plc.toc());
      rate.sleep();
    }
  }

  /**
 * @brief 回环检测及位姿图更新
 * ICP 匹配添加回环约束
 */
  void performLoopClosure()
  {
    if (cloud_keyposes_3d_->points.empty())
    {
      return;
    }

    if (!detectLoopClosure())
    {
      return;
    }
    else
    {
      RCLCPP_WARN(this->get_logger(), "detected loop closure");
    }

    TicToc t_icp;

    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setMaxCorrespondenceDistance(100);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    icp.setInputSource(latest_keyframe_);
    icp.setInputTarget(near_history_keyframes_);
    PointCloudT::Ptr unused_result(new PointCloudT());
    Eigen::Matrix4f initial_guess(Eigen::Matrix4f::Identity());
    initial_guess.block<3, 3>(0, 0) = (Eigen::AngleAxisf(cloud_keyposes_6d_->points[latest_history_frame_id_].yaw, Eigen::Vector3f::UnitZ()) *
                                       Eigen::AngleAxisf(cloud_keyposes_6d_->points[latest_history_frame_id_].pitch, Eigen::Vector3f::UnitY()) *
                                       Eigen::AngleAxisf(cloud_keyposes_6d_->points[latest_history_frame_id_].roll, Eigen::Vector3f::UnitX()))
                                          .toRotationMatrix();
    initial_guess(0, 3) = cloud_keyposes_6d_->points[latest_history_frame_id_].x;
    initial_guess(1, 3) = cloud_keyposes_6d_->points[latest_history_frame_id_].y;
    initial_guess(2, 3) = cloud_keyposes_6d_->points[latest_history_frame_id_].z;
    icp.align(*unused_result);

    bool has_converged = icp.hasConverged();
    double fitness_score = icp.getFitnessScore();
    Eigen::Matrix4f correction_frame = icp.getFinalTransformation();

    if (has_converged == false || fitness_score > history_fitness_score_)
    {
      RCLCPP_WARN(this->get_logger(), "loop cannot closed");
      return;
    }
    else
    {
      RCLCPP_WARN(this->get_logger(), "loop closed");
      if (pub_icp_keyframes_->get_publisher_handle() > 0)
      {
        PointCloudT::Ptr closed_cloud(new PointCloudT());
        pcl::transformPointCloud(*latest_keyframe_, *closed_cloud, correction_frame);
        sensor_msgs::msg::PointCloud2::SharedPtr msg(new sensor_msgs::msg::PointCloud2);
        pcl::toROSMsg(*closed_cloud, *msg);
        msg->header.stamp.sec = int(cloud_keyposes_6d_->points[latest_history_frame_id_].time);
        msg->header.stamp.nanosec = (cloud_keyposes_6d_->points[latest_history_frame_id_].time - msg->header.stamp.sec) * 1e9;
        msg->header.frame_id = "map";
        pub_icp_keyframes_->publish(msg);
      }
    }

    Eigen::Matrix4f t_wrong = initial_guess;
    Eigen::Matrix4f t_correct = correction_frame * t_wrong;
    Eigen::Quaternionf r_correct(t_correct.block<3, 3>(0, 0));
    gtsam::Pose3 pose_from = gtsam::Pose3(Rot3::Quaternion(r_correct.w(), r_correct.x(), r_correct.y(), r_correct.z()),
                                          Point3(t_correct(0, 3), t_correct(1, 3), t_correct(2, 3)));
    gtsam::Pose3 pose_to = gtsam::Pose3(Rot3::RzRyRx(cloud_keyposes_6d_->points[closest_history_frame_id_].roll, cloud_keyposes_6d_->points[closest_history_frame_id_].pitch, cloud_keyposes_6d_->points[closest_history_frame_id_].yaw),
                                        Point3(cloud_keyposes_6d_->points[closest_history_frame_id_].x, cloud_keyposes_6d_->points[closest_history_frame_id_].y, cloud_keyposes_6d_->points[closest_history_frame_id_].z));
    float noise_score = fitness_score;
    gtsam::Vector vector6(6);
    vector6 << noise_score, noise_score, noise_score, noise_score, noise_score, noise_score;
    constraint_noise_ = noiseModel::Diagonal::Variances(vector6);

    Eigen::Quaternionf tmp_q(t_correct.block<3, 3>(0, 0));
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf2::Quaternion(tmp_q.x(), tmp_q.y(), tmp_q.z(), tmp_q.w())).getRPY(roll, pitch, yaw);

    std::lock_guard<std::mutex> lock(mtx_);
    gtSAMgraph_.add(BetweenFactor<gtsam::Pose3>(latest_history_frame_id_, closest_history_frame_id_, pose_from.between(pose_to), constraint_noise_));
    isam_->update(gtSAMgraph_);
    isam_->update();
    gtSAMgraph_.resize(0);
    correction_ = correction_frame.cast<double>();

    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "Time elapsed: " << t_icp.toc() << "ms" << std::endl;
    std::cout << "Number of source points: " << latest_keyframe_->size() << " points." << std::endl;
    std::cout << "target: " << near_history_keyframes_->points.size() << " points." << std::endl;
    std::cout << "ICP has converged: " << has_converged << std::endl;
    std::cout << "Fitness score: " << fitness_score << std::endl;
    std::cout << "initial (x,y,z,roll,pitch,yaw):" << std::endl;
    std::cout << "(" << initial_guess(0, 3) << ", " << initial_guess(1, 3) << ", " << initial_guess(2, 3) << ", "
              << cloud_keyposes_6d_->points[latest_history_frame_id_].roll << ", " << cloud_keyposes_6d_->points[latest_history_frame_id_].pitch << ", " << cloud_keyposes_6d_->points[latest_history_frame_id_].yaw << ")" << std::endl;
    std::cout << "final (x,y,z,roll,pitch,yaw):" << std::endl;
    std::cout << "(" << t_correct(0, 3) << ", " << t_correct(1, 3) << ", " << t_correct(2, 3) << ", "
              << roll << ", " << pitch << ", " << yaw << ")" << std::endl;
    std::cout << "Transformation Matrix:" << std::endl;
    std::cout << t_correct << std::endl;
    std::cout << "shift: " << std::sqrt(std::pow(t_correct(0, 3) - initial_guess(0, 3), 2) + std::pow(t_correct(1, 3) - initial_guess(1, 3), 2) + std::pow(t_correct(2, 3) - initial_guess(2, 3), 2)) << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;

    loop_closed_ = true;
  }

  /**
 * @brief 基于里程计的回环检测，直接提取当前位姿附近(radiusSearch)的 keyframe 作为 icp 的 target
 * 
 */
  bool detectLoopClosure()
  {
    if (cloud_keyposes_3d_->empty())
    {
      return false;
    }
    RCLCPP_INFO(this->get_logger(), "detectLoopClosure");
    TicToc t_dlc;
    latest_keyframe_->clear();
    near_history_keyframes_->clear();

    std::lock_guard<std::mutex> lock(mtx_);

    PointT cur_pose;
    cur_pose.x = t_map2laser_.x();
    cur_pose.y = t_map2laser_.y();
    cur_pose.z = t_map2laser_.z();
    kd_keyposes_->setInputCloud(cloud_keyposes_3d_);
    kd_keyposes_->radiusSearch(cur_pose, history_search_radius_, point_idx_, point_dist_);

    latest_history_frame_id_ = cloud_keyposes_3d_->size() - 1;
    closest_history_frame_id_ = -1;
    for (int i = 0; i < point_idx_.size(); ++i)
    {
      // 时间太短不做回环
      if (cloud_keyposes_6d_->points[latest_history_frame_id_].time - cloud_keyposes_6d_->points[point_idx_[i]].time > 30.)
      {
        closest_history_frame_id_ = point_idx_[i];
        break;
      }
    }
    if (closest_history_frame_id_ == -1)
    {
      return false;
    }

    *latest_keyframe_ += *transformPointCloud(surf_frames_[latest_history_frame_id_], cloud_keyposes_6d_->points[latest_history_frame_id_]);
    *latest_keyframe_ += *transformPointCloud(corner_frames_[latest_history_frame_id_], cloud_keyposes_6d_->points[latest_history_frame_id_]);
    *latest_keyframe_ += *transformPointCloud(outlier_frames_[latest_history_frame_id_], cloud_keyposes_6d_->points[latest_history_frame_id_]);

    PointCloudT::Ptr tmp_cloud(new PointCloudT());
    for (int i = -history_search_num_, j; i <= history_search_num_; ++i)
    {
      j = closest_history_frame_id_ + i;
      if (j < 0 || j >= latest_history_frame_id_)
      {
        continue;
      }
      *tmp_cloud += *transformPointCloud(surf_frames_[j], cloud_keyposes_6d_->points[j]);
      *tmp_cloud += *transformPointCloud(corner_frames_[j], cloud_keyposes_6d_->points[j]);
      *tmp_cloud += *transformPointCloud(outlier_frames_[j], cloud_keyposes_6d_->points[j]);
    }

    ds_history_keyframes_.setInputCloud(tmp_cloud);
    ds_history_keyframes_.filter(*near_history_keyframes_);

    if (pub_history_keyframes_->get_publisher_handle() > 0)
    {
      sensor_msgs::msg::PointCloud2::SharedPtr msg(new sensor_msgs::msg::PointCloud2);
      pcl::toROSMsg(*near_history_keyframes_, *msg);
      msg->header.stamp.sec = int(cloud_keyposes_6d_->points[latest_history_frame_id_].time);
      msg->header.stamp.nanosec = (cloud_keyposes_6d_->points[latest_history_frame_id_].time - msg->header.stamp.sec) * 1e9;
      msg->header.frame_id = "map";
      pub_history_keyframes_->publish(msg);
    }
    RCLCPP_INFO(this->get_logger(), "detectLoopClosure: %.3fms", t_dlc.toc());
    return true;
  }
  PointCloudT::Ptr transformPointCloud(const PointCloudT::ConstPtr cloud_in, const PointTypePose &trans)
  {
    Eigen::Matrix4f this_transformation(Eigen::Matrix4f::Identity());
    this_transformation.block<3, 3>(0, 0) = (Eigen::AngleAxisf(trans.yaw, Eigen::Vector3f::UnitZ()) *
                                             Eigen::AngleAxisf(trans.pitch, Eigen::Vector3f::UnitY()) *
                                             Eigen::AngleAxisf(trans.roll, Eigen::Vector3f::UnitX()))
                                                .toRotationMatrix();
    this_transformation(0, 3) = trans.x;
    this_transformation(1, 3) = trans.y;
    this_transformation(2, 3) = trans.z;
    PointCloudT::Ptr tf_cloud(new PointCloudT());
    pcl::transformPointCloud(*cloud_in, *tf_cloud, this_transformation);
    return tf_cloud;
  }
  void pointAssociateToMap(const PointT &p_in, PointT &p_out)
  {
    Eigen::Vector3d out = q_map2laser_ * Eigen::Vector3d(p_in.x, p_in.y, p_in.z) + t_map2laser_;
    p_out.x = out.x();
    p_out.y = out.y();
    p_out.z = out.z();
    p_out.intensity = p_in.intensity;
  }
  PointCloudT::Ptr transformPointCloud(const PointCloudT::ConstPtr cloud_in, const PointTypePose &trans, int idx)
  {
    PointCloudT::Ptr tf_cloud = transformPointCloud(cloud_in, trans);
    for (auto &p : tf_cloud->points)
    {
      p.intensity = idx;
    }
    return tf_cloud;
  }
  bool saveMapCB(std_srvs::srv::Empty::Request::SharedPtr req, std_srvs::srv::Empty::Response::SharedPtr res)
  {
    // cloudKeyPoses3D cloudKeyPoses6D cornerCloudKeyFrames surfCloudKeyFrames outlierCloudKeyFrames
    PointCloudT::Ptr map_keypose(new PointCloudT);
    PointCloudT::Ptr map_corner(new PointCloudT);
    PointCloudT::Ptr map_surf(new PointCloudT);
    PointCloudT::Ptr map_outlier(new PointCloudT);

    for (int i = 0; i < cloud_keyposes_3d_->points.size(); ++i)
    {
      PointT p = cloud_keyposes_3d_->points[i];
      p.intensity = i;
      map_keypose->points.push_back(p);
    }

    PointCloudT::Ptr tmp(new PointCloudT);

    for (int i = 0; i < cloud_keyposes_3d_->points.size(); ++i)
    {
      tmp = transformPointCloud(corner_frames_[i], cloud_keyposes_6d_->points[i], i);
      map_corner->points.insert(map_corner->points.end(), tmp->points.begin(), tmp->points.end());

      tmp = transformPointCloud(surf_frames_[i], cloud_keyposes_6d_->points[i], i);
      map_surf->points.insert(map_surf->points.end(), tmp->points.begin(), tmp->points.end());

      tmp = transformPointCloud(outlier_frames_[i], cloud_keyposes_6d_->points[i], i);
      map_outlier->points.insert(map_outlier->points.end(), tmp->points.begin(), tmp->points.end());
    }

    map_keypose->width = map_keypose->points.size();
    map_keypose->height = 1;
    map_keypose->is_dense = false;
    map_corner->width = map_corner->points.size();
    map_corner->height = 1;
    map_corner->is_dense = true;
    map_surf->width = map_surf->points.size();
    map_surf->height = 1;
    map_surf->is_dense = true;
    map_outlier->width = map_outlier->points.size();
    map_outlier->height = 1;
    map_outlier->is_dense = true;

    pcl::io::savePCDFile("/home/zh/keypose.pcd", *map_keypose);
    pcl::io::savePCDFile("/home/zh/corner.pcd", *map_corner);
    pcl::io::savePCDFile("/home/zh/surf.pcd", *map_surf);
    pcl::io::savePCDFile("/home/zh/outlier.pcd", *map_outlier);

    return true;
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto lm = std::make_shared<LM>();
  lm->mainLoop(lm);
  rclcpp::spin(lm);
  rclcpp::shutdown();
  return 0;
}