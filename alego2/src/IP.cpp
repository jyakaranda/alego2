/**
 * @file IP.cpp
 * @author heng zhang (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2019-07-11
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include "alego2/utility.h"

using namespace std;

class IP : public rclcpp::Node
{
private:
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr pub_tf_;
  rclcpp::Publisher<alego2_msgs::msg::CloudInfo>::SharedPtr pub_seg_info_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_segmented_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_outlier_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pc_;
  sensor_msgs::msg::PointCloud2::Ptr segmented_cloud_msg_;
  sensor_msgs::msg::PointCloud2::Ptr outlier_cloud_msg_;
  alego2_msgs::msg::CloudInfo::Ptr seg_info_msg_;

  PointCloudT::Ptr full_cloud_;
  PointCloudT::Ptr segmented_cloud_;
  PointCloudT::Ptr outlier_cloud_;

  Eigen::Matrix<double, Eigen::Dynamic, Horizon_SCAN> range_mat_;
  Eigen::Matrix<int, Eigen::Dynamic, Horizon_SCAN> label_mat_;
  Eigen::Matrix<bool, Eigen::Dynamic, Horizon_SCAN> ground_mat_;

  int label_cnt_;
  vector<pair<int, int>> neighbor_iter_;

  // params

public:
  IP() : Node("IP")
  {
    onInit();
  }

  void onInit()
  {
    RCLCPP_INFO(this->get_logger(), "---------- ImageProjection init -----------");
    TicToc t_init;

    segmented_cloud_msg_.reset(new sensor_msgs::msg::PointCloud2);
    outlier_cloud_msg_.reset(new sensor_msgs::msg::PointCloud2);
    seg_info_msg_.reset(new alego2_msgs::msg::CloudInfo);

    seg_info_msg_->start_ring_index.resize(N_SCAN);
    seg_info_msg_->end_ring_index.resize(N_SCAN);
    seg_info_msg_->segmented_cloud_col_id.assign(N_SCAN * Horizon_SCAN, false);
    seg_info_msg_->segmented_cloud_ground_flag.assign(N_SCAN * Horizon_SCAN, 0);
    seg_info_msg_->segmented_cloud_range.assign(N_SCAN * Horizon_SCAN, 0);
    full_cloud_.reset(new PointCloudT());
    segmented_cloud_.reset(new PointCloudT());
    outlier_cloud_.reset(new PointCloudT());
    PointT nan_p;
    nan_p.intensity = -1;
    full_cloud_->points.resize(N_SCAN * Horizon_SCAN);
    std::fill(full_cloud_->points.begin(), full_cloud_->points.end(), nan_p);

    range_mat_.resize(N_SCAN, Eigen::NoChange);
    label_mat_.resize(N_SCAN, Eigen::NoChange);
    ground_mat_.resize(N_SCAN, Eigen::NoChange);
    range_mat_.setConstant(std::numeric_limits<double>::max());
    label_mat_.setZero();
    ground_mat_.setZero();
    label_cnt_ = 1;
    neighbor_iter_.emplace_back(-1, 0);
    neighbor_iter_.emplace_back(1, 0);
    neighbor_iter_.emplace_back(0, -1);
    neighbor_iter_.emplace_back(0, 1);

    pub_segmented_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/segmented_cloud");
    pub_outlier_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/outlier");
    pub_seg_info_ = this->create_publisher<alego2_msgs::msg::CloudInfo>("/seg_info");
    sub_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("lslidar_point_cloud", std::bind(&IP::pcCB, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "ImageProjection onInit end: %.3fms", t_init.toc());
  }

  template <typename PointT>
  void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
  {
    if (&cloud_in != &cloud_out)
    {
      cloud_out.header = cloud_in.header;
      cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
      if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
        continue;
      cloud_out.points[j] = cloud_in.points[i];
      j++;
    }
    if (j != cloud_in.points.size())
    {
      cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
  }

  void pcCB(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    TicToc t_whole;

    seg_info_msg_->header = msg->header;
    PointCloudT::Ptr cloud_in(new PointCloudT());
    pcl::fromROSMsg(*msg, *cloud_in);
    // RCLCPP_INFO(this->get_logger(), "cloud_in size: %d", cloud_in->points.size());

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, indices);
    removeClosedPointCloud(*cloud_in, *cloud_in, 1.0);

    int cloud_size = cloud_in->points.size();
    seg_info_msg_->start_orientation = -atan2(cloud_in->points[0].y, cloud_in->points[0].x);
    seg_info_msg_->end_orientation = -atan2(cloud_in->points[cloud_size - 1].y, cloud_in->points[cloud_size - 1].x) + 2 * M_PI;
    if (seg_info_msg_->end_orientation - seg_info_msg_->start_orientation > 3 * M_PI)
    {
      seg_info_msg_->end_orientation -= 2 * M_PI;
    }
    else if (seg_info_msg_->end_orientation - seg_info_msg_->start_orientation < M_PI)
    {
      seg_info_msg_->end_orientation += 2 * M_PI;
    }
    seg_info_msg_->orientation_diff = seg_info_msg_->end_orientation - seg_info_msg_->start_orientation;

    double vertical_ang, horizon_ang;
    int row_id, col_id, index;
    for (int i = 0; i < cloud_size; ++i)
    {
      auto &p = cloud_in->points[i];
      vertical_ang = RAD2ANGLE(atan2(p.z, hypot(p.x, p.y)));
      row_id = (vertical_ang + ang_bottom) / ang_res_y + 0.5;
      if (row_id < 0 || row_id >= N_SCAN)
      {
        RCLCPP_WARN(this->get_logger(), "error row_id");
        continue;
      }

      horizon_ang = RAD2ANGLE(-atan2(p.y, p.x) + 2 * M_PI);
      col_id = horizon_ang / ang_res_x;
      if (col_id >= Horizon_SCAN)
      {
        col_id -= Horizon_SCAN;
      }
      if (col_id < 0 || col_id >= Horizon_SCAN)
      {
        RCLCPP_WARN(this->get_logger(), "error col_id %d ", col_id);
        continue;
      }

      range_mat_(row_id, col_id) = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);

      p.intensity = row_id + col_id / 10000.0;
      index = col_id + row_id * Horizon_SCAN;
      full_cloud_->points[index] = p;
    }

    // groundRemoval
    int lower_id, upper_id;
    double diff_x, diff_y, diff_z, angle;
    for (int j = 0; j < Horizon_SCAN; ++j)
    {
      for (int i = 0; i < ground_scan_id; ++i)
      {
        lower_id = j + i * Horizon_SCAN;
        upper_id = j + (i + 1) * Horizon_SCAN;

        if (-1 == full_cloud_->points[lower_id].intensity || -1 == full_cloud_->points[upper_id].intensity)
        {
          continue;
        }

        diff_x = full_cloud_->points[upper_id].x - full_cloud_->points[lower_id].x;
        diff_y = full_cloud_->points[upper_id].y - full_cloud_->points[lower_id].y;
        diff_z = full_cloud_->points[upper_id].z - full_cloud_->points[lower_id].z;
        angle = RAD2ANGLE(atan2(diff_z, hypot(diff_x, diff_y)));

        if (abs(angle - sensor_mount_ang) < 10.)
        {
          ground_mat_(i, j) = ground_mat_(i + 1, j) = true;
        }
      }
    }

    for (int i = 0; i < N_SCAN; ++i)
    {
      for (int j = 0; j < Horizon_SCAN; ++j)
      {
        if (ground_mat_(i, j) == true || range_mat_(i, j) == std::numeric_limits<double>::max())
        {
          label_mat_(i, j) = -1;
        }
      }
    }

    // cloudSegmentation
    // 对非地面点进行简单聚类，为了找 outlier
    for (int i = 0; i < N_SCAN; ++i)
    {
      for (int j = 0; j < Horizon_SCAN; ++j)
      {
        if (label_mat_(i, j) == 0)
        {
          labelComponents(i, j);
        }
      }
    }

    int line_size = 0;
    for (int i = 0; i < N_SCAN; ++i)
    {
      seg_info_msg_->start_ring_index[i] = line_size + 5;
      for (int j = 0; j < Horizon_SCAN; ++j)
      {
        if (label_mat_(i, j) > 0 || ground_mat_(i, j) == true)
        {
          // TODO: 这里对噪点和地面点的滤波对结果有提升吗？
          if (label_mat_(i, j) == 999999)
          {
            if (i > ground_scan_id && j % 5 == 0)
            {
              outlier_cloud_->points.push_back(full_cloud_->points[j + i * Horizon_SCAN]);
            }
            continue;
          }
          else if (ground_mat_(i, j) == true)
          {
            if (j % 5 != 0 && j > 4 && j < Horizon_SCAN - 5)
            {
              continue;
            }
          }

          seg_info_msg_->segmented_cloud_ground_flag[line_size] = (ground_mat_(i, j) == true);
          seg_info_msg_->segmented_cloud_col_id[line_size] = j;
          seg_info_msg_->segmented_cloud_range[line_size] = range_mat_(i, j);
          segmented_cloud_->points.push_back(full_cloud_->points[j + i * Horizon_SCAN]);
          ++line_size;
        }
      }
      seg_info_msg_->end_ring_index[i] = line_size - 1 - 5;
    }
    publish();

    RCLCPP_INFO(this->get_logger(), "segmented_cloud size: %d", segmented_cloud_->points.size());
    RCLCPP_INFO(this->get_logger(), "outlier_cloud size: %d", outlier_cloud_->points.size());

    segmented_cloud_->clear();
    outlier_cloud_->clear();
    range_mat_.setConstant(std::numeric_limits<double>::max());
    label_mat_.setZero();
    ground_mat_.setZero();
    label_cnt_ = 1;
    PointT nan_p;
    nan_p.intensity = -1;
    std::fill(full_cloud_->points.begin(), full_cloud_->points.end(), nan_p);

    // std::cout << "image projection time: " << t_whole.toc() << " ms" << std::endl;
  }

  void labelComponents(int row, int col)
  {
    double d1, d2, alpha, angle;
    int from_id_i, from_id_j, this_id_i, this_id_j;
    bool line_cnt_flag[N_SCAN] = {false};

    queue<int> que_id_i, que_id_j;
    queue<int> all_pushed_id_i, all_pushed_id_j;
    que_id_i.push(row);
    que_id_j.push(col);
    line_cnt_flag[row] = true; // 原 LeGO_LOAM 源码中没有
    all_pushed_id_i.push(row);
    all_pushed_id_j.push(col);

    while (!que_id_i.empty())
    {
      from_id_i = que_id_i.front();
      from_id_j = que_id_j.front();
      que_id_i.pop();
      que_id_j.pop();
      label_mat_(from_id_i, from_id_j) = label_cnt_;
      line_cnt_flag[from_id_i] = true;

      for (const auto &iter : neighbor_iter_)
      {
        this_id_i = from_id_i + iter.first;
        this_id_j = from_id_j + iter.second;
        if (this_id_i < 0 || this_id_i >= N_SCAN)
        {
          continue;
        }
        if (this_id_j < 0)
        {
          this_id_j = Horizon_SCAN - 1;
        }
        else if (this_id_j >= Horizon_SCAN)
        {
          this_id_j = 0;
        }
        if (label_mat_(this_id_i, this_id_j))
        {
          // 已访问过，或者是地面/Nan 点
          continue;
        }

        d1 = max(range_mat_(from_id_i, from_id_j), range_mat_(this_id_i, this_id_j));
        d2 = min(range_mat_(from_id_i, from_id_j), range_mat_(this_id_i, this_id_j));

        if (iter.first == 0)
        {
          alpha = seg_alpha_x;
        }
        else
        {
          alpha = seg_alpha_y;
        }

        // 在 seg_alpha_y 上的远处点很难聚类，在近处估计还好
        // 在 seg_alpha_x 上聚类可以接受
        angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));
        if (angle > seg_theta)
        {
          que_id_i.push(this_id_i);
          que_id_j.push(this_id_j);
          label_mat_(this_id_i, this_id_j) = label_cnt_;
          line_cnt_flag[this_id_i] = true;
          all_pushed_id_i.push(this_id_i);
          all_pushed_id_j.push(this_id_j);
        }
      }
    }

    bool feasible_seg = false;
    if (all_pushed_id_i.size() >= 30)
    {
      feasible_seg = true;
    }
    else if (all_pushed_id_i.size() >= seg_valid_point_num)
    {
      int line_cnt = 0;
      for (int i = 0; i < N_SCAN; ++i)
      {
        if (line_cnt_flag[i])
        {
          ++line_cnt;
        }
      }
      if (line_cnt >= seg_valid_line_num)
      {
        feasible_seg = true;
      }
    }

    if (feasible_seg)
    {
      ++label_cnt_;
    }
    else
    {
      while (!all_pushed_id_i.empty())
      {
        label_mat_(all_pushed_id_i.front(), all_pushed_id_j.front()) = 999999;
        all_pushed_id_i.pop();
        all_pushed_id_j.pop();
      }
    }
  }

  void publish()
  {
    if (pub_seg_info_->get_publisher_handle() > 0)
    {
      pub_seg_info_->publish(seg_info_msg_);
    }
    if (pub_segmented_cloud_->get_publisher_handle() > 0)
    {
      pcl::toROSMsg(*segmented_cloud_, *segmented_cloud_msg_);
      segmented_cloud_msg_->header = seg_info_msg_->header;
      pub_segmented_cloud_->publish(segmented_cloud_msg_);
    }
    if (pub_outlier_->get_publisher_handle() > 0)
    {
      pcl::toROSMsg(*outlier_cloud_, *outlier_cloud_msg_);
      outlier_cloud_msg_->header = seg_info_msg_->header;
      pub_outlier_->publish(outlier_cloud_msg_);
    }
  }

}; // namespace loam

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IP>());
  rclcpp::shutdown();
  return 0;
}