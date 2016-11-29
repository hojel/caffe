#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layers/detection_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
  Dtype left = std::max(x1 - w1 / 2, x2 - w2 / 2);
  Dtype right = std::min(x1 + w1 / 2, x2 + w2 / 2);
  return right - left;
}

template <typename Dtype>
Dtype Calc_iou(const vector<Dtype>& box1, const vector<Dtype>& box2) {
  Dtype w = Overlap(box1[0], box1[2], box2[0], box2[2]);
  Dtype h = Overlap(box1[1], box1[3], box2[1], box2[3]);
  if (w < 0 || h < 0) return 0;
  Dtype inter_area = w * h;
  Dtype union_area = box1[2] * box1[3] + box2[2] * box2[3] - inter_area;
  return inter_area / union_area;
}

template <typename Dtype>
Dtype Calc_rmse(const vector<Dtype>& box1, const vector<Dtype>& box2) {
  return sqrt(pow(box1[0]-box2[0], 2) +
              pow(box1[1]-box2[1], 2) +
              pow(box1[2]-box2[2], 2) +
              pow(box1[3]-box2[3], 2));
}

/*
    m = side * side
    data
          classes[m,20]
          obj_scale[m,2]
          boxes(m,2,4]
    label
          difficult[m]
          isobj[m]
          classes[m]
          boxes[m,4]
 */

template <typename Dtype>
void DetectionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  DetectionLossParameter param = this->layer_param_.detection_loss_param();
  side_ = param.side();
  num_class_ = param.num_class();
  num_object_ = param.num_object();
  num_coord_ = param.num_coord();
  sqrt_ = param.sqrt();
  rescore_ = param.rescore();
  object_scale_ = param.object_scale();
  noobject_scale_ = param.noobject_scale();
  class_scale_ = param.class_scale();
  coord_scale_ = param.coord_scale();

  int input_count = bottom[0]->count(1);
  int label_count = bottom[1]->count(1);
  int tmp_input_count = side_ * side_ * (num_class_ + (1 + num_coord_) * num_object_);
  int tmp_label_count = side_ * side_ * (1 + 1 + 1 + num_coord_);
  CHECK_EQ(input_count, tmp_input_count);
  CHECK_EQ(label_count, tmp_label_count);
}

template <typename Dtype>
void DetectionLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void DetectionLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* diff = diff_.mutable_cpu_data();
  Dtype loss(0.0), class_loss(0.0), noobj_loss(0.0), obj_loss(0.0), coord_loss(0.0), area_loss(0.0), iou_loss(0.0);
  Dtype avg_iou(0.0), avg_obj(0.0), avg_cls(0.0), avg_pos_cls(0.0), avg_no_obj(0.0);
  Dtype obj_count(0);
  int locations = pow(side_, 2);
  caffe_set(diff_.count(), Dtype(0.), diff);
  for (int i = 0; i < bottom[0]->num(); ++i) {	// batch
    int index = i * bottom[0]->count(1);
    int true_index = i * bottom[1]->count(1);
    for (int j = 0; j < locations; ++j) {
      for (int k = 0; k < num_object_; ++k) {
        int p_index = index + locations*num_class_ + j*num_object_ + k;
        noobj_loss += noobject_scale_ * pow(input_data[p_index], 2);
        diff[p_index] = noobject_scale_ * (input_data[p_index] - 0);
        avg_no_obj += input_data[p_index];
      }
      bool isobj = label_data[true_index + locations + j];
      if (!isobj) {
        continue;
      }
      obj_count += 1;
      int label = static_cast<int>(label_data[true_index + locations * 2 + j]);
      CHECK_GE(label, 0) << "label start at 0";
      CHECK_LT(label, num_class_) << "label must below num_class";
      for (int c = 0; c < num_class_; ++c) {
        int class_index = index + j * num_class_ + c;
        Dtype target = Dtype(c == label ? 1 : 0);	// truth mask
        avg_cls += input_data[class_index];
        if (c == label)
          avg_pos_cls += input_data[class_index];
        class_loss += class_scale_ * pow(input_data[class_index] - target, 2);
        diff[class_index] = class_scale_ * (input_data[class_index] - target);
      }
      const Dtype* true_box_pt = label_data + true_index + locations*3 + j*num_coord_;
      vector<Dtype> true_box(true_box_pt, true_box_pt + num_coord_);
      true_box[0] /= side_;
      true_box[1] /= side_;

      Dtype best_iou = 0.;
      Dtype best_rmse = 20.;
      int best_index = -1;
      for (int k = 0; k < num_object_; ++k) {
        const Dtype* box_pt = input_data + index + locations*(num_class_ + num_object_) + (j*num_object_ + k)*num_coord_;
        vector<Dtype> box(box_pt, box_pt + num_coord_);
        box[0] /= side_;
        box[1] /= side_;
        if (sqrt_) {
          box[2] = pow(box[2], 2);
          box[3] = pow(box[3], 2);
        }
        Dtype iou = Calc_iou(box, true_box);
        Dtype rmse = Calc_rmse(box, true_box);
        if (best_iou > 0 || iou > 0) {
          if (iou > best_iou) {
            best_iou = iou;
            best_index = k;
          }
        } else {
          if (rmse < best_rmse) {
            best_rmse = rmse;
            best_index = k;
          }
        }
      }

      //CHECK_GE(best_index, 0) << "best_index must >= 0";
      if (best_index < 0) {
        DLOG(WARNING) << "best_index must >= 0";
        best_index = 0;
      }

      int p_index = index + locations*num_class_ + j*num_object_ + best_index;
      int box_index = index + locations*(num_class_ + num_object_) + (j*num_object_ + best_index)*num_coord_;
      noobj_loss -= noobject_scale_ * pow(input_data[p_index], 2);
      obj_loss += object_scale_ * pow(input_data[p_index] - 1., 2);
      avg_no_obj -= input_data[p_index];
      avg_obj += input_data[p_index];

      const Dtype* best_box_pt = input_data + box_index;
      vector<Dtype> box(best_box_pt, best_box_pt+num_coord_);
      box[0] /= side_;
      box[1] /= side_;
      if (sqrt_) {
        box[2] = pow(box[2], 2);
        box[3] = pow(box[3], 2);
      }
      Dtype iou = Calc_iou(box, true_box);
      iou_loss += pow(iou - 1., 2);
      avg_iou += iou;
      if (rescore_) {
        diff[p_index] = object_scale_ * (input_data[p_index] - iou);
      } else {
        diff[p_index] = object_scale_ * (input_data[p_index] - 1.);
      }

      vector<Dtype> true_box2(true_box_pt, true_box_pt + num_coord_);
      if (sqrt_) {
        true_box2[2] = sqrt(true_box2[2]);
        true_box2[3] = sqrt(true_box2[3]);
      }
      vector<Dtype> best_box(best_box_pt, best_box_pt+num_coord_);

      for (int o = 0; o < num_coord_; ++o) {
        diff[box_index + o] = coord_scale_ * (best_box[o] - true_box2[o]);
      }

      coord_loss += coord_scale_ * pow(best_box[0] - true_box2[0], 2);
      coord_loss += coord_scale_ * pow(best_box[1] - true_box2[1], 2);
      area_loss += coord_scale_ * pow(best_box[2] - true_box2[2], 2);
      area_loss += coord_scale_ * pow(best_box[3] - true_box2[3], 2);
    }
  }

  class_loss /= obj_count;
  coord_loss /= obj_count;
  area_loss /= obj_count;
  iou_loss /= obj_count;
  obj_loss /= obj_count;
  noobj_loss /= (locations * num_object_ * bottom[0]->num() - obj_count);

  avg_iou /= obj_count;
  avg_obj /= obj_count;
  avg_no_obj /= (locations * num_object_ * bottom[0]->num() - obj_count);
  avg_cls /= obj_count;
  avg_pos_cls /= obj_count;

#if 0
  loss = class_loss + coord_loss + area_loss + iou_loss + obj_loss + noobj_loss;
#else
  loss = caffe_cpu_dot(diff_.count(), diff, diff);
  loss /= bottom[0]->num();
#endif
  top[0]->mutable_cpu_data()[0] = loss;

  // obj_count /= bottom[0]->num();
  // DLOG(INFO) << "average objects: " << obj_count;
  DLOG(INFO) << "loss: " << loss << " class_loss: " << class_loss << " obj_loss: "
        << obj_loss << " noobj_loss: " << noobj_loss << " coord_loss: " << coord_loss
        << " area_loss: " << area_loss;
  LOG(INFO) << "avg_iou: " << avg_iou << " avg_obj: " << avg_obj << " avg_no_obj: "
        << avg_no_obj << " avg_cls: " << avg_cls << " avg_pos_cls: " << avg_pos_cls;
}

template <typename Dtype>
void DetectionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype sign(1);
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();
    caffe_cpu_axpby(
        bottom[0]->count(),
        alpha,
        diff_.cpu_data(),
        Dtype(0),	// beta
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
//STUB_GPU(DetectionLossLayer);
#endif

INSTANTIATE_CLASS(DetectionLossLayer);
REGISTER_LAYER_CLASS(DetectionLoss);

}  // namespace caffe
