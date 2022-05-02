#
# Created  on 2020/2/23
#
import os
import time

import numpy as np
import cv2 as cv

from hpatch_related.hpatch_dataset import HPatchDataset

# import evaluation tools
from utils.evaluation_tools import RepeatabilityCalculator
from utils.evaluation_tools import HomoAccuracyCalculator
from utils.evaluation_tools import MeanMatchingAccuracy
from utils.evaluation_tools import PointStatistics
from utils.utils import Matcher


class Evaluation(object):
    """
    专用于hpatch相关的测试
    """

    def __init__(self, model, logger, height=480, width=640, dataset_dir=None, correct_epsilon=3):
        self.logger = logger
        self.logger.info("height=%d, width=%d, correct_epsilon=%d" % (height, width, correct_epsilon))

        if dataset_dir is None:
            dataset_dir = "/data/MegPoint/dataset/hpatch"

        # model initialized by input
        self.logger.info("Testing model: %s" % model.name)
        self.model = model

        if model.name in ["superpoint", "megpoint", "orb", "sift"]:
            grayscale = True
        else:
            grayscale = False

        # initialize HPatch dataset
        self.test_dataset = HPatchDataset(dataset_dir=dataset_dir, grayscale=grayscale, resize=False)

        # initialize test calculator
        self.illum_repeat = RepeatabilityCalculator(correct_epsilon)
        self.view_repeat = RepeatabilityCalculator(correct_epsilon)

        self.illum_homo_acc = HomoAccuracyCalculator(correct_epsilon)
        self.view_homo_acc = HomoAccuracyCalculator(correct_epsilon)

        self.illum_mma = MeanMatchingAccuracy(correct_epsilon)
        self.view_mma = MeanMatchingAccuracy(correct_epsilon)
        self.correct_epsilon = correct_epsilon

        self.point_statistics = PointStatistics()

        # initialize matcher
        if model.name == "orb":
            self.matcher = cv.BFMatcher_create(cv.NORM_HAMMING, crossCheck=True)
        else:
            self.matcher = Matcher(dtype="float")

    def test(self):

        # 重置测评算子参数
        self.illum_repeat.reset()
        self.view_repeat.reset()
        self.illum_homo_acc.reset()
        self.illum_mma.reset()
        self.view_homo_acc.reset()
        self.view_mma.reset()
        self.point_statistics.reset()

        start_time = time.time()
        count = 0
        skip = 0
        bad = 0

        for i, data in enumerate(self.test_dataset):
            first_image = data['first_image']
            second_image = data['second_image']
            gt_homography = data['gt_homography']
            image_type = data['image_type']
            first_shape = data['first_shape']
            second_shape = data['second_shape']

            results = self.model(first_image, second_image)

            if results is None:
                skip += 1
                continue

            first_point = results[0]
            first_point_num = results[1]
            select_first_desp = results[2]
            second_point = results[3]
            second_point_num = results[4]
            select_second_desp = results[5]

            # 得到匹配点
            if self.model.name == "orb":
                matched = self.matcher.match(select_first_desp, select_second_desp)
                first_point = np.float32([first_point[m.queryIdx].pt for m in matched]).reshape(-1, 2)
                second_point = np.float32([second_point[m.trainIdx].pt for m in matched]).reshape(-1, 2)
                first_point = first_point[:, ::-1]
                second_point = second_point[:, ::-1]
                matched_point = (first_point, second_point)
            else:
                matched_point = self.matcher(first_point, select_first_desp, second_point, select_second_desp)

            if matched_point is None:
                print("skip this pair because there's no match point!")
                skip += 1
                continue

            # 计算得到单应变换
            pred_homography, _ = cv.findHomography(matched_point[0][:, np.newaxis, ::-1],
                                                   matched_point[1][:, np.newaxis, ::-1], cv.RANSAC)

            if pred_homography is None:
                print("skip this pair because no homo can be predicted!.")
                skip += 1
                continue

            # 对单样本进行测评
            if image_type == 'illumination':
                self.illum_repeat.update(first_point, second_point, gt_homography, shape_0=first_shape, shape_1=second_shape)
                self.illum_homo_acc.update(pred_homography, gt_homography, shape_0=first_shape)
                self.illum_mma.update(gt_homography, matched_point)

                # if not correct:
                #     self.illum_bad_mma.update(gt_homography, matched_point)
                #     bad += 1

            elif image_type == 'viewpoint':
                self.view_repeat.update(first_point, second_point, gt_homography, shape_0=first_shape, shape_1=second_shape)
                self.view_homo_acc.update(pred_homography, gt_homography, shape_0=first_shape)
                self.view_mma.update(gt_homography, matched_point)

                # if not correct:
                #     self.view_bad_mma.update(gt_homography, matched_point)
                #     bad += 1

            else:
                print("The image type magicpoint_tester.test(ckpt_file)must be one of illumination of viewpoint ! "
                      "Please check !")
                assert False

            # 统计检测的点的数目
            self.point_statistics.update((first_point_num+second_point_num)/2.)

            if i % 10 == 0:
                print("Having tested %d samples, which takes %.3fs" % (i, (time.time() - start_time)))
                start_time = time.time()
            count += 1

        # 计算各自的重复率以及总的重复率
        illum_repeat, view_repeat, total_repeat = self._compute_total_metric(self.illum_repeat,
                                                                             self.view_repeat)

        # 计算估计的单应变换准确度
        illum_homo_acc, view_homo_acc, total_homo_acc = self._compute_total_metric(self.illum_homo_acc,
                                                                                   self.view_homo_acc)

        # 计算匹配的准确度
        illum_match_acc, view_match_acc, total_match_acc = self._compute_total_metric(self.illum_mma,
                                                                                      self.view_mma)

        # 计算匹配外点的分布情况
        illum_dis, view_dis = self._compute_match_outlier_distribution(self.illum_mma,
                                                                       self.view_mma)

        # illum_bad_dis, view_bad_dis = self._compute_match_outlier_distribution(self.illum_bad_mma,
        #                                                                        self.view_bad_mma)

        # 统计最终的检测点数目的平均值和方差
        point_avg, point_std = self.point_statistics.average()

        self.logger.info("Having skiped %d test pairs" % skip)

        self.logger.info("Homography Accuracy: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_homo_acc, view_homo_acc, total_homo_acc))
        self.logger.info("Mean Matching Accuracy: illumination: %.4f, viewpoint: %.4f, total: %.4f " %
                         (illum_match_acc, view_match_acc, total_match_acc))
        self.logger.info("Repeatability: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_repeat, view_repeat, total_repeat))
        self.logger.info("Detection point, average: %.4f, variance: %.4f" % (point_avg, point_std))

        self.logger.info("Illumination Matching Distribution, e=3:"
                         " [0, e/2]: %.4f, (e/2,e]: %.4f, (e,2e]: %.4f, (2e,4e]: %.4f, (4e,+): %.4f" %
                         (illum_dis[0], illum_dis[1], illum_dis[2],
                          illum_dis[3], illum_dis[4]))
        self.logger.info("Viewpoint Matching Distribution, e=3:"
                         " [0, e/2]: %.4f, (e/2,e]: %.4f, (e,2e]: %.4f, (2e,4e]: %.4f, (4e,+): %.4f" %
                         (view_dis[0], view_dis[1], view_dis[2],
                          view_dis[3], view_dis[4]))

    def test_matches(self, save_dir, save_img):
        save_dir = os.path.join(save_dir, self.model.name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.logger.info("save_dir: %s" % save_dir)

        # 重置测评算子参数
        self.illum_repeat.reset()
        self.illum_homo_acc.reset()
        self.illum_mma.reset()
        self.view_repeat.reset()
        self.view_homo_acc.reset()
        self.view_mma.reset()
        self.point_statistics.reset()

        start_time = time.time()
        count = 0
        illum_skip = 0
        view_skip = 0
        bad = 0
        skip = 0

        for i, data in enumerate(self.test_dataset):
            first_image = data['first_image']
            second_image = data['second_image']
            gt_homography = data['gt_homography']
            image_type = data['image_type']

            results = self.model(first_image, second_image)

            if results is None:
                skip += 1
                continue

            first_point = results[0]
            first_point_num = results[1]
            select_first_desp = results[2]
            second_point = results[3]
            second_point_num = results[4]
            select_second_desp = results[5]

            # 得到匹配点
            if self.model.name == "orb":
                matched = self.matcher.match(select_first_desp, select_second_desp)
                first_point = np.float32([first_point[m.queryIdx].pt for m in matched]).reshape(-1, 2)
                second_point = np.float32([second_point[m.trainIdx].pt for m in matched]).reshape(-1, 2)
                first_point = first_point[:, ::-1]
                second_point = second_point[:, ::-1]
                matched_point = (first_point, second_point)
            else:
                matched_point = self.matcher(first_point, select_first_desp, second_point, select_second_desp)

            # debug
            correct_first_list = []
            correct_second_list = []
            wrong_first_list = []
            wrong_second_list = []
            xy_first_point = matched_point[0][:, ::-1]
            xy1_first_point = np.concatenate((xy_first_point, np.ones((xy_first_point.shape[0], 1))), axis=1)
            xyz_second_point = np.matmul(gt_homography, xy1_first_point[:, :, np.newaxis])[:, :, 0]
            xy_second_point = xyz_second_point[:, :2] / xyz_second_point[:, 2:3]

            matched_second_point = []
            # 重新计算经误差缩小后的投影误差
            diff = np.linalg.norm(xy_second_point - matched_point[1][:, ::-1], axis=1)
            for j in range(xy_first_point.shape[0]):
                # 重投影误差小于3的判断为正确匹配
                if diff[j] <= self.correct_epsilon:
                    correct_first_list.append(matched_point[0][j])
                    correct_second_list.append(matched_point[1][j])
                    matched_second_point.append(matched_point[1][j])
                else:
                    wrong_first_list.append(matched_point[0][j])
                    wrong_second_list.append(matched_point[1][j])
                    matched_second_point.append(matched_point[1][j])

            matched_second_point = np.stack(matched_second_point, axis=0)
            matched_point = (matched_point[0], matched_second_point)

            cv_correct_first, cv_correct_second, cv_correct_matched = self._convert_match2cv(
                correct_first_list,
                correct_second_list)
            cv_wrong_first, cv_wrong_second, cv_wrong_matched = self._convert_match2cv(
                wrong_first_list,
                wrong_second_list,
                0.25)

            if matched_point is None:
                print("skip this pair because there's no match point!")
                if image_type == "illumination":
                    illum_skip += 1
                else:
                    view_skip += 1
                continue

            pred_homography, _ = cv.findHomography(matched_point[0][:, np.newaxis, ::-1],
                                                   matched_point[1][:, np.newaxis, ::-1], cv.RANSAC, maxIters=3000)

            if pred_homography is None:
                print("skip this pair because no homo can be predicted!.")
                if image_type == "illumination":
                    illum_skip += 1
                else:
                    view_skip += 1
                continue

            # 对单样本进行测评
            if image_type == 'illumination':
                self.illum_repeat.update(
                    first_point, second_point, gt_homography)

                self.illum_homo_acc.update(pred_homography, gt_homography, True)
                self.illum_mma.update(gt_homography, matched_point)

                # draw correct match
                # matched_image = cv.drawMatches(
                #     first_image, cv_correct_first, second_image, cv_correct_second,
                #     cv_correct_matched, None, matchColor=(0, 255, 0))
                matched_image = self.draw_matches(first_image, correct_first_list, second_image, correct_second_list,
                                                  color=(0, 255, 0))
                first_image, second_image = np.split(matched_image, 2, axis=1)

                # draw wrong match
                # matched_image = cv.drawMatches(
                #     first_image, cv_wrong_first, second_image, cv_wrong_second,
                #     cv_wrong_matched, None, matchColor=(0, 0, 255))
                matched_image = self.draw_matches(first_image, wrong_first_list, second_image, wrong_second_list,
                                                  color=(0, 0, 255))

                # metric_str = "correct match: %d/ total: %d, %.4f" % (
                #     len(correct_first_list), matched_point[0].shape[0],
                #     len(correct_first_list) / matched_point[0].shape[0]
                # )
                # cv.putText(matched_image, metric_str, (0, 40), cv.FONT_HERSHEY_COMPLEX, fontScale=0.8,
                #            color=(200, 0, 0), thickness=2)
                if save_img:
                    cv.imwrite(os.path.join(save_dir, "image_%03d.jpg" % i), matched_image)

            elif image_type == 'viewpoint':
                self.view_repeat.update(first_point, second_point, gt_homography)

                self.view_homo_acc.update(pred_homography, gt_homography, True)
                self.view_mma.update(gt_homography, matched_point)

                # draw correct match
                # matched_image = cv.drawMatches(
                #     first_image, cv_correct_first, second_image, cv_correct_second,
                #     cv_correct_matched, None, matchColor=(0, 255, 0))
                matched_image = self.draw_matches(first_image, correct_first_list, second_image, correct_second_list,
                                                  color=(0, 255, 0))
                first_image, second_image = np.split(matched_image, 2, axis=1)

                # draw wrong match
                # matched_image = cv.drawMatches(
                #     first_image, cv_wrong_first, second_image, cv_wrong_second,
                #     cv_wrong_matched, None, matchColor=(0, 0, 255))
                matched_image = self.draw_matches(first_image, wrong_first_list, second_image, wrong_second_list,
                                                  color=(0, 0, 255))

                # metric_str = "correct match: %d/ total: %d, %.4f" % (
                #     len(correct_first_list), matched_point[0].shape[0],
                #     len(correct_first_list) / matched_point[0].shape[0]
                # )
                # cv.putText(matched_image, metric_str, (0, 40), cv.FONT_HERSHEY_COMPLEX, fontScale=0.8,
                #            color=(200, 0, 0), thickness=2)
                if save_img:
                    cv.imwrite(os.path.join(save_dir, "image_%03d.jpg" % i), matched_image)

            else:
                print("The image type magicpoint_tester.test(ckpt_file)must be one of illumination of viewpoint ! "
                      "Please check !")
                assert False

            # 统计检测的点的数目
            self.point_statistics.update((first_point_num + second_point_num) / 2.)

            if i % 10 == 0:
                print("Having tested %d samples, which takes %.3fs" % (i, (time.time() - start_time)))
                start_time = time.time()
            count += 1
            # if count % 1000 == 0:
            #     break

        self.logger.info("Totally skip {} illumination samples and {} view samples.".format(illum_skip, view_skip))

        # 计算各自的重复率以及总的重复率
        illum_repeat, view_repeat, total_repeat = self._compute_total_metric(self.illum_repeat,
                                                                             self.view_repeat)
        # 计算估计的单应变换准确度
        illum_homo_acc, view_homo_acc, total_homo_acc = self._compute_total_metric(self.illum_homo_acc,
                                                                                   self.view_homo_acc)
        # 计算匹配的准确度
        illum_match_acc, view_match_acc, total_match_acc = self._compute_total_metric(self.illum_mma,
                                                                                      self.view_mma)

        # 计算匹配外点的分布情况
        illum_dis, view_dis = self._compute_match_outlier_distribution(self.illum_mma,
                                                                       self.view_mma)

        # 统计最终的检测点数目的平均值和方差
        point_avg, point_std = self.point_statistics.average()

        self.logger.info("Homography Accuracy: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_homo_acc, view_homo_acc, total_homo_acc))
        self.logger.info("Mean Matching Accuracy: illumination: %.4f, viewpoint: %.4f, total: %.4f " %
                         (illum_match_acc, view_match_acc, total_match_acc))
        self.logger.info("Repeatability: illumination: %.4f, viewpoint: %.4f, total: %.4f" %
                         (illum_repeat, view_repeat, total_repeat))
        self.logger.info("Detection point, average: %.4f, variance: %.4f" % (point_avg, point_std))

        self.logger.info("Illumination Matching Distribution:"
                         " [0, e/2]: %.4f, (e/2,e]: %.4f, (e,2e]: %.4f, (2e,4e]: %.4f, (4e,+): %.4f" %
                         (illum_dis[0], illum_dis[1], illum_dis[2],
                          illum_dis[3], illum_dis[4]))
        self.logger.info("Viewpoint Matching Distribution:"
                         " [0, e/2]: %.4f, (e/2,e]: %.4f, (e,2e]: %.4f, (2e,4e]: %.4f, (4e,+): %.4f" %
                         (view_dis[0], view_dis[1], view_dis[2],
                          view_dis[3], view_dis[4]))

    def test_time(self):

        all_image = []
        for i, data in enumerate(self.test_dataset):
            first_image = data['first_image']
            second_image = data['second_image']

            all_image.append(cv.resize(first_image, None, None, fx=1, fy=1))
            all_image.append(cv.resize(second_image, None, None, fx=1, fy=1))
            if i % 100 == 0:
                print("loading %d images" % i)

        start_time = time.time()
        for img in all_image:
            self.model.generate_feature(img)

        time_spend = time.time() - start_time
        avg_time = time_spend / len(all_image)
        fps = len(all_image) / time_spend
        self.logger.info("The average time of %d images is: %.4fs, FPS: %.4f" % (len(all_image), avg_time, fps))

    @staticmethod
    def _compute_total_metric(illum_metric, view_metric):
        illum_acc, illum_sum, illum_num = illum_metric.average()
        view_acc, view_sum, view_num = view_metric.average()
        return illum_acc, view_acc, (illum_sum+view_sum)/(illum_num+view_num+1e-4)

    @staticmethod
    def _compute_match_outlier_distribution(illum_metric, view_metric):
        illum_distribution = illum_metric.average_outlier()
        view_distribution = view_metric.average_outlier()
        return illum_distribution, view_distribution

    @staticmethod
    def _convert_match2cv(first_point_list, second_point_list, sample_ratio=1.0):
        cv_first_point = []
        cv_second_point = []
        cv_matched_list = []

        assert len(first_point_list) == len(second_point_list)

        inc = 1
        if sample_ratio < 1:
            inc = int(1.0 / sample_ratio)

        count = 0
        if len(first_point_list) > 0:
            for j in range(0, len(first_point_list), inc):
                cv_point = cv.KeyPoint()
                cv_point.pt = tuple(first_point_list[j][::-1])
                cv_first_point.append(cv_point)

                cv_point = cv.KeyPoint()
                cv_point.pt = tuple(second_point_list[j][::-1])
                cv_second_point.append(cv_point)

                cv_match = cv.DMatch()
                cv_match.queryIdx = count
                cv_match.trainIdx = count
                cv_matched_list.append(cv_match)

                count += 1

        return cv_first_point, cv_second_point, cv_matched_list

    @staticmethod
    def draw_matches(first_image, first_list, second_image, second_list, color, height=480, width=640):
        if len(first_image.shape) == 2:
            first_image = np.tile(first_image[:, :, np.newaxis], [1, 1, 3])
            second_image = np.tile(second_image[:, :, np.newaxis], [1, 1, 3])
        first_array_list = []
        second_array_list = []
        dpt = np.array((width, 0), dtype=np.int)
        for i in range(len(first_list)):
            first_pt = np.round(first_list[i][::-1]).astype(np.int)
            second_pt = np.round(second_list[i][::-1] + dpt).astype(np.int)
            first_array_list.append((first_pt[0], first_pt[1]))
            second_array_list.append((second_pt[0], second_pt[1]))

        cat_image = np.concatenate((first_image, second_image), axis=1)
        for i in range(len(first_list)):
            cat_image = cv.line(cat_image, pt1=first_array_list[i], pt2=(second_array_list[i]), color=color)
        return cat_image

    @staticmethod
    def _convert_pt2cv(point_list):
        cv_point_list = []

        for i in range(len(point_list)):
            cv_point = cv.KeyPoint()
            cv_point.pt = tuple(point_list[i][::-1])
            cv_point_list.append(cv_point)

        return cv_point_list

    @staticmethod
    def _convert_pt2cv_np(point):
        cv_point_list = []
        for i in range(point.shape[0]):
            cv_point = cv.KeyPoint()
            cv_point.pt = tuple(point[i, ::-1])
            cv_point_list.append(cv_point)

        return cv_point_list

    @staticmethod
    def _convert_cv2pt(cv_point):
        point_list = []
        for i, cv_pt in enumerate(cv_point):
            pt = np.array((cv_pt.pt[1], cv_pt.pt[0]))  # y,x的顺序
            point_list.append(pt)
        point = np.stack(point_list, axis=0)
        return point


