import os
import time

import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader

from nets import get_model
from data_utils import get_dataset
from trainers.base_trainer import BaseTrainer
from utils.utils import spatial_nms
from utils.utils import DescriptorGeneralTripletLoss
from utils.utils import PointHeatmapWeightedBCELoss


class MegPointTrainer(BaseTrainer):

    def __init__(self, **config):
        super(MegPointTrainer, self).__init__(**config)
        self.point_weight = 1

    def _initialize_dataset(self):
        # 初始化数据集
        self.logger.info('Initialize {}'.format(self.config['train']['dataset']))
        self.train_dataset = get_dataset(self.config['train']['dataset'])(**self.config['train'])

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
            num_workers=self.config['train']['num_workers'],
            drop_last=True
        )
        self.epoch_length = len(self.train_dataset) // self.config['train']['batch_size']

    def _initialize_model(self):
        self.logger.info("Initialize network arch {}".format(self.config['model']['backbone']))
        model = get_model(self.config['model']['backbone'])()

        self.logger.info("Initialize network arch {}".format(self.config['model']['extractor']))
        extractor = get_model(self.config['model']['extractor'])()

        if self.multi_gpus:
            model = torch.nn.DataParallel(model)
            extractor = torch.nn.DataParallel(extractor)
        self.model = model.to(self.device)
        self.extractor = extractor.to(self.device)

    def _initialize_loss(self):
        # 初始化loss算子
        # 初始化heatmap loss
        self.logger.info("Initialize the PointHeatmapWeightedBCELoss.")
        self.point_loss = PointHeatmapWeightedBCELoss()

        # 初始化描述子loss
        self.logger.info("Initialize the DescriptorGeneralTripletLoss.")
        self.descriptor_loss = DescriptorGeneralTripletLoss(self.device)

    def _initialize_optimizer(self):
        # 初始化网络训练优化器
        self.logger.info("Initialize Adam optimizer with weight_decay: {:.5f}.".format(self.config['train']['weight_decay']))
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.config['train']['lr'],
            weight_decay=self.config['train']['weight_decay'])
        self.extractor_optimizer = torch.optim.Adam(
            params=self.extractor.parameters(),
            lr=self.config['train']['lr'],
            weight_decay=self.config['train']['weight_decay'])

    def _initialize_scheduler(self):

        # 初始化学习率调整算子
        if self.config['train']['lr_mod'] == 'LambdaLR':
            self.logger.info("Initialize lr_scheduler of LambdaLR: (%d, %d)" % (
            self.config['train']['maintain_epoch'], self.config['train']['decay_epoch']))

            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - self.config['train']['maintain_epoch']) / float(
                    self.config['train']['decay_epoch'] + 1)
                return lr_l

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        else:
            milestones = [20, 30]
            self.logger.info("Initialize lr_scheduler of MultiStepLR: (%d, %d)" % (milestones[0], milestones[1]))
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

    def _train_one_epoch(self, epoch_idx):
        self.model.train()

        self.logger.info("-----------------------------------------------------")
        self.logger.info("Training epoch %2d begin:" % epoch_idx)

        self._train_func(epoch_idx)

        self.logger.info("Training epoch %2d done." % epoch_idx)
        self.logger.info("-----------------------------------------------------")

    def _train_func(self, epoch_idx):
        self.model.train()
        self.extractor.train()
        stime = time.time()
        for i, data in enumerate(self.train_dataloader):

            # 读取相关数据
            image = data["image"].to(self.device)
            heatmap_gt = data['heatmap'].to(self.device)
            point_mask = data['point_mask'].to(self.device)
            desp_point = data["desp_point"].to(self.device)

            warped_image = data["warped_image"].to(self.device)
            warped_heatmap_gt = data['warped_heatmap'].to(self.device)
            warped_point_mask = data['warped_point_mask'].to(self.device)
            warped_desp_point = data["warped_desp_point"].to(self.device)

            valid_mask = data["valid_mask"].to(self.device)
            not_search_mask = data["not_search_mask"].to(self.device)

            image_pair = torch.cat((image, warped_image), dim=0)

            # 模型预测
            heatmap_pred_pair, c1_pair, c2_pair, c3_pair, c4_pair = self.model(image_pair)

            # 计算描述子loss
            desp_point_pair = torch.cat((desp_point, warped_desp_point), dim=0)
            c1_feature_pair = f.grid_sample(c1_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c2_feature_pair = f.grid_sample(c2_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c3_feature_pair = f.grid_sample(c3_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            c4_feature_pair = f.grid_sample(c4_pair, desp_point_pair, mode="bilinear", padding_mode="border")

            feature_pair = self.cat(c1_feature_pair, c2_feature_pair, c3_feature_pair, c4_feature_pair, dim=1)
            feature_pair = feature_pair[:, :, :, 0].transpose(1, 2)
            desp_pair = self.extractor(feature_pair)
            desp_0, desp_1 = torch.chunk(desp_pair, 2, dim=0)

            desp_loss = self.descriptor_loss(desp_0, desp_1, valid_mask, not_search_mask)

            # 计算关键点loss
            heatmap_gt_pair = torch.cat((heatmap_gt, warped_heatmap_gt), dim=0)
            point_mask_pair = torch.cat((point_mask, warped_point_mask), dim=0)
            point_loss = self.point_loss(heatmap_pred_pair[:, 0, :, :], heatmap_gt_pair, point_mask_pair)

            loss = desp_loss + point_loss

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()
            self.extractor_optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
            self.extractor_optimizer.step()

            # debug use
            # if i == 200:
            #     break

            if i % self.config['train']['log_freq'] == 0:

                point_loss_val = point_loss.item()
                desp_loss_val = desp_loss.item()
                loss_val = loss.item()

                self.logger.info(
                    "[Epoch:%2d][Step:%5d:%5d]: loss = %.4f, point_loss = %.4f, desp_loss = %.4f"
                    " one step cost %.4fs. " % (
                        epoch_idx, i, self.epoch_length,
                        loss_val,
                        point_loss_val,
                        desp_loss_val,
                        (time.time() - stime) / self.config['train']['log_freq'],
                    ))
                stime = time.time()

        # save the model
        if self.multi_gpus:
            torch.save(
                self.model.module.state_dict(), os.path.join(self.config['ckpt_path'], 'model_%02d.pt' % epoch_idx))
            torch.save(
                self.extractor.module.state_dict(), os.path.join(self.config['ckpt_path'], 'extractor_%02d.pt' % epoch_idx))
        else:
            torch.save(
                self.model.state_dict(), os.path.join(self.config['ckpt_path'], 'model_%02d.pt' % epoch_idx))
            torch.save(
                self.extractor.state_dict(), os.path.join(self.config['ckpt_path'], 'extractor_%02d.pt' % epoch_idx))

    def _inference_func(self, image_pair):
        """
        image_pair: [2,1,h,w]
        """
        self.model.eval()
        self.extractor.eval()
        _, _, height, width = image_pair.shape
        heatmap_pair, c1_pair, c2_pair, c3_pair, c4_pair = self.model(image_pair)

        c1_0, c1_1 = torch.chunk(c1_pair, 2, dim=0)
        c2_0, c2_1 = torch.chunk(c2_pair, 2, dim=0)
        c3_0, c3_1 = torch.chunk(c3_pair, 2, dim=0)
        c4_0, c4_1 = torch.chunk(c4_pair, 2, dim=0)

        heatmap_pair = torch.sigmoid(heatmap_pair)
        prob_pair = spatial_nms(heatmap_pair)

        prob_pair = prob_pair.detach().cpu().numpy()
        first_prob = prob_pair[0, 0]
        second_prob = prob_pair[1, 0]

        # 得到对应的预测点
        first_point, first_point_num = self._generate_predict_point(
            first_prob,
            detection_threshold=self.config['test']['detection_threshold'],
            top_k=self.config['test']['top_k'])  # [n,2]

        second_point, second_point_num = self._generate_predict_point(
            second_prob,
            detection_threshold=self.config['test']['detection_threshold'],
            top_k=self.config['test']['top_k'])  # [n,2]

        if first_point_num <= 4 or second_point_num <= 4:
            print("skip this pair because there's little point!")
            return None

        # 得到点对应的描述子
        select_first_desp = self._generate_combined_descriptor_fast(first_point, c1_0, c2_0, c3_0, c4_0, height, width)
        select_second_desp = self._generate_combined_descriptor_fast(second_point, c1_1, c2_1, c3_1, c4_1, height, width)

        return first_point, first_point_num, second_point, second_point_num, select_first_desp, select_second_desp

    def _generate_combined_descriptor_fast(self, point, c1, c2, c3, c4, height, width):
        """
        用多层级的组合特征构造描述子
        Args:
            point: [n,2] 顺序是y,x
            c1,c2,c3,c4: 分别对应resnet4个block输出的特征,batchsize都是1
        Returns:
            desp: [n,dim]
        """
        point = torch.from_numpy(point[:, ::-1].copy()).to(torch.float).to(self.device)
        # 归一化采样坐标到[-1,1]
        point = point * 2. / torch.tensor((width - 1, height - 1), dtype=torch.float, device=self.device) - 1
        point = point.unsqueeze(dim=0).unsqueeze(dim=2)  # [1,n,1,2]

        c1_feature = f.grid_sample(c1, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)
        c2_feature = f.grid_sample(c2, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)
        c3_feature = f.grid_sample(c3, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)
        c4_feature = f.grid_sample(c4, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)

        feature = self.cat(c1_feature, c2_feature, c3_feature, c4_feature, dim=2)
        desp = self.extractor(feature)[0]  # [n,128]

        desp = desp.detach().cpu().numpy()

        return desp

    def _generate_descriptor_for_superpoint_desp_head(self, point, desp, height, width):
        """
        构建superpoint描述子端的描述子
        """
        point = torch.from_numpy(point[:, ::-1].copy()).to(torch.float).to(self.device)
        # 归一化采样坐标到[-1,1]
        point = point * 2. / torch.tensor((width - 1, height - 1), dtype=torch.float, device=self.device) - 1
        point = point.unsqueeze(dim=0).unsqueeze(dim=2)  # [1,n,1,2]

        desp = f.grid_sample(desp, point, mode="bilinear")[0, :, :, 0].transpose(0, 1)
        desp = desp / torch.norm(desp, dim=1, keepdim=True).clamp(1e-5)

        desp = desp.detach().cpu().numpy()

        return desp

class SuperPointTrainer(MegPointTrainer):

    def __init__(self, **config):
        super(SuperPointTrainer, self).__init__(**config)

    def _initialize_model(self):
        self.logger.info("Initialize network arch {}".format(self.config['model']['backbone']))
        model = get_model(self.config['model']['backbone'])()

        if self.multi_gpus:
            model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)

    def _initialize_loss(self):
        # 初始化point loss
        self.logger.info("Initialize the CrossEntropyLoss for SuperPoint.")
        self.point_loss = torch.nn.CrossEntropyLoss(reduction="none")

        # 初始化描述子loss
        self.logger.info("Initialize the DescriptorTripletLoss for SuperPoint.")
        self.descriptor_loss = DescriptorGeneralTripletLoss(self.device)

    def _initialize_optimizer(self):
        # 初始化网络训练优化器
        self.logger.info("Initialize Adam optimizer with weight_decay: {:.5f}.".format(self.config['train']['weight_decay']))
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.config['train']['lr'],
            weight_decay=self.config['train']['weight_decay'])

    def _train_func(self, epoch_idx):
        self.model.train()

        stime = time.time()
        for i, data in enumerate(self.train_dataloader):

            image = data['image'].to(self.device)
            label = data['label'].to(self.device)
            mask = data['mask'].to(self.device)

            warped_image = data['warped_image'].to(self.device)
            warped_label = data['warped_label'].to(self.device)
            warped_mask = data['warped_mask'].to(self.device)

            desp_point = data["desp_point"].to(self.device)
            warped_desp_point = data["warped_desp_point"].to(self.device)
            valid_mask = data["valid_mask"].to(self.device)
            not_search_mask = data["not_search_mask"].to(self.device)

            shape = image.shape

            image_pair = torch.cat((image, warped_image), dim=0)
            label_pair = torch.cat((label, warped_label), dim=0)
            mask_pair = torch.cat((mask, warped_mask), dim=0)

            logit_pair, desp_pair, _ = self.model(image_pair)

            unmasked_point_loss = self.point_loss(logit_pair, label_pair)
            point_loss = self._compute_masked_loss(unmasked_point_loss, mask_pair)

            # compute descriptor loss
            desp_point_pair = torch.cat((desp_point, warped_desp_point), dim=0)
            desp_pair = f.grid_sample(desp_pair, desp_point_pair, mode="bilinear", padding_mode="border")
            desp_pair = desp_pair[:, :, :, 0].transpose(1, 2)
            desp_pair = desp_pair / torch.norm(desp_pair, dim=2, keepdim=True)
            desp_0, desp_1 = torch.chunk(desp_pair, 2, dim=0)

            desp_loss = self.descriptor_loss(desp_0, desp_1, valid_mask, not_search_mask)

            loss = point_loss + desp_loss

            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if i % self.config['train']['log_freq'] == 0:

                point_loss_val = point_loss.item()
                desp_loss_val = desp_loss.item()
                loss_val = loss.item()

                self.summary_writer.add_histogram('descriptor', desp_pair)
                self.logger.info("[Epoch:%2d][Step:%5d:%5d]: loss = %.4f, point_loss = %.4f, desp_loss = %.4f"
                                 " one step cost %.4fs. "
                                 % (epoch_idx, i, self.epoch_length, loss_val,
                                    point_loss_val, desp_loss_val,
                                    (time.time() - stime) / self.config['train']['log_freq'],
                                    ))
                stime = time.time()

        # save the model
        if self.multi_gpus:
            torch.save(self.model.module.state_dict(), os.path.join(self.config['ckpt_path'], 'model_%02d.pt' % epoch_idx))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.config['ckpt_path'], 'model_%02d.pt' % epoch_idx))

    def _inference_func(self, image_pair):
        """
        image_pair: [2,1,h,w]
        """
        self.model.eval()
        _, _, height, width = image_pair.shape
        _, desp_pair, prob_pair = self.model(image_pair)
        prob_pair = f.pixel_shuffle(prob_pair, 8)
        prob_pair = spatial_nms(prob_pair)

        # 得到对应的预测点
        prob_pair = prob_pair.detach().cpu().numpy()
        first_prob = prob_pair[0, 0]
        second_prob = prob_pair[1, 0]

        first_point, first_point_num = self._generate_predict_point(
            first_prob,
            detection_threshold=self.config['test']['detection_threshold'],
            top_k=self.config['test']['top_k'])  # [n,2]

        second_point, second_point_num = self._generate_predict_point(
            second_prob,
            detection_threshold=self.config['test']['detection_threshold'],
            top_k=self.config['test']['top_k'])  # [n,2]

        if first_point_num <= 4 or second_point_num <= 4:
            print("skip this pair because there's little point!")
            return None

        # 得到点对应的描述子
        first_desp, second_desp = torch.chunk(desp_pair, 2, dim=0)

        select_first_desp = self._generate_descriptor_for_superpoint_desp_head(first_point, first_desp, height, width)
        select_second_desp = self._generate_descriptor_for_superpoint_desp_head(second_point, second_desp, height, width)

        return first_point, first_point_num, second_point, second_point_num, select_first_desp, select_second_desp


