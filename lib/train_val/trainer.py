import os
import torch
import shutil
import numpy as np

from lib.loss.loss import mpjpe, n_mpjpe, p_mpjpe, mean_velocity_error, weighted_mpjpe, AngleLoss
from lib.dataloader.generators import ChunkedGenerator, UnchunkedGenerator
from lib.camera.camera import image_coordinates
from lib.skeleton.bone import get_bone_length_from_3d_pose, get_bone_unit_vector_from_3d_pose

import torch
import torch.nn.functional as F


def postprocess_angle(cls_logits, reg_output, num_bins=8):
    """
    将模型输出的分类和回归结果转换为真实角度
    :param cls_logits: (B, num_bins) 分类输出
    :param reg_output: (B,) 回归输出
    :return: (B,) 真实角度，范围 [0, 360)
    """
    B = cls_logits.size(0)
    bin_width = 360.0 / num_bins
    bin_width_half = bin_width / 2.0

    # 分类概率
    cls_probs = F.softmax(cls_logits, dim=1)
    bin_indices = cls_probs.argmax(dim=1).float()

    # 计算 bin 中心角度
    bin_centers = bin_indices * bin_width

    # 计算偏移量
    reg_offset = reg_output * bin_width_half

    # 计算最终角度
    final_angles = bin_centers + reg_offset
    final_angles = final_angles % 360.0  # 确保在 [0, 360)

    return final_angles


class Trainer:
    def __init__(
        self,
        data_config,
        model_config,
        train_config,
        plot_config,
        train_generator,
        test_generator,
        models,
        optimizer,
        kps_left,
        kps_right,
        joints_left,
        joints_right,
        plotter,
        best_performance,
    ):

        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config
        self.plot_config = plot_config

        self.lr = train_config["LEARNING_RATE"]
        self.optimizer = optimizer

        self.train_generator = train_generator
        self.test_generator = test_generator

        self.ori_model_train = models["train_pos"]
        self.pos_model_test = models["test_pos"]
        self.trj_model_train = models["train_trj"]
        self.trj_model_test = models["test_trj"]

        self.min_loss = 1e5 if best_performance is None else best_performance
        self.losses_ori_train = []
        self.losses_3d_valid = []
        self.min_vars = 100000
        self.min_means = 100000

        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.receptive_field = model_config["NUM_FRAMES"]

        self.plotter = plotter

        self.num_class = model_config["NUM_COARSE_ANG"]
        self.loss = AngleLoss(self.num_class, 1.0)

    @staticmethod
    def eval_data_prepare(receptive_field, inputs_2d, inputs_3d, inputs_ori):
        inputs_2d_p = torch.squeeze(inputs_2d)
        inputs_3d_p = inputs_3d.permute(1, 0, 2, 3)
        inputs_ori_p = torch.squeeze(inputs_2d)

        out_num = inputs_2d_p.shape[0] - receptive_field + 1
        eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
        for i in range(out_num):
            eval_input_2d[i, :, :, :] = inputs_2d_p[i : i + receptive_field, :, :]
        return eval_input_2d, inputs_3d_p

    def train(self, epoch, mlog):
        epoch_loss_ori_train = 0

        self.ori_model_train.train()
        if self.model_config["TRAJECTORY_MODEL"]:
            self.trj_model_train.train()

        iter = 0
        for _, batch_3d, batch_2d, batch_ori in self.train_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype("float32"))
            inputs_3d = torch.from_numpy(batch_3d.astype("float32"))
            inputs_ori = torch.from_numpy(batch_ori.astype("float32"))

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_3d = inputs_3d.cuda()
                inputs_ori = inputs_ori.cuda()

            if self.model_config["TRAJECTORY_MODEL"]:
                inputs_traj = inputs_3d[:, :, :1].clone()

            if self.data_config["RAY_ENCODING"]:
                # do nothing
                if self.model_config["TRAJECTORY_MODEL"]:
                    inputs_3d[:, :, 1:] -= inputs_3d[:, :, 0:1]
                    inputs_3d[:, :, 0] = 0
            else:
                inputs_3d[:, :, 1:] -= inputs_3d[:, :, 0:1]
                inputs_3d[:, :, 0] = 0

            self.optimizer.zero_grad()

            # Predict 3D poses
            # predicted_3d_pos = self.pos_model_train(inputs_2d)
            out_cls, out_reg = self.ori_model_train(inputs_2d)
            loss_ori = self.loss(out_cls, out_reg, inputs_ori)

            epoch_loss_ori_train += loss_ori.item()
            total_loss = loss_ori

            # if self.model_config["TRAJECTORY_MODEL"]:
            #     predicted_3d_trj = self.trj_model_train(inputs_2d)
            #     w = torch.abs(1 / inputs_traj[:, :, :, 2])  # Weight inversely proportional to depth
            #     loss_3d_traj = weighted_mpjpe(predicted_3d_trj, inputs_traj, w)
            #     assert inputs_traj.shape[0] * inputs_traj.shape[1] == inputs_3d.shape[0] * inputs_3d.shape[1]
            #     epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_traj.item()
            #     epoch_loss_3d_trj += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_traj.item()
            #     total_loss += loss_3d_traj

            # ---------------- visualization ---------------- #
            # if iter % 2048 == 0 and self.plotter is not None and epoch % 64 == 0:
            #     self.plotter.show_plot(
            #         epoch,
            #         inputs_2d.detach().cpu().numpy(),
            #         inputs_3d.detach().cpu().numpy(),
            #         dataset=self.data_config["DATASET"],
            #         gt=self.data_config["KEYPOINTS"],
            #     )
            # ---------------- visualization ---------------- #
            if iter % 1000 == 0:
                print(f"iter {iter}, total_loss {total_loss}")

            if iter % 5000 == 0:
                print(f"iter {iter}, total_loss {total_loss}")
                final_angles = postprocess_angle(out_cls, out_reg, self.num_class)
                delta_angle = (inputs_ori - final_angles + 180.0) % 360.0 - 180.0
                print(delta_angle)
            iter += 1
            total_loss.backward()
            self.optimizer.step()

        self.losses_ori_train.append(total_loss)
        torch.cuda.empty_cache()

        # if self.plotter:
        #     # plot all the losses
        #     self.plotter.log_metric("train", self.losses_ori_train[-1] * 1000, epoch)
        #     # self.plotter.log_metric("train_pos", epoch_loss_3d_pos / N * 1000, epoch)
        #     # self.plotter.log_metric("train_trj", epoch_loss_3d_trj / N * 1000, epoch)
        #     # self.plotter.log_metric("train_bone", epoch_loss_3d_bone / N * 1000, epoch)
        #
        #     # plot all the learning rates
        #     self.plotter.log_metric("lr", self.lr, epoch)

        # return the current epoch's mpjme
        return self.losses_ori_train[-1], self.lr

    def test(self, epoch, mlog):
        with torch.no_grad():
            self.pos_model_test.load_state_dict(self.ori_model_train.state_dict(), strict=True)
            self.pos_model_test.eval()

            all_ang_diff = []
            # Evaluate on test set
            for cam, batch, batch_2d, batch_ori in self.test_generator.next_epoch():
                inputs_2d = torch.from_numpy(batch_2d.astype("float32"))
                inputs_3d = torch.from_numpy(batch.astype("float32"))
                inputs_ori = torch.from_numpy(batch_ori.astype("float32"))

                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                    inputs_3d = inputs_3d.cuda()
                    inputs_ori = inputs_ori.cuda()

                if self.model_config["TRAJECTORY_MODEL"]:
                    inputs_traj = inputs_3d.clone()

                if self.data_config["RAY_ENCODING"]:
                    # do nothing
                    if self.model_config["TRAJECTORY_MODEL"]:
                        inputs_3d[:, :, 1:] -= inputs_3d[:, :, 0:1]
                        inputs_3d[:, :, 0] = 0
                else:
                    inputs_3d[:, :, 1:] -= inputs_3d[:, :, 0:1]
                    inputs_3d[:, :, 0] = 0

                out_cls, out_reg = self.pos_model_test(inputs_2d)
                final_angles = postprocess_angle(out_cls, out_reg, self.num_class)
                delta_angle = (inputs_ori.squeeze() - final_angles + 180.0) % 360.0 - 180.0
                all_ang_diff.append(delta_angle.cpu())

            # 连接所有张量为一个一维张量
            concatenated = torch.cat(all_ang_diff)

            # 计算均值和方差
            min_diff = torch.min(torch.abs(concatenated))
            max_diff = torch.max(torch.abs(concatenated))
            mean = torch.mean(concatenated)
            variance = torch.var(concatenated)

            # Save checkpoint if necessary
            if epoch % self.train_config["CHECKPOINT_FREQUENCY"] == 0:
                chk_path = os.path.join(self.train_config["CHECKPOINT"], "epoch_{}.bin".format(epoch))
                mlog.info("Saving epochs {}'s checkpoint to {}.".format(epoch, chk_path))
                if self.model_config["TRAJECTORY_MODEL"]:
                    torch.save(
                        {
                            "epoch": epoch,
                            "lr": self.lr,
                            "performance": {"mean": mean, "var": variance, "min": min_diff, "max": max_diff},
                            "random_state": self.train_generator.random_state(),
                            "optimizer": self.optimizer.state_dict(),
                            "model_pos": self.ori_model_train.state_dict(),
                            "model_trj": self.trj_model_train.state_dict(),
                        },
                        chk_path,
                    )
                else:
                    torch.save(
                        {
                            "epoch": epoch,
                            "lr": self.lr,
                            "performance": {"mean": mean, "var": variance, "min": min_diff, "max": max_diff},
                            "random_state": self.train_generator.random_state(),
                            "optimizer": self.optimizer.state_dict(),
                            "model_pos": self.ori_model_train.state_dict(),
                        },
                        chk_path,
                    )

                #### save best checkpoint
                best_chk_path = os.path.join(self.train_config["CHECKPOINT"], "best_epoch.bin".format(epoch))
                if mean < self.min_means:
                    self.min_means = mean
                    self.min_vars = variance
                    mlog.info(
                        "Saving best checkpoint to {} with mean var: {}.".format(
                            best_chk_path, self.min_means, self.min_vars
                        )
                    )
                    shutil.copy(chk_path, best_chk_path)

                cmd = "rm {}".format(chk_path)
                os.system(cmd)

            # Decay learning rate exponentially
            self.lr *= self.train_config["LR_DECAY"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.train_config["LR_DECAY"]

            # Decay BatchNorm momentum
            if self.model_config["MODEL"] == "VideoPose3D":
                momentum = self.train_config["INITIAL_MOMENTUM"] * np.exp(
                    -(epoch - 1)
                    / self.train_config["EPOCHS"]
                    * np.log(self.train_config["INITIAL_MOMENTUM"] / self.train_config["FINAL_MOMENTUM"])
                )
                self.ori_model_train.module.set_bn_momentum(momentum)
                if self.model_config["TRAJECTORY_MODEL"]:
                    self.trj_model_train.module.set_bn_momentum(momentum)

        if self.plotter:
            # plot all the losses
            self.plotter.log_metric("test", self.min_means, epoch)
            # self.plotter.log_metric("test_pos", epoch_loss_3d_pos / N * 1000, epoch)
            # self.plotter.log_metric("test_trj", epoch_loss_3d_trj / N * 1000, epoch)
            # self.plotter.log_metric("test_bone", epoch_loss_3d_bone / N * 1000, epoch)

        # return the current epoch's mpjme
        return self.min_means

    def evaluate_core(self, test_generator, action=None, flip_test=False):

        epoch_loss_oris = 0

        all_ang_diff = []
        with torch.no_grad():
            self.pos_model_test.eval()
            for cam, batch, batch_2d, batch_ori in test_generator.next_epoch():
                inputs_2d = torch.from_numpy(batch_2d.astype("float32"))
                inputs_3d = torch.from_numpy(batch.astype("float32"))
                inputs_ori = torch.from_numpy(batch_ori.astype("float32"))

                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                    inputs_3d = inputs_3d.cuda()
                    inputs_ori = inputs_ori.cuda()

                out_cls, out_reg = self.pos_model_test(inputs_2d)

                final_angles = postprocess_angle(out_cls, out_reg, self.num_class)
                delta_angle = (inputs_ori.squeeze() - final_angles + 180.0) % 360.0 - 180.0
                all_ang_diff.append(delta_angle.cpu())

        # 连接所有张量为一个一维张量
        concatenated = torch.cat(all_ang_diff)

        # 计算均值和方差
        min_diff = torch.min(torch.abs(concatenated))
        max_diff = torch.max(torch.abs(concatenated))
        mean = torch.mean(concatenated)
        variance = torch.var(concatenated)

        return mean, variance, min_diff, max_diff

    def evaluate(self, mlog, subjects_test, pose_data, action_filter, pad, causal_shift, epoch, plot=False):

        all_actions = dict()
        for subject in subjects_test:
            # all_actions.setdefault('Sitting 1', list()).append((subject, 'Sitting 1'))
            if action_filter == None:
                action_keys = pose_data.get_dataset()[subject].keys()
            else:
                action_keys = action_filter
            for action in action_keys:
                all_actions.setdefault(action.split(" ")[0], list()).append((subject, action))

        print(all_actions)
        errors_means_diff = []

        for action_key in all_actions.keys():
            poses_cam, poses_act_3d, poses_2d_act, poses_ori_act = pose_data.fetch_via_action(all_actions[action_key])
            action_generator = ChunkedGenerator(
                self.train_config["BATCH_SIZE"] // self.data_config["STRIDE"],
                poses_cam,
                poses_act_3d,
                poses_2d_act,
                poses_ori_act,
                self.data_config["STRIDE"],
                pad=pad,
                causal_shift=causal_shift,
                shuffle=True,
                augment=False,
                kps_left=self.kps_left,
                kps_right=self.kps_right,
                joints_left=self.joints_left,
                joints_right=self.joints_right,
            )
            if action_key is None:
                mlog.info("----------")
            else:
                mlog.info("----" + action_key + "----")
            mean, variance, min_diff, max_diff = self.evaluate_core(
                action_generator, action_key, flip_test=self.train_config["TEST_TIME_AUGMENTATION"]
            )
            mlog.info("ang mean     {} °".format(mean))
            mlog.info("ang var      {} °".format(variance))
            mlog.info("ang min      {} °".format(min_diff))
            mlog.info("ang max      {} °".format(max_diff))
            mlog.info("----------")
            errors_means_diff.append(mean)

        mlog.info("ang diff  action-wise average: {} °".format(round(np.mean(errors_means_diff), 1)))
