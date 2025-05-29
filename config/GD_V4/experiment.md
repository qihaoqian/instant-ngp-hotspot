### gd_6 and gd_5
加入dir loss 效果更不好了，把eikonal loss拉得很大，它们在对抗
### gd_2,3,4
increase proj loss weight(3->10->20) could make sdf loss smaller
使用dists*dist_gt_space，让远离表面的区域受到更大的proj loss约束，效果更好了（减缓一定的训练速度）